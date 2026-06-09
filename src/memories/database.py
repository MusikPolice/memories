"""Database initialisation and repository functions.

All functions accept an aiosqlite.Connection as their first argument.
init_db() must be called on every connection before any other function is used;
it sets row_factory and creates all tables.
"""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from memories.exceptions import NotFoundError
from memories.models import (
    Character,
    Decision,
    Experience,
    Fact,
    Inference,
    Message,
    Segment,
    Session,
)

# ---------------------------------------------------------------------------
# Full schema — created once at startup; all tables present from Phase 1
# so later phases require no migrations.
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS characters (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    modelfile_base   TEXT NOT NULL,
    current_model_name TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id               INTEGER PRIMARY KEY,
    character_id     INTEGER REFERENCES characters(id),
    started_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at         TIMESTAMP,
    closing_journal  TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_character ON sessions(character_id);

CREATE TABLE IF NOT EXISTS facts (
    id               INTEGER PRIMARY KEY,
    character_id     INTEGER REFERENCES characters(id),
    key              TEXT NOT NULL,
    value            TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT 'character',
    mutability       TEXT NOT NULL DEFAULT 'immutable',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(character_id, category, key)
);
CREATE INDEX IF NOT EXISTS idx_facts_character ON facts(character_id);

CREATE TABLE IF NOT EXISTS inferences (
    id                    INTEGER PRIMARY KEY,
    character_id          INTEGER REFERENCES characters(id),
    statement             TEXT NOT NULL,
    derivation            TEXT NOT NULL,
    source_fact_ids       TEXT,
    source_inference_ids  TEXT,
    depth                 INTEGER NOT NULL DEFAULT 1,
    inference_type        TEXT NOT NULL DEFAULT 'logical',
    status                TEXT NOT NULL DEFAULT 'active',
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_inferences_character_status
    ON inferences(character_id, status);

CREATE TABLE IF NOT EXISTS experiences (
    id               INTEGER PRIMARY KEY,
    character_id     INTEGER REFERENCES characters(id),
    session_id       INTEGER REFERENCES sessions(id),
    statement        TEXT NOT NULL,
    source           TEXT NOT NULL,
    embedding        BLOB,
    approved_at      TIMESTAMP NOT NULL,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_experiences_character ON experiences(character_id);

CREATE TABLE IF NOT EXISTS decisions (
    id               INTEGER PRIMARY KEY,
    character_id     INTEGER REFERENCES characters(id),
    session_id       INTEGER REFERENCES sessions(id),
    turn_id          INTEGER,
    reasoning        TEXT NOT NULL,
    verdict          TEXT NOT NULL,
    violations       TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decisions_session_turn
    ON decisions(session_id, turn_id);

CREATE TABLE IF NOT EXISTS segments (
    id               INTEGER PRIMARY KEY,
    session_id       INTEGER REFERENCES sessions(id),
    start_turn       INTEGER NOT NULL,
    end_turn         INTEGER,
    boundary_reason  TEXT,
    status           TEXT NOT NULL DEFAULT 'verbatim',
    journal_text     TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id);

CREATE TABLE IF NOT EXISTS messages (
    id                       INTEGER PRIMARY KEY,
    character_id             INTEGER REFERENCES characters(id),
    session_id               INTEGER REFERENCES sessions(id),
    segment_id               INTEGER REFERENCES segments(id),
    role                     TEXT NOT NULL,
    content                  TEXT NOT NULL,
    turn_id                  INTEGER,
    captured_by              TEXT,
    ungrounded_implications  TEXT,
    created_at               TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session_turn
    ON messages(session_id, turn_id);
"""


async def init_db(db: aiosqlite.Connection) -> None:
    """Create all tables and set the row factory on *db*."""
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    await db.executescript(_DDL)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row(row: aiosqlite.Row) -> dict[str, Any]:
    # sqlite3.Row iterates over values, not keys; use zip to build a named dict.
    return dict(zip(row.keys(), tuple(row), strict=True))


def _parse_message(row: aiosqlite.Row) -> Message:
    d = _row(row)
    if d.get("captured_by"):
        d["captured_by"] = json.loads(d["captured_by"])
    if d.get("ungrounded_implications"):
        d["ungrounded_implications"] = json.loads(d["ungrounded_implications"])
    return Message.model_validate(d)


# ---------------------------------------------------------------------------
# Characters
# ---------------------------------------------------------------------------


async def create_character(
    db: aiosqlite.Connection, *, name: str, modelfile_base: str
) -> Character:
    cursor = await db.execute(
        "INSERT INTO characters (name, modelfile_base) VALUES (?, ?)",
        (name, modelfile_base),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    row = await (
        await db.execute("SELECT * FROM characters WHERE id = ?", (cursor.lastrowid,))
    ).fetchone()
    assert row is not None
    return Character.model_validate(_row(row))


async def get_character(db: aiosqlite.Connection, character_id: int) -> Character | None:
    row = await (
        await db.execute("SELECT * FROM characters WHERE id = ?", (character_id,))
    ).fetchone()
    return Character.model_validate(_row(row)) if row else None


async def list_characters(db: aiosqlite.Connection) -> list[Character]:
    cursor = await db.execute("SELECT * FROM characters ORDER BY id")
    rows = await cursor.fetchall()
    return [Character.model_validate(_row(r)) for r in rows]


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


async def create_fact(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    key: str,
    value: str,
    category: str = "character",
    mutability: str = "immutable",
) -> Fact:
    cursor = await db.execute(
        "INSERT INTO facts (character_id, key, value, category, mutability) VALUES (?, ?, ?, ?, ?)",
        (character_id, key, value, category, mutability),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    row = await (
        await db.execute("SELECT * FROM facts WHERE id = ?", (cursor.lastrowid,))
    ).fetchone()
    assert row is not None
    return Fact.model_validate(_row(row))


async def get_facts(db: aiosqlite.Connection, character_id: int) -> list[Fact]:
    cursor = await db.execute(
        "SELECT * FROM facts WHERE character_id = ? ORDER BY id",
        (character_id,),
    )
    rows = await cursor.fetchall()
    return [Fact.model_validate(_row(r)) for r in rows]


async def get_fact(db: aiosqlite.Connection, character_id: int, fact_id: int) -> Fact | None:
    """Return a single fact owned by character_id, or None if not found."""
    row = await (
        await db.execute(
            "SELECT * FROM facts WHERE id = ? AND character_id = ?",
            (fact_id, character_id),
        )
    ).fetchone()
    return Fact.model_validate(_row(row)) if row is not None else None


async def update_fact(
    db: aiosqlite.Connection,
    *,
    fact_id: int,
    value: str,
    category: str | None = None,
    mutability: str | None = None,
) -> Fact:
    updates = ["value = ?"]
    params: list[Any] = [value]
    if category is not None:
        updates.append("category = ?")
        params.append(category)
    if mutability is not None:
        updates.append("mutability = ?")
        params.append(mutability)

    params.append(fact_id)
    cursor = await db.execute(
        f"UPDATE facts SET {', '.join(updates)} WHERE id = ?",  # nosec B608
        tuple(params),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Fact {fact_id} not found")
    row = await (await db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))).fetchone()
    assert row is not None
    return Fact.model_validate(_row(row))


async def delete_fact(
    db: aiosqlite.Connection,
    *,
    fact_id: int,
) -> None:
    cursor = await db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Fact {fact_id} not found")


async def patch_fact(
    db: aiosqlite.Connection,
    *,
    fact_id: int,
    category: str | None = None,
    mutability: str | None = None,
) -> Fact:
    if category is None and mutability is None:
        raise ValueError("At least one of category or mutability must be provided")
    updates: list[str] = []
    params: list[Any] = []
    if category is not None:
        updates.append("category = ?")
        params.append(category)
    if mutability is not None:
        updates.append("mutability = ?")
        params.append(mutability)
    params.append(fact_id)
    cursor = await db.execute(
        f"UPDATE facts SET {', '.join(updates)} WHERE id = ?",  # nosec B608
        tuple(params),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Fact {fact_id} not found")
    row = await (await db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))).fetchone()
    assert row is not None
    return Fact.model_validate(_row(row))


async def get_fact_by_category_key(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    category: str,
    key: str,
) -> Fact | None:
    row = await (
        await db.execute(
            "SELECT * FROM facts WHERE character_id = ? AND category = ? AND key = ?",
            (character_id, category, key),
        )
    ).fetchone()
    return Fact.model_validate(_row(row)) if row else None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def create_session(db: aiosqlite.Connection, *, character_id: int) -> Session:
    cursor = await db.execute(
        "INSERT INTO sessions (character_id) VALUES (?)",
        (character_id,),
    )
    assert cursor.lastrowid is not None
    session_id = cursor.lastrowid

    # Every session begins with a single verbatim segment.
    await db.execute(
        "INSERT INTO segments (session_id, start_turn, boundary_reason) VALUES (?, ?, ?)",
        (session_id, 1, "session_start"),
    )
    await db.commit()

    row = await (await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))).fetchone()
    assert row is not None
    return Session.model_validate(_row(row))


async def get_session(db: aiosqlite.Connection, session_id: int) -> Session | None:
    row = await (await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))).fetchone()
    return Session.model_validate(_row(row)) if row else None


async def end_session(db: aiosqlite.Connection, session_id: int) -> Session:
    cursor = await db.execute(
        "UPDATE sessions SET ended_at = CURRENT_TIMESTAMP WHERE id = ?",
        (session_id,),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Session {session_id} not found")
    row = await (await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))).fetchone()
    assert row is not None
    return Session.model_validate(_row(row))


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


async def get_active_segment(db: aiosqlite.Connection, session_id: int) -> Segment:
    """Return the open (end_turn IS NULL) segment for *session_id*."""
    row = await (
        await db.execute(
            "SELECT * FROM segments WHERE session_id = ? AND end_turn IS NULL "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
    ).fetchone()
    if row is None:
        raise NotFoundError(f"No active segment for session {session_id}")
    return Segment.model_validate(_row(row))


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


async def store_message(
    db: aiosqlite.Connection,
    *,
    session_id: int,
    segment_id: int,
    character_id: int,
    role: str,
    content: str,
    turn_id: int,
    ungrounded_implications: list[dict[str, Any]] | None = None,
) -> Message:
    ungrounded_json = (
        json.dumps(ungrounded_implications) if ungrounded_implications is not None else None
    )
    cursor = await db.execute(
        """INSERT INTO messages
               (character_id, session_id, segment_id, role, content, turn_id,
                ungrounded_implications)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (character_id, session_id, segment_id, role, content, turn_id, ungrounded_json),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    row = await (
        await db.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,))
    ).fetchone()
    assert row is not None
    return _parse_message(row)


async def get_messages(db: aiosqlite.Connection, session_id: int) -> list[Message]:
    cursor = await db.execute(
        "SELECT * FROM messages WHERE session_id = ? ORDER BY turn_id, id",
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_parse_message(r) for r in rows]


async def next_turn_id(db: aiosqlite.Connection, session_id: int) -> int:
    """Return the next turn_id for *session_id* (1 if no messages exist yet)."""
    row = await (
        await db.execute(
            "SELECT MAX(turn_id) FROM messages WHERE session_id = ?",
            (session_id,),
        )
    ).fetchone()
    max_id: int | None = row[0] if row else None
    return (max_id or 0) + 1


async def replace_message_content(
    db: aiosqlite.Connection,
    *,
    session_id: int,
    turn_id: int,
    new_content: str,
) -> Message:
    """Replace content and clear ungrounded_implications on the assistant message."""
    cursor = await db.execute(
        "UPDATE messages SET content = ?, ungrounded_implications = NULL "
        "WHERE session_id = ? AND turn_id = ? AND role = 'assistant'",
        (new_content, session_id, turn_id),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"No assistant message for session {session_id} turn {turn_id}")
    row = await (
        await db.execute(
            "SELECT * FROM messages WHERE session_id = ? AND turn_id = ? AND role = 'assistant'",
            (session_id, turn_id),
        )
    ).fetchone()
    assert row is not None
    return _parse_message(row)


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def _parse_decision(row: aiosqlite.Row) -> Decision:
    d = _row(row)
    if d.get("violations"):
        d["violations"] = json.loads(d["violations"])
    return Decision.model_validate(d)


async def store_decision(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    session_id: int,
    turn_id: int,
    reasoning: str,
    verdict: str,
    violations: list[dict[str, Any]] | None = None,
) -> Decision:
    violations_json = json.dumps(violations) if violations is not None else None
    cursor = await db.execute(
        """INSERT INTO decisions (character_id, session_id, turn_id, reasoning, verdict, violations)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (character_id, session_id, turn_id, reasoning, verdict, violations_json),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    row = await (
        await db.execute("SELECT * FROM decisions WHERE id = ?", (cursor.lastrowid,))
    ).fetchone()
    assert row is not None
    return _parse_decision(row)


async def get_decisions(db: aiosqlite.Connection, session_id: int) -> list[Decision]:
    cursor = await db.execute(
        "SELECT * FROM decisions WHERE session_id = ? ORDER BY turn_id DESC",
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_parse_decision(r) for r in rows]


# ---------------------------------------------------------------------------
# Inferences
# ---------------------------------------------------------------------------


def _parse_inference(row: aiosqlite.Row) -> Inference:
    d = _row(row)
    if d.get("source_fact_ids"):
        d["source_fact_ids"] = json.loads(d["source_fact_ids"])
    else:
        d["source_fact_ids"] = []
    if d.get("source_inference_ids"):
        d["source_inference_ids"] = json.loads(d["source_inference_ids"])
    else:
        d["source_inference_ids"] = []
    return Inference.model_validate(d)


async def create_inference(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    statement: str,
    derivation: str,
    source_fact_ids: list[int] | None = None,
    source_inference_ids: list[int] | None = None,
    depth: int = 1,
    inference_type: str = "logical",
) -> Inference:
    fact_ids_json = json.dumps(source_fact_ids or [])
    inf_ids_json = json.dumps(source_inference_ids or [])
    cursor = await db.execute(
        """INSERT INTO inferences
               (character_id, statement, derivation, source_fact_ids, source_inference_ids,
                depth, inference_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (character_id, statement, derivation, fact_ids_json, inf_ids_json, depth, inference_type),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    row = await (
        await db.execute("SELECT * FROM inferences WHERE id = ?", (cursor.lastrowid,))
    ).fetchone()
    assert row is not None
    return _parse_inference(row)


async def get_inferences(
    db: aiosqlite.Connection, character_id: int, status: str = "active"
) -> list[Inference]:
    if status == "all":
        cursor = await db.execute(
            "SELECT * FROM inferences WHERE character_id = ? ORDER BY id",
            (character_id,),
        )
    else:
        cursor = await db.execute(
            "SELECT * FROM inferences WHERE character_id = ? AND status = ? ORDER BY id",
            (character_id, status),
        )
    rows = await cursor.fetchall()
    return [_parse_inference(r) for r in rows]


async def get_inference(db: aiosqlite.Connection, inference_id: int) -> Inference | None:
    row = await (
        await db.execute("SELECT * FROM inferences WHERE id = ?", (inference_id,))
    ).fetchone()
    return _parse_inference(row) if row else None


async def update_inference_status(
    db: aiosqlite.Connection, inference_id: int, new_status: str
) -> Inference:
    cursor = await db.execute(
        "UPDATE inferences SET status = ? WHERE id = ?",
        (new_status, inference_id),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Inference {inference_id} not found")
    row = await (
        await db.execute("SELECT * FROM inferences WHERE id = ?", (inference_id,))
    ).fetchone()
    assert row is not None
    return _parse_inference(row)


async def delete_inference(db: aiosqlite.Connection, inference_id: int) -> None:
    cursor = await db.execute(
        "DELETE FROM inferences WHERE id = ?",
        (inference_id,),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Inference {inference_id} not found")


# ---------------------------------------------------------------------------
# Experiences
# ---------------------------------------------------------------------------


def _embedding_to_blob(embedding: list[float]) -> bytes:
    return json.dumps(embedding).encode()


def _blob_to_embedding(blob: bytes) -> list[float]:
    result: list[float] = json.loads(blob.decode())
    return result


def _parse_experience(row: aiosqlite.Row) -> Experience:
    d = _row(row)
    d.pop("embedding", None)
    return Experience.model_validate(d)


async def create_experience(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    session_id: int,
    statement: str,
    source: str,
    embedding: bytes,
) -> Experience:
    cursor = await db.execute(
        """INSERT INTO experiences
               (character_id, session_id, statement, source, embedding, approved_at)
           VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
        (character_id, session_id, statement, source, embedding),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    row = await (
        await db.execute("SELECT * FROM experiences WHERE id = ?", (cursor.lastrowid,))
    ).fetchone()
    assert row is not None
    return _parse_experience(row)


async def get_experience(db: aiosqlite.Connection, experience_id: int) -> Experience | None:
    row = await (
        await db.execute("SELECT * FROM experiences WHERE id = ?", (experience_id,))
    ).fetchone()
    return _parse_experience(row) if row else None


async def get_experiences(db: aiosqlite.Connection, character_id: int) -> list[Experience]:
    cursor = await db.execute(
        "SELECT * FROM experiences WHERE character_id = ? ORDER BY created_at",
        (character_id,),
    )
    rows = await cursor.fetchall()
    return [_parse_experience(r) for r in rows]


async def get_experiences_with_embeddings(
    db: aiosqlite.Connection, character_id: int
) -> list[tuple[Experience, list[float]]]:
    cursor = await db.execute(
        "SELECT * FROM experiences"
        " WHERE character_id = ? AND embedding IS NOT NULL"
        " ORDER BY created_at",
        (character_id,),
    )
    rows = await cursor.fetchall()
    result = []
    for row in rows:
        d = _row(row)
        blob: bytes = d.pop("embedding")
        exp = Experience.model_validate(d)
        vec = _blob_to_embedding(blob)
        result.append((exp, vec))
    return result


async def delete_experience(db: aiosqlite.Connection, experience_id: int) -> None:
    cursor = await db.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Experience {experience_id} not found")


async def update_session_closing_journal(
    db: aiosqlite.Connection, session_id: int, closing_journal: str
) -> Session:
    cursor = await db.execute(
        "UPDATE sessions SET closing_journal = ? WHERE id = ?",
        (closing_journal, session_id),
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Session {session_id} not found")
    row = await (await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))).fetchone()
    assert row is not None
    return Session.model_validate(_row(row))


async def get_previous_session(
    db: aiosqlite.Connection, character_id: int, before_session_id: int
) -> Session | None:
    row = await (
        await db.execute(
            """SELECT * FROM sessions
               WHERE character_id = ?
                 AND id < ?
                 AND closing_journal IS NOT NULL
               ORDER BY id DESC LIMIT 1""",
            (character_id, before_session_id),
        )
    ).fetchone()
    return Session.model_validate(_row(row)) if row else None
