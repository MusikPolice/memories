"""Database initialisation and repository functions.

All functions accept an aiosqlite.Connection as their first argument.
init_db() must be called on every connection before any other function is used;
it sets row_factory and creates all tables.
"""

from __future__ import annotations

import json
import struct
from typing import Any

import aiosqlite

from memories.exceptions import NotFoundError
from memories.models import (
    Character,
    Decision,
    Experience,
    Inference,
    Message,
    Session,
)
from memories.schema_loader import apply_mask

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

CREATE TABLE IF NOT EXISTS character_facts (
    character_id INTEGER PRIMARY KEY REFERENCES characters(id),
    facts_json   TEXT NOT NULL DEFAULT '{}',
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inferences (
    id                    INTEGER PRIMARY KEY,
    character_id          INTEGER REFERENCES characters(id),
    statement             TEXT NOT NULL,
    derivation            TEXT NOT NULL,
    source_inference_ids  TEXT,
    source_fact_paths     TEXT,
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
CREATE INDEX IF NOT EXISTS idx_experiences_character_embedding
    ON experiences(character_id) WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY,
    character_id INTEGER REFERENCES characters(id),
    session_id   INTEGER REFERENCES sessions(id),
    turn_id      INTEGER,
    pass_name    TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    tool_args    TEXT NOT NULL,
    user_input   TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_decisions_session_turn
    ON decisions(session_id, turn_id);

CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY,
    character_id INTEGER REFERENCES characters(id),
    session_id   INTEGER REFERENCES sessions(id),
    role         TEXT NOT NULL,
    content      TEXT NOT NULL,
    turn_id      INTEGER,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
    return Message.model_validate(_row(row))


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
# Character facts blob (new schema-constrained fact store)
# ---------------------------------------------------------------------------


async def get_facts(db: aiosqlite.Connection, character_id: int) -> dict[str, Any]:
    """Return the schema-masked fact blob for character_id, or {} if none exists."""
    row = await (
        await db.execute(
            "SELECT facts_json FROM character_facts WHERE character_id = ?",
            (character_id,),
        )
    ).fetchone()
    if row is None:
        return {}
    blob: dict[str, Any] = json.loads(row[0])
    return apply_mask(blob)


async def set_facts(db: aiosqlite.Connection, character_id: int, blob: dict[str, Any]) -> None:
    """Write blob as the facts_json for character_id, replacing any existing row."""
    await db.execute(
        """INSERT INTO character_facts (character_id, facts_json, updated_at)
           VALUES (?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(character_id) DO UPDATE SET
               facts_json = excluded.facts_json,
               updated_at = excluded.updated_at""",
        (character_id, json.dumps(blob)),
    )
    await db.commit()


async def patch_fact(
    db: aiosqlite.Connection,
    character_id: int,
    path_tuple: tuple[str, ...],
    value: str | int | float | bool | None,
) -> None:
    """Set a single leaf in the character's fact blob to {"Value": value}.

    Creates intermediate grouping dicts as needed. Non-destructive to other paths.
    """
    blob = await get_facts(db, character_id)
    node: dict[str, Any] = blob
    for key in path_tuple[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[path_tuple[-1]] = {"Value": value}
    await set_facts(db, character_id, blob)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def create_session(db: aiosqlite.Connection, *, character_id: int) -> Session:
    cursor = await db.execute(
        "INSERT INTO sessions (character_id) VALUES (?)",
        (character_id,),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    session_id = cursor.lastrowid

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
# Messages
# ---------------------------------------------------------------------------


async def store_message(
    db: aiosqlite.Connection,
    *,
    session_id: int,
    character_id: int,
    role: str,
    content: str,
    turn_id: int,
) -> Message:
    cursor = await db.execute(
        """INSERT INTO messages
               (character_id, session_id, role, content, turn_id)
           VALUES (?, ?, ?, ?, ?)""",
        (character_id, session_id, role, content, turn_id),
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
    """Replace content on the assistant message for the given (session_id, turn_id)."""
    cursor = await db.execute(
        "UPDATE messages SET content = ? "
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
    d["tool_args"] = json.loads(d["tool_args"])
    if d.get("user_input") is not None:
        d["user_input"] = json.loads(d["user_input"])
    return Decision.model_validate(d)


async def store_decision(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    session_id: int,
    turn_id: int,
    pass_name: str,
    tool_name: str,
    tool_args: dict[str, Any],
    user_input: dict[str, Any] | None = None,
) -> Decision:
    user_input_json = json.dumps(user_input) if user_input is not None else None
    cursor = await db.execute(
        """INSERT INTO decisions
               (character_id, session_id, turn_id, pass_name, tool_name, tool_args, user_input)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            character_id,
            session_id,
            turn_id,
            pass_name,
            tool_name,
            json.dumps(tool_args),
            user_input_json,
        ),
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
        "SELECT * FROM decisions WHERE session_id = ? ORDER BY turn_id DESC, id DESC",
        (session_id,),
    )
    rows = await cursor.fetchall()
    return [_parse_decision(r) for r in rows]


# ---------------------------------------------------------------------------
# Inferences
# ---------------------------------------------------------------------------


def _parse_inference(row: aiosqlite.Row) -> Inference:
    d = _row(row)
    d["source_inference_ids"] = (
        json.loads(d["source_inference_ids"]) if d.get("source_inference_ids") else []
    )
    d["source_fact_paths"] = (
        json.loads(d["source_fact_paths"]) if d.get("source_fact_paths") else []
    )
    return Inference.model_validate(d)


async def create_inference(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    statement: str,
    derivation: str,
    source_inference_ids: list[int] | None = None,
    source_fact_paths: list[str] | None = None,
    depth: int = 1,
    inference_type: str = "logical",
) -> Inference:
    inf_ids_json = json.dumps(source_inference_ids or [])
    paths_json = json.dumps(source_fact_paths or [])
    cursor = await db.execute(
        """INSERT INTO inferences
               (character_id, statement, derivation, source_inference_ids,
                source_fact_paths, depth, inference_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            character_id,
            statement,
            derivation,
            inf_ids_json,
            paths_json,
            depth,
            inference_type,
        ),
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
    return struct.pack(f"{len(embedding)}d", *embedding)


def _blob_to_embedding(blob: bytes) -> list[float]:
    if blob[:1] == b"[":  # legacy JSON-encoded blob — transparent migration path
        result: list[float] = json.loads(blob.decode())
        return result
    n = len(blob) // 8
    return list(struct.unpack(f"{n}d", blob))


# Module-level embedding cache: character_id → {experience_id → (Experience, vec)}
# Populated lazily on first get_experiences_with_embeddings call per character.
# Write-through on create (only when already loaded); evicted by ID on delete.
_experience_embedding_cache: dict[int, dict[int, tuple[Experience, list[float]]]] = {}


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
    exp = _parse_experience(row)
    if character_id in _experience_embedding_cache:
        _experience_embedding_cache[character_id][exp.id] = (exp, _blob_to_embedding(embedding))
    return exp


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
    if character_id in _experience_embedding_cache:
        return list(_experience_embedding_cache[character_id].values())
    cursor = await db.execute(
        "SELECT * FROM experiences"
        " WHERE character_id = ? AND embedding IS NOT NULL"
        " ORDER BY created_at",
        (character_id,),
    )
    rows = await cursor.fetchall()
    result: list[tuple[Experience, list[float]]] = []
    for row in rows:
        d = _row(row)
        blob: bytes = d.pop("embedding")
        exp = Experience.model_validate(d)
        vec = _blob_to_embedding(blob)
        result.append((exp, vec))
    _experience_embedding_cache[character_id] = {exp.id: (exp, vec) for exp, vec in result}
    return result


async def delete_experience(db: aiosqlite.Connection, experience_id: int) -> None:
    cursor = await db.execute("DELETE FROM experiences WHERE id = ?", (experience_id,))
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Experience {experience_id} not found")
    for char_cache in _experience_embedding_cache.values():
        char_cache.pop(experience_id, None)


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
