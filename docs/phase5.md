# Phase 5 Implementation Plan — Experiences and Session Memory

## Goals (from plan.md)

- Implement the end-of-session evaluator pass: a single LLM call that writes a closing journal entry and proposes a list of Experiences for user review
- Store the closing journal in `sessions.closing_journal`
- Surface proposed Experiences in the sidechannel; user can accept, edit, or discard each proposal; accepted Experiences are embedded and written to DB; discarded proposals are simply ignored
- Embed approved Experiences via `nomic-embed-text` (Ollama `/api/embed`) and store the raw embedding vector as a BLOB in the `experiences` table
- Per-turn retrieval: embed the current user message, query the DB for semantically similar Experiences, add newly-retrieved ones to the session's active set, inject the full active set into the system prompt for every subsequent turn
- Cold start: at the first turn of a new session, embed the previous session's closing journal and use it as the retrieval query to seed the active experience set before the user speaks
- Surface source annotation ("told by user" vs. "observed") on each Experience in the sidechannel
- Handle the `experience_update` evaluator verdict: deliver the response, delete the contradicted Experience from the DB and from the session active set, emit a sidechannel notification
- Remove the `experience_update` → `pass` coercion in `run_evaluator`

**Deliverable:** Characters accumulate episodic memory across sessions. A character that learned the user lives in Chicago in one session remembers it the next. Things observed in conversation — emotional tone, topics the user deflects — are captured and inform future roleplay without the user having to re-state them.

---

## Task List

### T1 — Model: Add `Experience`

**`src/memories/models/__init__.py`**

Add the `Experience` Pydantic model alongside the existing models. The `embedding` BLOB is not included — it lives only in the DB and is never serialised in API responses.

```python
from typing import Literal

class Experience(BaseModel):
    id: int
    character_id: int
    session_id: int
    statement: str
    source: Literal["told_by_user", "observed"]
    approved_at: datetime
    created_at: datetime
```

Also update the `database.py` import list to include `Experience`.

Note: `Session` in `models/__init__.py` already has `closing_journal: str | None = None` — no model change needed.

---

### T2 — Ollama Client: Add `embed()`

**`src/memories/services/ollama_client.py`**

Add a new `embed` method to `OllamaClient`. The Ollama embedding endpoint is `POST /api/embed`.

```python
async def embed(self, model: str, text: str) -> list[float]:
    """Embed *text* using *model* and return the embedding vector.

    Uses POST /api/embed.  Raises OllamaConnectionError or OllamaResponseError
    on failure.  Returns the first (and only) embedding from the response.
    """
    url = f"{self.base_url}/api/embed"
    payload = {"model": model, "input": text}
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        ) as http:
            response = await http.post(url, json=payload)
            if response.status_code != 200:
                raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")
            data = response.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise OllamaResponseError("Ollama embed response contained no embeddings")
            return embeddings[0]
    except httpx.ConnectError as exc:
        raise OllamaConnectionError(str(exc)) from exc
```

The Ollama embed endpoint accepts `input` as either a string or a list of strings and returns `{"embeddings": [[float, ...]]}`. We always send a single string and take `embeddings[0]`.

---

### T3 — Database: Experience repository functions

**`src/memories/database.py`**

The `experiences` table already exists in `_DDL`. Add four new repository functions.

**T3.1 — `_parse_experience`**

```python
def _parse_experience(row: aiosqlite.Row) -> Experience:
    d = _row(row)
    d.pop("embedding", None)   # exclude BLOB from model validation
    return Experience.model_validate(d)
```

**T3.2 — `create_experience`**

```python
async def create_experience(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    session_id: int,
    statement: str,
    source: str,
    embedding: bytes,
    approved_at: str | None = None,
) -> Experience:
    ts = approved_at or "CURRENT_TIMESTAMP"
    cursor = await db.execute(
        f"""INSERT INTO experiences
               (character_id, session_id, statement, source, embedding, approved_at)
           VALUES (?, ?, ?, ?, ?, {ts if approved_at else 'CURRENT_TIMESTAMP'})""",
        (character_id, session_id, statement, source, embedding),
    )
    await db.commit()
    assert cursor.lastrowid is not None
    row = await (
        await db.execute("SELECT * FROM experiences WHERE id = ?", (cursor.lastrowid,))
    ).fetchone()
    assert row is not None
    return _parse_experience(row)
```

Simplified implementation:

```python
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
```

**T3.3 — `get_experiences`**

Returns all approved Experiences for a character, ordered by creation date. The `embedding` BLOB is excluded from the returned `Experience` objects (the model doesn't include it).

```python
async def get_experiences(db: aiosqlite.Connection, character_id: int) -> list[Experience]:
    cursor = await db.execute(
        "SELECT * FROM experiences WHERE character_id = ? ORDER BY created_at",
        (character_id,),
    )
    rows = await cursor.fetchall()
    return [_parse_experience(r) for r in rows]
```

**T3.4 — `get_experiences_with_embeddings`**

Used by the similarity-retrieval code. Returns `(Experience, embedding_vector)` pairs. Only fetches experiences that have a non-NULL embedding.

```python
async def get_experiences_with_embeddings(
    db: aiosqlite.Connection, character_id: int
) -> list[tuple[Experience, list[float]]]:
    cursor = await db.execute(
        "SELECT * FROM experiences WHERE character_id = ? AND embedding IS NOT NULL ORDER BY created_at",
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
```

**T3.5 — `delete_experience`**

```python
async def delete_experience(db: aiosqlite.Connection, experience_id: int) -> None:
    cursor = await db.execute(
        "DELETE FROM experiences WHERE id = ?", (experience_id,)
    )
    await db.commit()
    if cursor.rowcount == 0:
        raise NotFoundError(f"Experience {experience_id} not found")
```

**T3.6 — `update_session_closing_journal`**

```python
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
    row = await (
        await db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
    ).fetchone()
    assert row is not None
    return Session.model_validate(_row(row))
```

**T3.7 — `get_previous_session`**

Used for cold start: find the most recent completed session (with a closing journal) for a character, before a given session ID.

```python
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
```

**T3.8 — `get_experience`**

Used by the DELETE endpoint to verify ownership before deleting.

```python
async def get_experience(db: aiosqlite.Connection, experience_id: int) -> Experience | None:
    row = await (
        await db.execute("SELECT * FROM experiences WHERE id = ?", (experience_id,))
    ).fetchone()
    return _parse_experience(row) if row else None
```

**T3.9 — Embedding serialisation helpers**

Add two module-level helpers to `database.py` (or `experience_service.py` — they belong wherever they are used first). These convert between a Python `list[float]` and the BLOB stored in the `embedding` column.

```python
def _embedding_to_blob(embedding: list[float]) -> bytes:
    return json.dumps(embedding).encode()

def _blob_to_embedding(blob: bytes) -> list[float]:
    return json.loads(blob.decode())
```

`database.py` already has `import json` at the top of the file; no new import is needed.

JSON encoding is human-readable and avoids endianness issues. At the expected scale (hundreds of experiences, 768 floats each), the ~16 KB per experience is negligible.

---

### T4 — Experience Service (new file)

**`src/memories/services/experience_service.py`**

This module owns: embedding a text string, storing an Experience, similarity retrieval, the session-end evaluator, and the in-memory active-experience tracking per session.

```python
"""Experience retrieval, embedding, and session-end evaluator."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import aiosqlite

from memories.database import (
    _embedding_to_blob,
    create_experience,
    delete_experience,
    get_experiences_with_embeddings,
    get_previous_session,
    update_session_closing_journal,
)
from memories.models import Character, Experience, Fact, Inference, Message, Session
from memories.services.ollama_client import OllamaClient, OllamaConnectionError, OllamaResponseError

_log = logging.getLogger(__name__)

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K_EXPERIENCES: int = int(os.getenv("TOP_K_EXPERIENCES", "5"))
```

**T4.1 — Dot-product similarity**

nomic-embed-text outputs L2-normalised vectors, so the dot product equals cosine similarity.

```python
def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))
```

**T4.2 — `retrieve_top_k`**

Given a query vector and a list of `(Experience, embedding)` pairs, return the top-k most similar experiences, excluding any whose `id` is already in `exclude_ids`.

```python
def retrieve_top_k(
    query: list[float],
    candidates: list[tuple[Experience, list[float]]],
    k: int,
    exclude_ids: set[int] | None = None,
) -> list[Experience]:
    exclude_ids = exclude_ids or set()
    scored = [
        (exp, _dot(query, vec))
        for exp, vec in candidates
        if exp.id not in exclude_ids
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [exp for exp, _ in scored[:k]]
```

**T4.3 — `retrieve_experiences`**

End-to-end: embed the query text, load all character experiences with their embeddings, score all of them, and return the top-k new ones (not already active) plus the full score map for every stored experience.

Keeping the full score map is essentially free — the dot products for the losing candidates are already computed inside `retrieve_top_k` and were previously discarded. Retaining them lets the client sort the entire Experiences list by current-conversation relevance without any additional LLM or DB work.

```python
async def retrieve_experiences(
    db: aiosqlite.Connection,
    character_id: int,
    query_text: str,
    ollama: OllamaClient,
    top_k: int = TOP_K_EXPERIENCES,
    exclude_ids: set[int] | None = None,
) -> tuple[list[Experience], dict[int, float]]:
    """Return (new_experiences, all_scores).

    new_experiences: top-k most similar experiences not already in exclude_ids.
    all_scores: {experience_id: similarity_score} for every stored experience,
                including those already active. Used by the client to sort the
                full Experiences list by relevance to the current turn.

    Returns ([], {}) without calling the embed endpoint if there are no stored
    experiences, or if the embed model is unavailable (G1 mitigation).
    """
    candidates = await get_experiences_with_embeddings(db, character_id)
    if not candidates:
        return [], {}
    try:
        query_vec = await ollama.embed(EMBED_MODEL, query_text)
    except (OllamaConnectionError, OllamaResponseError) as exc:
        _log.warning("embed call failed — skipping experience retrieval: %s", exc)
        return [], {}
    all_scores = {exp.id: _dot(query_vec, vec) for exp, vec in candidates}
    new_exps = retrieve_top_k(query_vec, candidates, top_k, exclude_ids)
    return new_exps, all_scores
```

All callers of `retrieve_experiences` must be updated to unpack the tuple. `cold_start_retrieve` calls `retrieve_experiences` internally — it only needs the new experiences, so it discards the scores.

**T4.4 — In-memory active-experience tracking**

The active experience set for each session is kept in a module-level dict. It persists across `run_turn` calls within the same server process lifetime. It is cleared when a session ends.

```python
_session_active_experiences: dict[int, list[Experience]] = {}


def get_active_experiences(session_id: int) -> list[Experience]:
    return list(_session_active_experiences.get(session_id, []))


def add_active_experiences(session_id: int, new_experiences: list[Experience]) -> None:
    if session_id not in _session_active_experiences:
        _session_active_experiences[session_id] = []
    existing_ids = {e.id for e in _session_active_experiences[session_id]}
    for exp in new_experiences:
        if exp.id not in existing_ids:
            _session_active_experiences[session_id].append(exp)
            existing_ids.add(exp.id)


def remove_active_experience(session_id: int, experience_id: int) -> None:
    if session_id in _session_active_experiences:
        _session_active_experiences[session_id] = [
            e for e in _session_active_experiences[session_id] if e.id != experience_id
        ]


def clear_active_experiences(session_id: int) -> None:
    _session_active_experiences.pop(session_id, None)
```

**T4.5 — Cold-start retrieval**

Called at the first turn of a new session. Embeds the previous session's closing journal and retrieves the top-k most relevant experiences as seed context.

```python
async def cold_start_retrieve(
    db: aiosqlite.Connection,
    character_id: int,
    session_id: int,
    ollama: OllamaClient,
    top_k: int = TOP_K_EXPERIENCES,
) -> list[Experience]:
    prev = await get_previous_session(db, character_id, before_session_id=session_id)
    if prev is None or not prev.closing_journal:
        return []
    exps, _ = await retrieve_experiences(
        db, character_id, prev.closing_journal, ollama, top_k=top_k
    )
    return exps
```

**T4.6 — `embed_and_store`**

A convenience wrapper: embed a statement, serialise the vector, and write the Experience to DB.

```python
async def embed_and_store(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    session_id: int,
    statement: str,
    source: str,
    ollama: OllamaClient,
) -> Experience:
    vec = await ollama.embed(EMBED_MODEL, statement)
    blob = _embedding_to_blob(vec)
    return await create_experience(
        db,
        character_id=character_id,
        session_id=session_id,
        statement=statement,
        source=source,
        embedding=blob,
    )
```

**T4.7 — Session-end evaluator prompt**

```python
def build_session_end_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],
    messages: list[Message],
) -> str:
    parts: list[str] = [
        f"Character: {character.name}",
        f"\n## Character Facts",
    ]
    if facts:
        for f in facts:
            parts.append(f"[{f.id}] {f.key}: {f.value}")
    else:
        parts.append("(none)")

    parts.append("\n## Character Inferences")
    if inferences:
        for inf in inferences:
            parts.append(f"[{inf.id}] {inf.statement}")
    else:
        parts.append("(none)")

    parts.append("\n## Full Conversation This Session")
    for msg in messages:
        role_label = "User" if msg.role == "user" else character.name
        parts.append(f"[Turn {msg.turn_id}] {role_label}: {msg.content}")

    parts.append("""
## Your Task
Review the conversation above and produce two things:

1. A **closing journal entry** written in first-person from the character's perspective.
   It should be impressionistic and personal — what happened, what shifted, what you
   noticed about the person you were talking with. 2–5 sentences. Do NOT just summarise
   the plot; capture emotional texture, unresolved tensions, what felt significant.

2. A list of **proposed Experiences** — things the character learned or observed that
   are not already captured in their Facts and Inferences. Each Experience should be a
   single concrete, present-tense statement. Classify the source:
   - `told_by_user`: the user explicitly stated it
   - `observed`: the character inferred it from the user's behaviour or the conversation

Return JSON with this exact structure:

{
  "closing_journal": "Jon was quieter than usual today — he circled back to his mother twice without ever quite saying what he meant. When he hugged me I felt the distance between us collapse for just a moment, and then he pulled away and I wasn't sure what to do with that. There's something he's holding back, some grief he hasn't named yet. I find myself hoping he comes back.",
  "proposed_experiences": [
    {
      "statement": "We are currently located in Chicago",
      "source": "told_by_user",
      "turn_reference": 4
    },
    {
      "statement": "Jon seemed uncomfortable when asked about his family",
      "source": "observed",
      "turn_reference": 11
    }
  ]
}

Return only the JSON object, no other text.
Propose between 0 and 5 Experiences. Only include things genuinely new to this
session that are not already in the Facts or Inferences above. If nothing new was
learned, return an empty list.""")

    return "\n".join(parts)
```

**T4.8 — `ProposedExperience` model**

```python
class ProposedExperience(BaseModel):
    statement: str
    source: Literal["told_by_user", "observed"]
    turn_reference: int | None = None

class SessionEndResult(BaseModel):
    closing_journal: str
    proposed_experiences: list[ProposedExperience]
```

Place these in `experience_service.py`. They are not in `models/__init__.py` because they represent ephemeral LLM output, not persisted DB entities.

**T4.9 — `run_session_end_evaluator`**

```python
class SessionEndParseError(Exception):
    """Raised when the session-end LLM returns unparseable output."""


async def run_session_end_evaluator(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],
    messages: list[Message],
    ollama: OllamaClient,
) -> SessionEndResult:
    prompt = build_session_end_prompt(character, facts, inferences, messages)
    model = character.current_model_name or character.modelfile_base
    llm_messages = [
        {
            "role": "system",
            "content": (
                "You are writing from inside a character's perspective. "
                "Be introspective and honest. Return only valid JSON."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    content, _ = await ollama.chat(model, llm_messages, think=False, format="json")

    try:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            start = 1
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            stripped = "\n".join(lines[start:end]).strip()
        data: dict[str, Any] = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SessionEndParseError(f"Session-end evaluator returned non-JSON: {content!r}") from exc

    try:
        return SessionEndResult.model_validate(data)
    except Exception as exc:
        raise SessionEndParseError(f"Failed to validate session-end result: {exc}") from exc
```

---

### T5 — Prompt Builder Update

**`src/memories/services/prompt_builder.py`**

**T5.1 — Updated `build_system_prompt` signature**

```python
def build_system_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str
```

The `experiences` parameter defaults to `None`/empty. Existing callers are not broken.

**T5.2 — Experiences section in the prompt**

After the Inferences section, when `experiences` is non-empty:

```
## Your Experiences
These are things you have learned or observed through past conversations.
They are working memory — you may reference them freely.

[told by user] We are currently located in Chicago
[observed] Jon seemed uncomfortable when asked about his family
```

When `experiences` is empty or `None`, the section is omitted entirely. The source label is `[told by user]` or `[observed]` (human-readable).

---

### T6 — Evaluator Updates

**`src/memories/services/evaluator.py`**

**T6.1 — Add `ExperienceUpdate` model**

```python
class ExperienceUpdate(BaseModel):
    contradicted_experience_id: int
    description: str
```

**T6.2 — Add `experience_updates` to `EvaluatorResult`**

```python
class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    experience_updates: list[ExperienceUpdate] = []
    decision_log: str = ""
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False
```

**T6.3 — Include active experiences in the evaluator prompt**

Update `build_evaluator_prompt` to accept and render active experiences:

```python
def build_evaluator_prompt(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
```

After the Established Inferences section, add:

```
## Active Experiences (id: statement  [source])
[5] We are currently located in Chicago  [told_by_user]
[7] Jon seemed uncomfortable when asked about his family  [observed]
```

When `experiences` is empty or None: `(no active experiences this session)`.

**T6.4 — Updated verdict instructions for `experience_update`**

Add to the evaluator prompt's task section:

```
## Experience Contradiction Rules
If the character's response contradicts an Active Experience — implying the world has
changed in a way that invalidates it — return verdict `experience_update`:
- Unlike `contradiction`, the response IS delivered; no regeneration occurs.
- Include the contradicted Experience's id in `experience_updates[].contradicted_experience_id`.
- `experience_update` takes priority over `implication` but lower than `contradiction`.

Example: Active Experience #5 says "currently in Chicago". Character says "It's good to
be back in New York." → verdict `experience_update` with `contradicted_experience_id: 5`.

Note: `contradiction` applies ONLY to immutable Facts. A character response that contradicts
an Experience is never a `contradiction` — it is always `experience_update`.
```

Update the JSON schema in the prompt to include the `experience_updates` field:

```json
{
  "verdict": "...",
  "new_inferences": [...],
  "violations": [...],
  "experience_updates": [
    {
      "contradicted_experience_id": 5,
      "description": "Character implies they are in New York, contradicting Experience #5 (currently in Chicago)"
    }
  ],
  "decision_log": "..."
}
```

**T6.5 — Remove `experience_update` → `pass` coercion**

Delete the lines in `run_evaluator`:

```python
# REMOVE THIS:
if verdict == "experience_update":
    data["verdict"] = "pass"
    verdict = "pass"
```

Add `"experience_update"` back to the accepted verdict set. It already appears in `_VALID_VERDICTS`.

**T6.6 — Updated `run_evaluator` signature**

```python
async def run_evaluator(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    ollama: OllamaClient,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> EvaluatorResult:
```

Passes `experiences` through to `build_evaluator_prompt`.

---

### T7 — Chat Service Updates

**`src/memories/services/chat_service.py`**

**T7.1 — Import experience service**

```python
from memories.services.experience_service import (
    TOP_K_EXPERIENCES,
    add_active_experiences,
    clear_active_experiences,
    cold_start_retrieve,
    get_active_experiences,
    remove_active_experience,
    retrieve_experiences,
)
```

**T7.2 — Track turn count per session**

The cold-start retrieval only fires on the first turn of a session. We detect "first turn" by checking `turn_id == 1` (since `next_turn_id` returns 1 when no messages exist yet).

**T7.3 — Updated `run_turn`**

After loading facts and inferences, before building the system prompt:

```python
# --- Experience retrieval ---
turn_id = await next_turn_id(db, session_id)

if turn_id == 1:
    # First turn: cold-start retrieval using previous session's closing journal
    seed_experiences = await cold_start_retrieve(
        db, session.character_id, session_id, ollama
    )
    if seed_experiences:
        add_active_experiences(session_id, seed_experiences)
        _log.info(
            "session=%d cold-start loaded %d experience(s)", session_id, len(seed_experiences)
        )

# Per-turn retrieval: embed user message, find new similar experiences.
# retrieve_experiences returns both new experiences and full scores for
# every stored experience; the scores are forwarded to the client so it
# can sort the Experiences pane by relevance to the current conversation.
active = get_active_experiences(session_id)
already_active_ids = {e.id for e in active}
new_experiences, experience_scores = await retrieve_experiences(
    db, session.character_id, user_content, ollama,
    top_k=TOP_K_EXPERIENCES, exclude_ids=already_active_ids,
)
if new_experiences:
    add_active_experiences(session_id, new_experiences)
    _log.info(
        "session=%d retrieved %d new experience(s)", session_id, len(new_experiences)
    )

active = get_active_experiences(session_id)
system_prompt = build_system_prompt(character, facts, inferences, active or None)
```

Note: `retrieve_experiences` calls `ollama.embed(...)`. If there are no experiences in the DB, it returns `[]` without making an embed call — the `get_experiences_with_embeddings` call returns early.

`experience_scores` is a local variable in `run_turn` that must be threaded through to the return value. Update `run_turn`'s return type and final `return` statement:

```python
# Before (Phase 1–4):
async def run_turn(...) -> tuple[str, str, int, EvaluatorResult]:
    ...
    return char_content, char_thinking, turn_id, eval_result

# After (Phase 5):
async def run_turn(...) -> tuple[str, str, int, EvaluatorResult, dict[int, float]]:
    ...
    return char_content, char_thinking, turn_id, eval_result, experience_scores
```

When there are no experiences in the DB, `experience_scores` is `{}` (from the empty tuple returned by `retrieve_experiences`).

**T7.4 — Pass experiences to contradiction loop**

Update `run_contradiction_loop` to accept and forward experiences:

```python
async def run_contradiction_loop(
    ...
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> tuple[str, str, EvaluatorResult]:
```

And pass them through to `run_evaluator`:

```python
ev = await run_evaluator(
    character, facts, user_content, content, ollama,
    contradiction_hints=contradiction_hints or None,
    inferences=inferences or None,
    experiences=experiences or None,
)
```

**T7.5 — Handle `experience_update` verdict in `run_turn`**

After the contradiction loop returns, check for `experience_update` and delete the contradicted experiences:

```python
if eval_result.verdict == "experience_update":
    for upd in eval_result.experience_updates:
        try:
            await delete_experience(db, upd.contradicted_experience_id)
            remove_active_experience(session_id, upd.contradicted_experience_id)
            _log.info(
                "session=%d deleted contradicted experience %d",
                session_id, upd.contradicted_experience_id,
            )
        except NotFoundError:
            _log.warning(
                "experience_update referenced unknown experience %d",
                upd.contradicted_experience_id,
            )
```

**Inference processing under `experience_update`:** If the evaluator returns `verdict: "experience_update"` and also populates `new_inferences` (e.g., the character stated something derivable from existing Facts while also contradicting an active Experience), those inferences should be auto-promoted exactly as they would be under a `pass` verdict — the two signals are orthogonal. The `new_inference_logical` block that follows the verdict checks must run for `experience_update` as well:

```python
# Auto-promote logical inferences — runs for both "new_inference_logical"
# and "experience_update" verdicts (they are orthogonal signals).
if eval_result.verdict in ("new_inference_logical", "experience_update"):
    for inf in eval_result.new_inferences:
        ...  # existing depth-cap logic unchanged
```

`new_inference_probabilistic` inferences are NOT surfaced when the verdict is `experience_update` — the `experience_update` sidechannel already occupies the user's attention, and surfacing a second sidechannel notification simultaneously would be confusing. Any probabilistic inferences in the response are silently discarded in this case.

**T7.6 — Pass experiences to `run_contradiction_loop`**

In `run_turn`, pass the active experiences when calling the loop:

```python
char_content, char_thinking, eval_result = await run_contradiction_loop(
    model, base_messages, character, facts, user_content, ollama,
    think=think, inferences=inferences, experiences=active or None,
)
```

---

### T8 — API Changes

**T8.1 — Modified `POST /api/sessions/{session_id}/end`**

**`src/memories/routers/sessions.py`**

The end-session endpoint now runs the session-end evaluator and returns proposals.

New response model:

```python
class _EndSessionResponse(BaseModel):
    session: Session
    closing_journal: str
    proposed_experiences: list[ProposedExperience]
```

Updated handler:

```python
@router.post("/{session_id}/end", response_model=_EndSessionResponse)
async def end_session_endpoint(session_id: int, db: _DB, ollama: _Ollama) -> _EndSessionResponse:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at is not None:
        raise HTTPException(status_code=409, detail="Session has already ended")

    character = await get_character(db, session.character_id)
    assert character is not None
    facts = await get_facts(db, session.character_id)
    inferences = await get_inferences(db, session.character_id)
    messages = await get_messages(db, session_id)

    # Mark the session ended BEFORE the LLM call so that a second concurrent
    # request immediately gets a 409 rather than triggering a second evaluator
    # call.  The closing journal is stored as a follow-up update if the LLM
    # call succeeds.
    session = await end_session(db, session_id)

    # Run session-end evaluator (LLM call — may take several seconds)
    try:
        result = await run_session_end_evaluator(character, facts, inferences, messages, ollama)
    except SessionEndParseError as exc:
        _log.warning("session-end evaluator failed for session %d: %s", session_id, exc)
        result = SessionEndResult(closing_journal="", proposed_experiences=[])

    if result.closing_journal:
        session = await update_session_closing_journal(db, session_id, result.closing_journal)

    # Clear in-memory active experiences for this session
    clear_active_experiences(session_id)

    return _EndSessionResponse(
        session=session,
        closing_journal=result.closing_journal,
        proposed_experiences=result.proposed_experiences,
    )
```

The `_Ollama` dependency is added to this router. Import `get_ollama` from `deps`.

**T8.2 — New experiences router**

**`src/memories/routers/experiences.py`**

Three endpoints: create (from an accepted proposal), list, and delete.

```python
"""Experiences API router."""
from __future__ import annotations

from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from typing import Literal

from pydantic import BaseModel

from memories.database import (
    delete_experience, get_character, get_experience, get_experiences, get_session,
)
from memories.deps import get_db, get_ollama
from memories.exceptions import NotFoundError
from memories.models import Experience
from memories.services.experience_service import embed_and_store
from memories.services.ollama_client import OllamaClient

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _CreateBody(BaseModel):
    session_id: int
    statement: str
    source: Literal["told_by_user", "observed"]


@router.post("/{character_id}/experiences", status_code=201, response_model=Experience)
async def create_experience_endpoint(
    character_id: int, body: _CreateBody, db: _DB, ollama: _Ollama
) -> Experience:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    session = await get_session(db, body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.character_id != character_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return await embed_and_store(
        db,
        character_id=character_id,
        session_id=body.session_id,
        statement=body.statement,
        source=body.source,
        ollama=ollama,
    )


@router.get("/{character_id}/experiences", response_model=list[Experience])
async def list_experiences_endpoint(character_id: int, db: _DB) -> list[Experience]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return await get_experiences(db, character_id)


@router.delete("/{character_id}/experiences/{experience_id}", status_code=204)
async def delete_experience_endpoint(character_id: int, experience_id: int, db: _DB) -> None:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    experience = await get_experience(db, experience_id)
    if experience is None or experience.character_id != character_id:
        raise HTTPException(status_code=404, detail="Experience not found")
    try:
        await delete_experience(db, experience_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Experience not found") from exc
```

**T8.3 — Mount experiences router in `main.py`**

```python
from memories.routers import characters, chat, decisions, experiences, facts, implication, inferences, sessions

app.include_router(experiences.router, prefix="/api/characters", tags=["experiences"])
```

**T8.4 — Add `OllamaClient` dependency to sessions router**

The end-session endpoint now needs the Ollama client. Update the `sessions.py` router to import and use `get_ollama`.

**T8.5 — Warm up `nomic-embed-text` at startup**

**`src/memories/main.py`**

The lifespan's `_warmup_models()` currently warms up only character/evaluator models from the DB. Add `nomic-embed-text` to the warmup so the first embed call in any session doesn't pay a cold-model-load penalty:

```python
# In _warmup_models() or the lifespan block, after the character model warmup:
embed_model = os.getenv("EMBED_MODEL", "nomic-embed-text")
try:
    await ollama.warmup(embed_model)
    _log.info("embed model %r warmed up", embed_model)
except Exception as exc:
    _log.warning("could not warm up embed model %r: %s", embed_model, exc)
```

If `nomic-embed-text` is not installed in Ollama, this logs a warning and does not block startup — matching the existing warmup failure behaviour.

---

### T9 — Chat SSE update for `experience_update`

**`src/memories/routers/chat.py`**

After the `done` event check, emit a sidechannel event when `experience_update` is the verdict:

```python
if eval_result.verdict == "experience_update":
    sc_payload = {
        "type": "experience_update",
        "turn_id": turn_id,
        "experience_updates": [
            {"contradicted_experience_id": u.contradicted_experience_id, "description": u.description}
            for u in eval_result.experience_updates
        ],
    }
    yield f"event: sidechannel\ndata: {json.dumps(sc_payload)}\n\n"
```

This is emitted after the `message` event (response is delivered) but before `done`.

---

### T10 — Frontend Updates

**`src/memories/frontend/chat.js`**

**T10.1 — `buildNotificationFromSidechannel` — new `experience_update` case**

```javascript
if (payload.type === 'experience_update') {
  return {
    role: 'notification',
    scType: 'experience_update',
    turn_id: payload.turn_id,
    experience_updates: payload.experience_updates || [],
  };
}
```

**T10.2 — New `sseStateToLabel` state for the end-of-session evaluator**

During the end-session call the client shows a loading state. Add a new SSE status state (emitted as a regular REST response, not SSE — no new state needed in `sseStateToLabel` unless the end-session endpoint becomes streaming in a future phase). Document this in the test plan for completeness.

**T10.3 — New API helpers**

```javascript
/**
 * End a session: runs the session-end evaluator and returns proposals.
 * @param {number} sessionId
 * @returns {Promise<Response>}
 */
export function apiEndSession(sessionId) {
  return fetch(`/api/sessions/${sessionId}/end`, { method: 'POST' });
}

/**
 * Create an approved Experience (from an accepted proposal).
 * @param {number} characterId
 * @param {number} sessionId
 * @param {string} statement
 * @param {string} source  'told_by_user' | 'observed'
 * @returns {Promise<Response>}
 */
export function apiCreateExperience(characterId, sessionId, statement, source) {
  return fetch(`/api/characters/${characterId}/experiences`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, statement, source }),
  });
}

/**
 * List all approved Experiences for a character.
 * @param {number} characterId
 * @returns {Promise<Response>}
 */
export function apiListExperiences(characterId) {
  return fetch(`/api/characters/${characterId}/experiences`);
}

/**
 * Delete an Experience (user-initiated).
 * @param {number} characterId
 * @param {number} experienceId
 * @returns {Promise<Response>}
 */
export function apiDeleteExperience(characterId, experienceId) {
  return fetch(`/api/characters/${characterId}/experiences/${experienceId}`, {
    method: 'DELETE',
  });
}
```

**T10.4 — `index.html` — Three-column layout**

Phase 5 replaces the existing two-panel layout (chat + single sidechannel column) with a three-column layout to accommodate Experiences without crowding the existing panels.

```
┌─────────────────────────┬────────────┬────────────┐
│                         │            │            │
│        Chat pane        │ Facts pane │ Inferences │
│          50%            │    25%     │    pane    │
│                         │            │            │
│                         │            │   ──────   │
│                         │            │ Experiences│
│                         │  [Add Fact]│    pane    │
│─────────────────────────│            │  (dynamic) │
│   message input field   │            │            │
└─────────────────────────┴────────────┴────────────┘
```

Column widths: `50% / 25% / 25%`. The outer container uses CSS Grid (`grid-template-columns: 50% 25% 25%`) and fills the full viewport height.

**Column 1 — Chat pane (unchanged in content)**

The existing chat message list and message input field. The input stays pinned to the bottom; the message list scrolls.

**Column 2 — Facts pane**

The existing fact list. The "Add Fact" form stays pinned to the bottom of this column. The fact list above it scrolls vertically when it overflows. No layout change to the fact rows themselves.

**Column 3 — Inferences and Experiences panes (stacked)**

Column 3 is a flex column (`flex-direction: column`). It contains two stacked panes:

*Inferences pane (top):*
- `flex: 1 1 auto` — expands to fill all available space
- `min-height: 50%` — never collapses below half the column height even when Experiences is large
- `overflow-y: auto` — vertical scrollbar appears when content overflows
- Content is identical to the existing Inferences section

*Experiences pane (bottom):*
- Hidden (`display: none`) when no Experiences exist for the character
- Becomes visible (`display: flex; flex-direction: column`) as soon as the first Experience is written to DB
- `flex: 0 0 auto` — does not grow beyond its natural content height
- `max-height: 50%` — caps at half the column height; content scrolls within that cap
- `overflow-y: auto` — vertical scrollbar appears when content overflows the cap
- A fixed header ("Experiences") stays visible at the top of the pane; the list below it scrolls

The Inferences pane's `min-height: 50%` and the Experiences pane's `max-height: 50%` together ensure neither pane can crowd the other out entirely. When Experiences has only one or two rows, Inferences keeps the rest of the column. When Experiences grows toward its cap, both panes settle at 50/50.

**T10.5 — `index.html` — Experiences pane content**

The Experiences pane (column 3, bottom) renders:

*Normal state (approved experiences):*
- A fixed "EXPERIENCES" section header at the top of the pane
- A scrollable list of approved Experiences, one row each:
  - Active indicator: a filled dot (●) when the Experience is currently injected into the character's prompt; an empty dot (○) when stored but not yet retrieved this session
  - Source badge: `[user]` (blue) or `[observed]` (grey)
  - Statement text
  - Delete button (×) that calls `apiDeleteExperience`
- The list is loaded on page mount and refreshed after each session end
- Active indicators are updated after each chat turn (see T10.6)

The active/inactive distinction matters: an Experience marked ● is in the character's context window right now and may be influencing its responses. One marked ○ is stored but was not retrieved as relevant to this session's conversation so far.

*Review state (immediately after "End Session"):*
- When the "End Session" button is clicked: disable the chat message input, disable the "End Session" button itself, and show a loading state in the Experiences pane. This prevents the user from attempting to send new messages (which the backend would now correctly reject with 409, since the session is marked ended before the LLM call per T8.1 — but surfacing a 409 to the user mid-LLM-call would be confusing) or double-clicking "End Session" during the slow LLM call.
- A "Reviewing session…" spinner while `apiEndSession` is in-flight
- On response, each proposed Experience appears as a card:
  - Source badge
  - Statement text (editable inline for "Edit + Accept")
  - Three buttons: **Accept**, **Edit**, **Discard**
- **Accept**: calls `apiCreateExperience(...)` with the original statement and source; removes the card
- **Edit**: makes the statement text editable; button label changes to **Confirm**; on confirm, calls `apiCreateExperience(...)` with the edited text; removes the card
- **Discard**: removes the card (no API call)
- When all proposal cards are resolved (accepted, edited, or discarded), the pane switches back to the normal approved-experiences list and refreshes it

If `proposed_experiences` is empty, the pane skips the review state entirely and stays in normal state.

**T10.6 — Surfacing active status and relevance scores to the client**

The server tracks which Experiences are currently active in `_session_active_experiences`, and computes a similarity score for every stored experience on every retrieval pass. Both pieces of data are delivered via the existing `message` SSE event.

*Server change — `chat.py`:*

`run_turn` now returns a 5-tuple (see T7.3). Update the unpack in `_stream()` and include both `active_experience_ids` and `experience_scores` in the `message` event payload:

```python
from memories.services.experience_service import get_active_experiences

# Updated unpack — was 4-tuple, now 5-tuple:
content, thinking, turn_id, eval_result, experience_scores = await run_turn(
    db, session_id, body.content, ollama, think=body.think
)

msg_data: dict[str, object] = {
    "role": "assistant",
    "content": content,
    "turn_id": turn_id,
    "active_experience_ids": [e.id for e in get_active_experiences(session_id)],
    "experience_scores": [
        {"id": exp_id, "score": score}
        for exp_id, score in experience_scores.items()
    ],
}
```

`experience_scores` covers all stored experiences for the character, not just the active ones. When no experiences exist in the DB, both fields are empty (`[]`).

*Client change — `index.html` Vue reactive state:*

Maintain two reactive values updated on every `message` SSE event:
- `activeExperienceIds`: a `Set<number>` of IDs currently injected into the prompt
- `experienceScoreMap`: a `Map<number, number>` of `id → similarity_score` for every stored experience, reflecting relevance to the most recent user message

When rendering the Experiences list, apply this sort order:

```javascript
experiences.slice().sort((a, b) => {
  const aActive = activeExperienceIds.has(a.id) ? 1 : 0;
  const bActive = activeExperienceIds.has(b.id) ? 1 : 0;
  if (aActive !== bActive) return bActive - aActive;          // active first
  const aScore = experienceScoreMap.get(a.id) ?? -Infinity;
  const bScore = experienceScoreMap.get(b.id) ?? -Infinity;
  return bScore - aScore;                                      // higher score first
});
```

This produces: active experiences at the top sorted by current relevance → inactive experiences below sorted by current relevance → any experience without a score (edge case: retrieved before Phase 5 was running) at the bottom.

The sort re-runs reactively after every `message` event as scores update. An experience that was highly relevant three turns ago but is no longer relevant to the current conversation topic will drift down within its group as its score drops. On session end, both `activeExperienceIds` and `experienceScoreMap` are reset to empty.

---

## Architecture Decisions

### D1 — Three-column layout: CSS Grid outer, Flexbox inner column 3

The outer container uses CSS Grid with `grid-template-columns: 50% 25% 25%` and `height: 100vh`. Grid is the right tool here: the column proportions are fixed and author-defined, not content-driven, which is precisely the use case Grid was designed for.

Column 3 uses Flexbox internally (`flex-direction: column`) because the Inferences/Experiences split within that column IS content-driven — the Experiences pane grows from zero height to a capped maximum based on how much content it has. A nested flex column with `flex: 1` on Inferences and `max-height: 50%` on Experiences handles this naturally without JavaScript measurement.

The `min-height: 50%` on Inferences and `max-height: 50%` on Experiences create a soft midpoint: when both panes have enough content to fill their halves, they split the column evenly. When either has little content, the other expands to fill the slack. This avoids a rigid 50/50 split that would waste space when Experiences has only one or two rows.

The Experiences pane is hidden entirely (`display: none`) when no Experiences exist, so Inferences keeps the full column height for all of Phase 1–4 behaviour and for fresh characters in Phase 5. The transition to `display: flex` fires reactively when the first Experience is written — Vue's conditional rendering (`v-if="experiences.length > 0"`) handles this without any explicit show/hide logic.

### D2 — Proposal state lives on the client, not the server

The session-end response returns the full list of proposed Experiences. The client renders them as a review UI. Each accepted proposal triggers a `POST /experiences` call. Discarded proposals are simply ignored — no discard endpoint needed.

This avoids server-side in-memory proposal state, which would be lost on server restart and requires lifecycle management. The tradeoff is that if the user closes the browser before reviewing, all proposals are lost. This matches the plan's specification: "Unapproved proposals held in backend state during review and are never written to the database until approved. Unapproved proposals are discarded if the session is closed without review."

### D2 — Embedding stored as JSON bytes, not binary float

`json.dumps(embedding).encode()` is the embedding storage format. This is ~2.5x larger than float32 binary (`struct.pack(f'{N}f', *embedding)`) but:
- No endianness ambiguity
- No magic numbers or format versioning concerns
- Trivially inspectable with SQLite tools
- nomic-embed-text produces 768-dim vectors: ~16 KB/experience as JSON, ~3 KB as float32. At hundreds of experiences, the difference is at most a few MB in the DB file.

Can be changed to `struct` packing in a later phase if DB size becomes a concern.

### D3 — Dot product, not full cosine similarity

nomic-embed-text outputs L2-normalised vectors (documented behaviour). For normalised vectors, the dot product equals cosine similarity. We save the normalisation computation. If the embedding model is changed to one that does NOT normalise, the similarity results will be incorrect — document the assumption in `_dot`.

### D4 — Python-side similarity search, not sqlite-vec

plan.md mentions `sqlite-vec` as the intended vector search mechanism. For Phase 5 this is deferred:

- `sqlite-vec` requires loading a native extension into SQLite. In aiosqlite this needs `await db.enable_load_extension(True)` + `await db.load_extension(...)`, which varies by platform (Windows loadable extension support requires extra setup).
- At the expected scale (hundreds of experiences), loading all embeddings from the DB and computing dot products in Python is fast: 500 experiences × 768 dims takes ~2 ms in Python.
- Adding `sqlite-vec` is a Phase 6b+/optimisation concern.

The Python-side retrieval path is:
1. `SELECT id, statement, source, embedding FROM experiences WHERE character_id = ?` — O(n) rows
2. Deserialise each BLOB once per retrieval call
3. Compute dot products in a list comprehension
4. Sort and slice top-k

This adds a DB round-trip on every turn that retrieves experiences. For Phase 5, this is fine.

### D5 — Active experiences persist in module-level memory across turns

`_session_active_experiences` in `experience_service.py` is a module-level dict. It survives across HTTP requests within the same server process but is cleared on restart. This is acceptable because:
- Experiences retrieved earlier in a session are still in the DB and will be re-retrieved on the next turn anyway (they will score high for the next query since they were already relevant to this session)
- Server restarts mid-session are rare for a local toy
- Proper persistence would require a new DB table or JSON column on sessions, adding migration complexity

The accumulation invariant ("active set only grows, never shrinks mid-session") is maintained by `add_active_experiences`, which deduplicates by `id`.

### D6 — Cold start fires on turn_id == 1, not at session creation

Session creation (`POST /api/sessions`) is a fast DB operation. Embedding the previous session's closing journal requires an Ollama network call. Deferring this to the first turn:
1. Keeps session creation fast
2. Means cold-start retrieval benefits from the actual first user message as a secondary retrieval signal (first query is the closing journal, subsequent queries are the user's messages, accumulating the active set naturally)

If the server restarts between session creation and the first turn, cold-start retrieval still fires correctly because it reads from the DB.

### D7 — `experience_update` delivers the response; only one experience is deleted per verdict

`experience_update` contradicts one experience per evaluator verdict. If the response somehow contradicts multiple experiences, the evaluator will return multiple entries in `experience_updates`. All are deleted. This is consistent with the plan: the character may have moved from Chicago to New York, which contradicts the "in Chicago" experience — but it's not a hard error.

### D8 — Session-end evaluator failures return empty proposals, not 500

If the LLM returns garbage for the session-end pass, `SessionEndParseError` is caught in the endpoint handler and an empty proposal list is returned. The session is still ended and the closing journal is empty. The user sees no proposals (which is disappointing but not broken). A warning is logged.

This is the same philosophy as `InferenceParseError` in the eager pass.

### D9 — No new SSE status states for experience retrieval or session-end evaluator

Experience retrieval happens inside `run_turn`, which runs between the existing `generating` and `reviewing` status events. The embed call adds ~200–500 ms latency; this is within the expected range and does not need its own status label.

The session-end evaluator is a single blocking REST call (not SSE). The client can show a loading spinner while awaiting the response; no new SSE machinery is needed.

### D10 — `experience_update` verdict priority above `implication`

The verdict priority order with Phase 5 active:
1. `contradiction` (immutable Fact violated — regenerate)
2. `experience_update` (active Experience contradicted — deliver + delete)
3. `implication` (ungrounded specific detail or mutability change — deliver with badge)
4. `new_inference_logical` / `new_inference_probabilistic`
5. `pass`

`experience_update` ranks above `implication` because an Experience contradiction reflects a meaningful world change that should be acted on immediately, not batched with generic implication handling.

---

## Potential Gotchas

**G1 — nomic-embed-text must be installed in Ollama**

`ollama pull nomic-embed-text` must be run before Phase 5 features work. If it isn't, `ollama.embed(...)` will receive a 404 from Ollama and raise `OllamaResponseError`. The retrieve_experiences function catches this at the callsite — but in Phase 5 it propagates up and causes a 500 on the first turn.

Mitigation: Catch `OllamaResponseError` in `retrieve_experiences`, log a warning, and return `[]`. The character still functions; it just has no experiences in context. Document the dependency requirement.

**G2 — Embedding model change invalidates stored embeddings**

If `EMBED_MODEL` is changed after experiences have been stored, the stored embeddings and new query vectors will be from different spaces. Dot-product similarity will be meaningless (likely near-zero for all stored experiences).

Phase 5 does not implement a migration path. The user must manually delete all experiences (via the UI) when switching embedding models. Document this limitation prominently in the README when Phase 5 ships.

**G3 — Session-end evaluator sees the entire conversation history**

For long sessions (hundreds of turns), the message history passed to `run_session_end_evaluator` may approach or exceed the model's context window. The evaluator prompt + all messages could be 10k+ tokens.

No mitigation in Phase 5. Document as a known limitation. Phase 6b compression is the long-term solution.

**G4 — Cold-start embed call adds latency to the first message of a session**

If the previous session had a closing journal, the first turn of the new session makes an extra Ollama embed call before the character LLM call. This adds ~200–500 ms. The client already shows a "Generating response…" indicator; the extra latency is invisible to the user but the turn will be noticeably slower.

No mitigation in Phase 5. Phase 6a budget visibility work may add better per-turn latency instrumentation.

**G5 — `run_turn` now makes an embed call even when there are no experiences**

`retrieve_experiences` calls `get_experiences_with_embeddings` first. If the result is empty (no experiences in the DB), it returns `[]` without calling `ollama.embed`. The embed call only fires when there are experiences to compare against. This is important to verify in tests.

**G6 — Concurrent sessions for the same character share the experience pool**

If two sessions are running simultaneously for the same character (e.g., two browser windows), `_session_active_experiences` tracks them independently by `session_id`. Both can independently retrieve from the shared experience pool in the DB. Experiences deleted via `experience_update` in one session are immediately gone from the DB and will not appear in new retrievals for the other session. This is correct behaviour.

**G7 — `contradicted_experience_id` in evaluator response may be invalid**

The evaluator LLM can hallucinate an ID not present in the active experiences list. `run_turn` handles this by catching `NotFoundError` from `delete_experience` and logging a warning rather than raising. The invalid ID is silently ignored.

**G8 — The end-session endpoint is now a slow operation**

Previously it was a trivial DB update. Now it runs a full LLM call (session-end evaluator). For long sessions, this adds several seconds. The client must handle this asynchronously with a loading indicator. The endpoint does not stream; it blocks until the LLM finishes.

**G9 — Experiences section in system prompt grows without bound**

Each newly-retrieved Experience adds text to the system prompt. A character with 100 Experiences, all retrieved across a long session, would inject 100 entries into every subsequent turn's prompt. No cap is enforced in Phase 5. Phase 6a token accounting will surface this; Phase 6b compression is the remedy.

**G10 — `build_system_prompt` signature change is backward-compatible**

Existing callers that pass only `(character, facts, inferences)` continue to work. The `experiences` parameter defaults to `None` and the section is omitted. Tests that construct prompts without experiences do not need to be updated.

---

## Test Plan

Tests are written first. The implementation is complete when all Phase 5 tests pass alongside all existing Phase 1–4 tests, and overall coverage stays at or above 80%.

Ollama embed calls are mocked with `respx` using the `POST /api/embed` endpoint. The mock returns a minimal valid embedding (e.g., a 4-dimensional vector of 1.0s for unit tests, a properly shaped 768-dim vector where exact values matter).

**Test isolation for `_session_active_experiences`:** The module-level dict in `experience_service.py` persists across test cases in the same process. Add an `autouse` function-scoped fixture in `tests/unit/conftest.py` that clears the dict before every test:

```python
import pytest
from memories.services import experience_service

@pytest.fixture(autouse=True)
def _clear_active_experiences():
    experience_service._session_active_experiences.clear()
    yield
    experience_service._session_active_experiences.clear()
```

Apply the same fixture (or an equivalent `clear_active_experiences` call) in `tests/integration/conftest.py` for integration tests that exercise the chat endpoint.

---

### Unit tests — `tests/unit/`

#### New file: `tests/unit/test_experience_service.py`

**Embedding helpers**

| # | Test name | Asserts |
|---|-----------|---------|
| 1 | `test_embedding_round_trip` | `_blob_to_embedding(_embedding_to_blob([1.0, 2.0, 3.0]))` == `[1.0, 2.0, 3.0]` |
| 2 | `test_embedding_to_blob_returns_bytes` | `_embedding_to_blob([0.1, 0.2])` returns `bytes` |
| 3 | `test_blob_to_embedding_returns_list_of_floats` | Decoded result contains floats, not strings |

**Dot product / similarity**

| # | Test name | Asserts |
|---|-----------|---------|
| 4 | `test_dot_product_of_identical_unit_vectors_is_one` | `_dot([1.0, 0.0], [1.0, 0.0])` == `1.0` |
| 5 | `test_dot_product_of_orthogonal_vectors_is_zero` | `_dot([1.0, 0.0], [0.0, 1.0])` == `0.0` |
| 6 | `test_dot_product_of_opposite_unit_vectors_is_minus_one` | `_dot([1.0, 0.0], [-1.0, 0.0])` == `−1.0` |
| 7 | `test_dot_raises_value_error_on_mismatched_lengths` | `_dot([1.0, 0.0], [1.0])` raises `ValueError` (`strict=True` in `zip`) |

**`retrieve_top_k`**

| # | Test name | Asserts |
|---|-----------|---------|
| 8 | `test_retrieve_top_k_returns_most_similar` | Given three experiences with known similarities, top-1 is the most similar |
| 9 | `test_retrieve_top_k_respects_k_limit` | With 5 candidates and `k=2`, returns exactly 2 |
| 10 | `test_retrieve_top_k_excludes_active_ids` | An experience whose `id` is in `exclude_ids` is not returned even if most similar |
| 11 | `test_retrieve_top_k_returns_empty_when_all_excluded` | All candidates in `exclude_ids` → empty list |
| 12 | `test_retrieve_top_k_returns_empty_for_empty_candidates` | Empty candidate list → empty list |
| 13 | `test_retrieve_top_k_handles_k_greater_than_candidates` | `k=10` with 3 candidates → returns all 3 |

**`get_active_experiences` / `add_active_experiences` / `remove_active_experience`**

| # | Test name | Asserts |
|---|-----------|---------|
| 14 | `test_get_active_experiences_returns_empty_for_unknown_session` | `get_active_experiences(99)` → `[]` |
| 15 | `test_add_active_experiences_adds_to_session` | After `add_active_experiences(1, [exp])`, `get_active_experiences(1)` contains `exp` |
| 16 | `test_add_active_experiences_deduplicates_by_id` | Adding the same experience twice → appears only once |
| 17 | `test_add_active_experiences_does_not_affect_other_sessions` | Adding to session 1 does not appear in session 2 |
| 18 | `test_remove_active_experience_removes_by_id` | After `remove_active_experience(1, exp.id)`, `get_active_experiences(1)` is empty |
| 19 | `test_remove_active_experience_no_op_for_unknown_id` | `remove_active_experience(1, 9999)` does not raise |
| 20 | `test_clear_active_experiences_empties_session` | After adding, then calling `clear_active_experiences(1)`, `get_active_experiences(1)` is `[]` |
| 21 | `test_clear_active_experiences_no_op_for_unknown_session` | `clear_active_experiences(99)` does not raise |

**`retrieve_experiences` (mocked Ollama)**

For these tests, mock `POST /api/embed` with `respx` to return a fixed 768-dim embedding. The DB fixture has pre-inserted experiences with known embeddings.

| # | Test name | Asserts |
|---|-----------|---------|
| 22 | `test_retrieve_experiences_calls_embed_endpoint` | `POST /api/embed` is called with the query text |
| 23 | `test_retrieve_experiences_sends_embed_model_name` | Request body contains the configured embed model name |
| 24 | `test_retrieve_experiences_returns_top_k_new_experiences` | With 5 DB experiences and `top_k=2`, the returned new-experiences list has 2 entries |
| 25 | `test_retrieve_experiences_skips_excluded_ids_in_new_experiences` | Experience in `exclude_ids` is not in the returned new-experiences list |
| 26 | `test_retrieve_experiences_excluded_id_still_appears_in_scores` | Experience in `exclude_ids` IS still present in the returned scores dict |
| 27 | `test_retrieve_experiences_scores_covers_all_stored_experiences` | With 5 DB experiences, the scores dict has 5 entries |
| 28 | `test_retrieve_experiences_scores_are_floats` | Every value in the scores dict is a `float` |
| 29 | `test_retrieve_experiences_returns_empty_tuple_when_no_db_entries` | DB has no experiences → returns `([], {})` without calling embed |
| 30 | `test_retrieve_experiences_returns_empty_tuple_when_no_embeddings` | All DB experiences have `NULL` embedding → returns `([], {})` without calling embed |
| 31 | `test_retrieve_experiences_returns_empty_tuple_on_ollama_error` | Embed endpoint returns non-200 → `OllamaResponseError` is caught; `([], {})` returned without raising |

**`cold_start_retrieve` (mocked Ollama + DB)**

| # | Test name | Asserts |
|---|-----------|---------|
| 32 | `test_cold_start_retrieve_returns_empty_when_no_previous_session` | No previous session with closing journal → `[]`, no embed call |
| 33 | `test_cold_start_retrieve_returns_empty_when_no_closing_journal` | Previous session exists but `closing_journal` is NULL → `[]`, no embed call |
| 34 | `test_cold_start_retrieve_embeds_closing_journal` | Previous session has closing journal → embed called with journal text |
| 35 | `test_cold_start_retrieve_returns_retrieved_experiences` | Returns the top-k experiences matched against the journal embedding |
| 36 | `test_cold_start_retrieve_returns_list_not_tuple` | Return value is `list[Experience]`; confirms correct unpacking of the inner `retrieve_experiences` tuple |

**Session-end evaluator prompt**

| # | Test name | Asserts |
|---|-----------|---------|
| 37 | `test_session_end_prompt_includes_character_name` | Character name appears in built prompt |
| 38 | `test_session_end_prompt_includes_all_facts` | Each fact key and value appears |
| 39 | `test_session_end_prompt_includes_all_inferences` | Each inference statement appears |
| 40 | `test_session_end_prompt_includes_all_messages` | Every message content appears in turn order |
| 41 | `test_session_end_prompt_labels_user_messages` | User messages are labelled "User" |
| 42 | `test_session_end_prompt_labels_character_messages` | Character messages are labelled with the character's name |
| 43 | `test_session_end_prompt_includes_task_instructions` | Prompt contains instructions about closing journal and proposed experiences |
| 44 | `test_session_end_prompt_no_facts_shows_fallback` | Empty facts list → "(none)" fallback, no crash |
| 45 | `test_session_end_prompt_no_inferences_shows_fallback` | Empty inferences list → "(none)" fallback, no crash |
| 46 | `test_session_end_prompt_no_messages_shows_empty_section` | Empty message list → session section exists but is empty |

**`run_session_end_evaluator` (mocked Ollama)**

| # | Test name | Asserts |
|---|-----------|---------|
| 47 | `test_session_end_evaluator_returns_closing_journal` | LLM returns valid JSON → `result.closing_journal` is populated |
| 48 | `test_session_end_evaluator_returns_proposed_experiences` | LLM returns two proposals → `result.proposed_experiences` has two entries |
| 49 | `test_session_end_evaluator_experience_has_statement` | Each proposal has non-empty `statement` |
| 50 | `test_session_end_evaluator_experience_has_source` | Each proposal has `source` of `"told_by_user"` or `"observed"` |
| 51 | `test_session_end_evaluator_empty_proposals_is_valid` | LLM returns `proposed_experiences: []` → `SessionEndResult` with empty list |
| 52 | `test_session_end_evaluator_raises_on_non_json` | LLM returns plain text → `SessionEndParseError` raised |
| 53 | `test_session_end_evaluator_raises_on_missing_closing_journal` | LLM omits `closing_journal` field → `SessionEndParseError` raised |
| 54 | `test_session_end_evaluator_sends_think_false` | Captured Ollama request body has `"think": false` |
| 55 | `test_session_end_evaluator_sends_format_json` | Captured Ollama request body has `"format": "json"` |
| 56 | `test_session_end_evaluator_strips_markdown_code_fences` | LLM wraps JSON in ``` fences → still parsed correctly |

---

#### `tests/unit/test_prompt_builder.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 57 | `test_experiences_section_included_when_active` | Non-empty `experiences` arg → `## Your Experiences` section appears |
| 58 | `test_experience_statement_appears_in_prompt` | Experience `statement` text appears verbatim |
| 59 | `test_experience_source_told_by_user_labelled` | `source="told_by_user"` → label contains "told by user" (case-insensitive) |
| 60 | `test_experience_source_observed_labelled` | `source="observed"` → label contains "observed" |
| 61 | `test_multiple_experiences_all_appear` | Three active experiences → all three statements present |
| 62 | `test_experiences_section_absent_when_empty_list` | `experiences=[]` → no experiences section header |
| 63 | `test_experiences_section_absent_when_none` | `experiences=None` → no experiences section header |
| 64 | `test_experiences_section_follows_inferences` | Experiences section appears after inferences section |

---

#### `tests/unit/test_evaluator_service.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 65 | `test_evaluator_prompt_includes_active_experiences` | Active experiences appear in the prompt under their own section |
| 66 | `test_evaluator_prompt_no_experiences_uses_fallback` | Empty experience list → fallback `(no active experiences this session)` |
| 67 | `test_evaluator_prompt_includes_experience_ids` | Experience ids appear in the prompt so the LLM can cite them |
| 68 | `test_evaluator_prompt_includes_experience_source_label` | Each experience listing includes its source annotation |
| 69 | `test_evaluator_prompt_contains_experience_update_instructions` | Prompt contains text explaining when to return `experience_update` |
| 70 | `test_evaluator_prompt_contains_experience_update_json_schema` | Prompt schema includes `experience_updates` field |
| 71 | `test_run_evaluator_accepts_experiences_parameter` | `run_evaluator(experiences=[...])` does not raise |
| 72 | `test_run_evaluator_returns_experience_update_verdict` | Mocked evaluator returns `experience_update` → `EvaluatorResult.verdict == "experience_update"` |
| 73 | `test_run_evaluator_parses_experience_updates_list` | Mocked response has two `experience_updates` entries → `result.experience_updates` has two `ExperienceUpdate` objects |
| 74 | `test_run_evaluator_experience_update_not_coerced_to_pass` | `experience_update` verdict is returned as-is, not changed to `pass` |
| 75 | `test_evaluator_result_model_has_experience_updates_field` | `EvaluatorResult()` has `experience_updates: list[ExperienceUpdate] = []` |
| 76 | `test_contradiction_takes_priority_over_experience_update` | Mocked LLM returns `verdict: "contradiction"` with `violations` containing a contradiction type AND `experience_updates` populated → `result.verdict == "contradiction"`; the `experience_updates` payload does not change the verdict |

---

#### `tests/unit/test_chat_service.py` — additions

These tests require mocking the embed endpoint in addition to the chat endpoint.

| # | Test name | Asserts |
|---|-----------|---------|
| 77 | `test_run_turn_retrieves_experiences_for_user_message` | When the DB has experiences, the embed endpoint is called with the user message |
| 78 | `test_run_turn_adds_retrieved_experiences_to_active_set` | After `run_turn`, `get_active_experiences(session_id)` is non-empty |
| 79 | `test_run_turn_no_embed_call_when_no_experiences_in_db` | DB has no experiences → embed endpoint NOT called |
| 80 | `test_run_turn_includes_active_experiences_in_system_prompt` | Active experiences text appears in the first Ollama chat request's system message |
| 81 | `test_run_turn_cold_start_embeds_previous_journal` | First turn of a new session with a previous journal → embed called with journal text |
| 82 | `test_run_turn_cold_start_seeds_active_experiences` | After cold start, `get_active_experiences(session_id)` includes experiences from previous session |
| 83 | `test_run_turn_no_cold_start_when_no_previous_session` | No previous session → embed not called for cold start; embed called only for user message retrieval |
| 84 | `test_run_turn_cold_start_only_fires_on_first_turn` | Second call to `run_turn` with same session does NOT re-embed the previous journal |
| 85 | `test_run_turn_active_set_accumulates_across_turns` | Two successive `run_turn` calls retrieve different experiences → active set grows |
| 86 | `test_run_turn_experience_update_deletes_experience_from_db` | Evaluator returns `experience_update` with a known experience id → experience row absent from DB after the call |
| 87 | `test_run_turn_experience_update_removes_from_active_set` | After `experience_update`, the contradicted experience is not in the active set |
| 88 | `test_run_turn_experience_update_invalid_id_logs_warning_but_does_not_raise` | `experience_update` with unknown `contradicted_experience_id` → no exception, just a logged warning |
| 89 | `test_run_turn_passes_active_experiences_to_evaluator` | Second Ollama call (evaluator) request body contains the active experience text |
| 90 | `test_run_turn_experience_update_promotes_logical_inferences` | Mocked evaluator returns `verdict: "experience_update"` with a `new_inferences` entry of type `"logical"` → inference row is created in DB |
| 91 | `test_run_turn_returns_five_tuple_with_scores_dict` | Return value of `run_turn` is a 5-element tuple; fifth element is a `dict` keyed by experience id |

---

#### `tests/unit/test_ollama_client.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 92 | `test_embed_calls_api_embed_endpoint` | `await ollama.embed(model, text)` posts to `/api/embed` |
| 93 | `test_embed_sends_model_and_input` | Request body contains `{"model": ..., "input": text}` |
| 94 | `test_embed_returns_first_embedding` | Response `{"embeddings": [[1.0, 2.0]]}` → returns `[1.0, 2.0]` |
| 95 | `test_embed_raises_connection_error_on_network_failure` | ConnectError from httpx → `OllamaConnectionError` raised |
| 96 | `test_embed_raises_response_error_on_non_200` | Ollama returns 404 → `OllamaResponseError` raised |
| 97 | `test_embed_raises_response_error_on_empty_embeddings` | Ollama returns `{"embeddings": []}` → `OllamaResponseError` raised |

---

### Integration tests — `tests/integration/`

#### New file: `tests/integration/test_experiences_repo.py`

Fixture: a character and a session. The `embed_bytes` fixture generates deterministic fake embeddings as JSON bytes.

| # | Test name | Asserts |
|---|-----------|---------|
| 98 | `test_get_experience_returns_experience_by_id` | `get_experience(db, experience_id)` returns the `Experience` for a known id |
| 99 | `test_get_experience_returns_none_for_unknown_id` | `get_experience(db, 99999)` returns `None` |
| 100 | `test_create_experience_returns_experience` | `create_experience(...)` returns an `Experience` model |
| 101 | `test_create_experience_persists_to_db` | After create, raw DB query shows the row exists |
| 102 | `test_create_experience_statement_stored` | Returned `Experience.statement` matches input |
| 103 | `test_create_experience_source_told_by_user` | `source="told_by_user"` → `Experience.source == "told_by_user"` |
| 104 | `test_create_experience_source_observed` | `source="observed"` → `Experience.source == "observed"` |
| 105 | `test_create_experience_approved_at_is_set` | `Experience.approved_at` is not None |
| 106 | `test_get_experiences_returns_all_for_character` | After creating 3 experiences, `get_experiences(character_id)` returns all 3 |
| 107 | `test_get_experiences_returns_empty_list_when_none` | No experiences in DB → `get_experiences` returns `[]` |
| 108 | `test_get_experiences_filters_by_character_id` | Experiences from two different characters are not mixed |
| 109 | `test_get_experiences_returned_in_creation_order` | Three experiences inserted in known order → returned oldest-first (`ORDER BY created_at`) |
| 110 | `test_get_experiences_excludes_embedding_from_model` | Returned `Experience` objects do not cause serialisation errors (embedding is stripped) |
| 111 | `test_get_experiences_with_embeddings_returns_vectors` | `get_experiences_with_embeddings` returns tuples of `(Experience, list[float])` |
| 112 | `test_get_experiences_with_embeddings_decodes_blob` | Stored embedding blob is decoded back to the original float list |
| 113 | `test_get_experiences_with_embeddings_skips_null_embedding` | An experience with `embedding=NULL` is excluded from results |
| 114 | `test_delete_experience_returns_none_on_success` | `delete_experience` does not raise for a valid id |
| 115 | `test_delete_experience_removes_row_from_db` | After delete, `get_experiences` does not include the deleted id |
| 116 | `test_delete_experience_raises_not_found_for_unknown_id` | `delete_experience(99999)` → `NotFoundError` |
| 117 | `test_update_session_closing_journal_stores_text` | `update_session_closing_journal(session_id, "text")` → re-fetched session has `closing_journal == "text"` |
| 118 | `test_update_session_closing_journal_raises_not_found` | `update_session_closing_journal(99999, "text")` → `NotFoundError` |
| 119 | `test_get_previous_session_returns_most_recent_with_journal` | Two prior sessions, only second has a journal → returns second |
| 120 | `test_get_previous_session_excludes_session_without_journal` | Prior session has `closing_journal=NULL` → returns `None` |
| 121 | `test_get_previous_session_excludes_same_and_later_sessions` | Only looks at sessions with `id < before_session_id` |
| 122 | `test_get_previous_session_returns_none_when_no_prior_sessions` | No prior sessions → returns `None` |

---

#### New file: `tests/integration/test_api_experiences.py`

Fixtures: a character with at least one session. Because `POST /sessions/{id}/end` now makes an LLM call, do not use the end-session endpoint to create the test session's ended state — instead insert the session record with `ended_at` set directly via a DB helper in the fixture. Ollama embed calls mocked with `respx`.

**Experience creation (from accepted proposal)**

| # | Test name | Asserts |
|---|-----------|---------|
| 123 | `test_create_experience_returns_201` | `POST /characters/{id}/experiences` → 201 |
| 124 | `test_create_experience_response_has_statement` | Response body contains `statement` matching request |
| 125 | `test_create_experience_response_has_source` | Response body contains `source` matching request |
| 126 | `test_create_experience_response_has_id` | Response body has an integer `id` |
| 127 | `test_create_experience_response_has_no_embedding_field` | Response body does NOT contain an `embedding` key |
| 128 | `test_create_experience_persisted_to_db` | After create, `GET /characters/{id}/experiences` includes the new experience |
| 129 | `test_create_experience_calls_embed_endpoint` | `POST /api/embed` was called with the statement text |
| 130 | `test_create_experience_unknown_character_returns_404` | `character_id` not in DB → 404 |
| 131 | `test_create_experience_unknown_session_returns_404` | `session_id` not in DB → 404 |
| 132 | `test_create_experience_session_wrong_character_returns_404` | Session belongs to a different character → 404 |
| 133 | `test_create_experience_invalid_source_returns_422` | `source="unknown"` → 422 |
| 134 | `test_create_experience_told_by_user_source_accepted` | `source="told_by_user"` → 201 |
| 135 | `test_create_experience_observed_source_accepted` | `source="observed"` → 201 |

**Experience list**

| # | Test name | Asserts |
|---|-----------|---------|
| 136 | `test_list_experiences_returns_200` | `GET /characters/{id}/experiences` → 200 |
| 137 | `test_list_experiences_returns_all_experiences` | Three created experiences → list has three items |
| 138 | `test_list_experiences_returns_empty_list_when_none` | No experiences → `[]` |
| 139 | `test_list_experiences_unknown_character_returns_404` | Unknown `character_id` → 404 |
| 140 | `test_list_experiences_has_no_embedding_field_in_items` | Each item in the list has no `embedding` key |

**Experience delete**

| # | Test name | Asserts |
|---|-----------|---------|
| 141 | `test_delete_experience_returns_204` | `DELETE /characters/{id}/experiences/{exp_id}` → 204 |
| 142 | `test_delete_experience_removes_from_db` | After delete, `GET /characters/{id}/experiences` does not include deleted id |
| 143 | `test_delete_experience_unknown_id_returns_404` | Unknown `experience_id` → 404 |
| 144 | `test_delete_experience_unknown_character_returns_404` | Unknown `character_id` → 404 |
| 145 | `test_delete_experience_wrong_character_returns_404` | Experience belongs to character A; request uses character B's id → 404 (ownership check from L3 fix) |

---

#### `tests/integration/test_api_sessions.py` — additions and changes

**Existing test to update:** The test currently named `test_end_session_returns_session` (or similar) asserts that `POST /sessions/{id}/end` returns a bare `Session` JSON object. This now fails because the response shape changed to `{"session": {...}, "closing_journal": "...", "proposed_experiences": [...]}`. Update that test to: (1) mock the Ollama session-end evaluator call with `respx`, (2) assert `response.json()["session"]["ended_at"]` is set, and (3) assert the two new keys are present.

| # | Test name | Asserts |
|---|-----------|---------|
| 146 | `test_end_session_returns_200` | `POST /sessions/{id}/end` → 200 |
| 147 | `test_end_session_response_has_session` | Response body has a `session` key with `ended_at` set |
| 148 | `test_end_session_response_has_closing_journal` | Response body has a `closing_journal` key (may be empty string on parse failure) |
| 149 | `test_end_session_response_has_proposed_experiences` | Response body has `proposed_experiences` key (list, may be empty) |
| 150 | `test_end_session_stores_closing_journal_in_db` | After end, reloading the session shows `closing_journal` is not NULL |
| 151 | `test_end_session_already_ended_returns_409` | Calling `POST /sessions/{id}/end` twice → second call returns 409 |
| 152 | `test_end_session_unknown_session_returns_404` | Unknown `session_id` → 404 |
| 153 | `test_end_session_no_messages_returns_empty_proposals` | Session with no messages → `proposed_experiences: []` |
| 154 | `test_end_session_evaluator_parse_failure_returns_empty_proposals` | Mocked Ollama returns garbage → 200 with `proposed_experiences: []` and `closing_journal: ""` |
| 155 | `test_end_session_evaluator_parse_failure_still_ends_session` | Even on parse failure, `session.ended_at` is set in DB |
| 156 | `test_end_session_calls_session_end_evaluator_ollama` | The Ollama mock records a call during end-session |
| 157 | `test_end_session_proposed_experience_has_statement` | Each proposal in response has a `statement` field |
| 158 | `test_end_session_proposed_experience_has_source` | Each proposal has a `source` field of `"told_by_user"` or `"observed"` |
| 159 | `test_end_session_clears_active_experiences_for_session` | After end, `get_active_experiences(session_id)` returns `[]` |

---

#### `tests/integration/test_api_chat.py` — additions

Requires the embed mock (`POST /api/embed`) in addition to the existing Ollama chat mock.

| # | Test name | Asserts |
|---|-----------|---------|
| 160 | `test_send_message_no_embed_call_when_no_experiences` | Character has no experiences in DB → embed endpoint NOT called |
| 161 | `test_send_message_embed_call_when_experiences_exist` | One experience in DB → embed endpoint IS called with user message text |
| 162 | `test_send_message_experience_appears_in_system_prompt` | Active experience statement appears in Ollama chat request system message |
| 163 | `test_send_message_experience_update_verdict_deletes_experience` | Evaluator returns `experience_update` with experience id → experience row absent from DB after SSE completes |
| 164 | `test_send_message_experience_update_emits_sidechannel_event` | SSE stream contains `sidechannel` event with `type: "experience_update"` |
| 165 | `test_send_message_experience_update_sidechannel_has_contradicted_id` | Sidechannel event payload contains `contradicted_experience_id` |
| 166 | `test_send_message_experience_update_sidechannel_after_message_before_done` | Parsing the full SSE stream: the `experience_update` sidechannel event appears after the `message` event and before the `done` event |
| 167 | `test_send_message_experience_update_delivers_message` | Response message is still emitted (not withheld) when `experience_update` is the verdict |
| 168 | `test_send_message_cold_start_embeds_previous_journal` | Session 2 for same character; session 1 has a closing journal; first message → embed called with journal text |
| 169 | `test_send_message_cold_start_seeds_active_set` | After cold start, system prompt contains experience from previous session |
| 170 | `test_send_message_message_event_includes_active_experience_ids` | SSE `message` event payload contains `active_experience_ids` key with a list of integers |
| 171 | `test_send_message_active_experience_ids_matches_retrieved` | The IDs in `active_experience_ids` correspond to the experiences injected into the system prompt for that turn |
| 172 | `test_send_message_active_experience_ids_empty_when_no_experiences` | When no experiences exist in DB, `active_experience_ids` is `[]` |
| 173 | `test_send_message_active_experience_ids_accumulates_across_turns` | Second turn retrieves a different experience; `active_experience_ids` in the second `message` event contains IDs from both turns |
| 174 | `test_send_message_message_event_includes_experience_scores` | SSE `message` event payload contains `experience_scores` key with a list of `{id, score}` objects |
| 175 | `test_send_message_experience_scores_covers_all_stored_experiences` | `experience_scores` contains one entry per stored experience, not just the active ones |
| 176 | `test_send_message_experience_scores_empty_when_no_experiences` | When no experiences exist in DB, `experience_scores` is `[]` |
| 177 | `test_send_message_experience_scores_scores_are_floats` | Each entry in `experience_scores` has a numeric `score` value |

---

#### `tests/integration/test_db_init.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 178 | `test_experiences_table_exists` | After `init_db`, `PRAGMA table_info(experiences)` returns rows |
| 179 | `test_experiences_table_has_embedding_column` | Column named `embedding` exists in `experiences` |
| 180 | `test_experiences_table_has_approved_at_column` | Column named `approved_at` exists in `experiences` |
| 181 | `test_sessions_table_has_closing_journal_column` | Column named `closing_journal` exists in `sessions` |

*(These tests confirm the schema that was already written in Phase 1 is still correct — this is a regression guard, not new schema.)*

---

### Frontend tests — `tests/frontend/chat.test.js` — additions

`sortExperiences` must be a named export from `chat.js` (e.g., `export function sortExperiences(experiences, activeIds, scoreMap)`) so Vitest can import and call it directly. The function takes the experience array, a `Set<number>` of active IDs, and a `Map<number, number>` of scores, and returns a sorted copy.

| # | Test name | Asserts |
|---|-----------|---------|
| 182 | `buildNotificationFromSidechannel_handles_experience_update_type` | Payload with `type: "experience_update"` → returns a non-null notification object |
| 183 | `buildNotificationFromSidechannel_experience_update_has_scType` | Returned notification has `scType: "experience_update"` |
| 184 | `buildNotificationFromSidechannel_experience_update_has_turn_id` | Returned notification has `turn_id` from the payload |
| 185 | `buildNotificationFromSidechannel_experience_update_has_experience_updates_array` | Returned notification has `experience_updates` array matching the payload |
| 186 | `buildNotificationFromSidechannel_experience_update_empty_updates_array` | Payload with `experience_updates: []` → notification has `experience_updates: []` |
| 187 | `apiEndSession_posts_to_correct_url` | `fetch` called with `POST /api/sessions/7/end` |
| 188 | `apiEndSession_uses_post_method` | Method is `POST` |
| 189 | `apiEndSession_sends_no_body` | No request body is sent |
| 190 | `apiCreateExperience_posts_to_correct_url` | `fetch` called with `POST /api/characters/7/experiences` |
| 191 | `apiCreateExperience_sends_session_id_in_body` | Request body JSON contains `session_id` |
| 192 | `apiCreateExperience_sends_statement_in_body` | Request body JSON contains `statement` |
| 193 | `apiCreateExperience_sends_source_in_body` | Request body JSON contains `source` |
| 194 | `apiCreateExperience_sends_told_by_user_source` | `apiCreateExperience(7, 3, "text", "told_by_user")` → body has `source: "told_by_user"` |
| 195 | `apiCreateExperience_sends_observed_source` | `apiCreateExperience(7, 3, "text", "observed")` → body has `source: "observed"` |
| 196 | `apiListExperiences_gets_correct_url` | `fetch` called with `GET /api/characters/7/experiences` |
| 197 | `apiListExperiences_uses_get_method` | Method is `GET` (default `fetch`) |
| 198 | `apiDeleteExperience_sends_delete_to_correct_url` | `fetch` called with `DELETE /api/characters/7/experiences/42` |
| 199 | `apiDeleteExperience_uses_delete_method` | Method is `DELETE` |
| 200 | `parseSSEBlock_message_event_exposes_active_experience_ids` | A `message` SSE block containing `"active_experience_ids": [3, 7]` → parsed data has the array |
| 201 | `parseSSEBlock_message_event_active_experience_ids_defaults_absent` | A `message` SSE block without `active_experience_ids` → parsed data has no key (caller must default to `[]`) |
| 202 | `parseSSEBlock_message_event_exposes_experience_scores` | A `message` SSE block containing `"experience_scores": [{id:3, score:0.8}]` → parsed data has the array |
| 203 | `sortExperiences_active_before_inactive` | Given one active and one inactive experience with the same score, active appears first |
| 204 | `sortExperiences_active_sorted_by_score_descending` | Two active experiences with scores 0.9 and 0.4 → higher score appears first |
| 205 | `sortExperiences_inactive_sorted_by_score_descending` | Two inactive experiences with scores 0.7 and 0.2 → higher score appears first |
| 206 | `sortExperiences_active_group_always_above_inactive_group` | Active experience with score 0.1 appears above inactive experience with score 0.9 |
| 207 | `sortExperiences_no_score_experience_falls_to_bottom` | Experience absent from score map ranks below all scored experiences |
| 208 | `sortExperiences_stable_when_scores_and_active_status_equal` | Two experiences with identical score and same active status maintain relative order |

---

## Not in Scope for Phase 5

- **`sqlite-vec` for vector search** — similarity is computed in Python. A future phase can migrate to `sqlite-vec` without changing the API or test surface.
- **Streaming the session-end evaluator** — `POST /sessions/{id}/end` is a blocking REST call. Future phases could stream the closing journal character-by-character, but the complexity outweighs the UX benefit for now.
- **Experience deduplication** — if the user accepts similar proposals across multiple sessions, near-duplicate experiences accumulate in the DB. No deduplication pass is implemented. This is a Phase 7+ concern.
- **User-initiated experience editing after approval** — once an experience is written to the DB, the user can only delete it. Editing would require a `PUT /experiences/{id}` endpoint that re-embeds the new statement; deferred.
- **Context budget tracking (Phase 6a)** — `prompt_eval_count`/`eval_count` from Ollama responses are not stored. Active experience accumulation is not bounded.
- **Compression (Phase 6b)** — no segment boundaries are created or used in Phase 5. All messages remain in a single `session_start` segment.
- **Modelfile export** — still deferred to Phase 7 stretch.
- **Experience injection from previous sessions at cold start beyond the closing journal** — the cold-start mechanism uses only the closing journal as the query. A more sophisticated approach (embedding the entire prior session's conversation) is not in scope.
