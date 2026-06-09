"""Experience retrieval, embedding, and session-end evaluator."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Literal

import aiosqlite
from pydantic import BaseModel

from memories.database import (
    _embedding_to_blob,
    create_experience,
    get_experiences_with_embeddings,
    get_previous_session,
)
from memories.models import Character, Experience, Fact, Inference, Message
from memories.services.ollama_client import OllamaClient, OllamaConnectionError, OllamaResponseError

_log = logging.getLogger(__name__)

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
TOP_K_EXPERIENCES: int = int(os.getenv("TOP_K_EXPERIENCES", "5"))
MIN_EXPERIENCE_SCORE: float = float(os.getenv("MIN_EXPERIENCE_SCORE", "0.0"))


# ---------------------------------------------------------------------------
# Ephemeral LLM output models (not persisted to DB)
# ---------------------------------------------------------------------------


class ProposedExperience(BaseModel):
    statement: str
    source: Literal["told_by_user", "observed"]
    turn_reference: int | None = None


class SessionEndResult(BaseModel):
    closing_journal: str
    proposed_experiences: list[ProposedExperience]


class SessionEndParseError(Exception):
    """Raised when the session-end LLM returns unparseable output."""


# ---------------------------------------------------------------------------
# Dot-product similarity (nomic-embed-text outputs L2-normalised vectors)
# ---------------------------------------------------------------------------


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


# ---------------------------------------------------------------------------
# Top-k retrieval (pure, no I/O)
# ---------------------------------------------------------------------------


def retrieve_top_k(
    query: list[float],
    candidates: list[tuple[Experience, list[float]]],
    k: int,
    exclude_ids: set[int] | None = None,
    min_score: float = 0.0,
) -> list[Experience]:
    exclude_ids = exclude_ids or set()
    scored = [(exp, _dot(query, vec)) for exp, vec in candidates if exp.id not in exclude_ids]
    scored = [(exp, score) for exp, score in scored if score >= min_score]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [exp for exp, _ in scored[:k]]


# ---------------------------------------------------------------------------
# In-memory active-experience tracking (per session, per process lifetime)
# ---------------------------------------------------------------------------

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


def remove_experience_from_all_sessions(experience_id: int) -> None:
    """Remove *experience_id* from every active session's experience set.

    Called when an experience is deleted via the API so in-flight sessions
    do not continue injecting a deleted experience into the character prompt.
    """
    for sid in list(_session_active_experiences):
        _session_active_experiences[sid] = [
            e for e in _session_active_experiences[sid] if e.id != experience_id
        ]


# ---------------------------------------------------------------------------
# Experience retrieval (async, uses DB + Ollama embed)
# ---------------------------------------------------------------------------


async def retrieve_experiences(
    db: aiosqlite.Connection,
    character_id: int,
    query_text: str,
    ollama: OllamaClient,
    top_k: int = TOP_K_EXPERIENCES,
    exclude_ids: set[int] | None = None,
    min_score: float = MIN_EXPERIENCE_SCORE,
) -> tuple[list[Experience], dict[int, float]]:
    """Return (new_experiences, all_scores).

    Returns ([], {}) without calling embed if no stored experiences exist
    or if the embed model is unavailable.
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
    new_exps = retrieve_top_k(query_vec, candidates, top_k, exclude_ids, min_score)
    return new_exps, all_scores


async def cold_start_retrieve(
    db: aiosqlite.Connection,
    character_id: int,
    session_id: int,
    ollama: OllamaClient,
    top_k: int = TOP_K_EXPERIENCES,
    min_score: float = MIN_EXPERIENCE_SCORE,
) -> list[Experience]:
    """Embed the previous session's closing journal and retrieve top-k experiences."""
    prev = await get_previous_session(db, character_id, before_session_id=session_id)
    if prev is None or not prev.closing_journal:
        return []
    try:
        journal_vec = await ollama.embed(EMBED_MODEL, prev.closing_journal)
    except (OllamaConnectionError, OllamaResponseError) as exc:
        _log.warning("embed call failed for cold-start — skipping: %s", exc)
        return []
    candidates = await get_experiences_with_embeddings(db, character_id)
    if not candidates:
        return []
    return retrieve_top_k(journal_vec, candidates, top_k, min_score=min_score)


async def embed_and_store(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    session_id: int,
    statement: str,
    source: str,
    ollama: OllamaClient,
) -> Experience:
    """Embed *statement* and write the Experience to DB."""
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


# ---------------------------------------------------------------------------
# Session-end evaluator
# ---------------------------------------------------------------------------


def build_session_end_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],
    messages: list[Message],
) -> str:
    parts: list[str] = [
        f"Character: {character.name}",
        "\n## Character Facts",
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
   noticed about the person you were talking with. 2-5 sentences. Do NOT just summarise
   the plot; capture emotional texture, unresolved tensions, what felt significant.

2. A list of **proposed Experiences** — things the character learned or observed that
   are not already captured in their Facts and Inferences. Each Experience should be a
   single concrete, present-tense statement. Classify the source:
   - `told_by_user`: the user explicitly stated it
   - `observed`: the character inferred it from the user's behaviour or the conversation

Return JSON with this exact structure:

{
  "closing_journal": "...",
  "proposed_experiences": [
    {
      "statement": "...",
      "source": "told_by_user",
      "turn_reference": 4
    }
  ]
}

Return only the JSON object, no other text.
Propose between 0 and 5 Experiences. Only include things genuinely new to this
session that are not already in the Facts or Inferences above. If nothing new was
learned, return an empty list.""")

    return "\n".join(parts)


async def run_session_end_evaluator(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],
    messages: list[Message],
    ollama: OllamaClient,
) -> SessionEndResult:
    """Run the session-end LLM call and return parsed closing journal + proposals."""
    prompt = build_session_end_prompt(character, facts, inferences, messages)
    model = character.current_model_name or character.modelfile_base
    llm_messages: list[dict[str, str]] = [
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
