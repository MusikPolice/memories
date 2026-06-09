"""Unit-test fixtures: pre-populated DB objects and Ollama mock helpers."""

from __future__ import annotations

import json
from typing import Any

import aiosqlite
import pytest

from memories import database
from memories.database import create_character, create_fact, create_session
from memories.models import Character, Fact, Session
from memories.services import experience_service
from memories.services.ollama_client import OllamaClient

# Base URL used by all unit tests that mock the Ollama HTTP layer.
OLLAMA_BASE_URL = "http://test-ollama:11434"


def make_ollama_ndjson(
    *chunks: str,
    prompt_eval_count: int = 10,
    eval_count: int = 5,
    thinking: str = "",
) -> bytes:
    """Build a minimal Ollama NDJSON streaming body from content *chunks*.

    If *thinking* is provided a thinking chunk is prepended before the content
    chunks, matching the format Ollama uses for thinking-enabled models.
    The last content chunk carries ``done: true`` and the token-count metadata.
    """
    lines: list[str] = []
    if thinking:
        lines.append(
            json.dumps(
                {
                    "message": {"role": "assistant", "content": "", "thinking": thinking},
                    "done": False,
                }
            )
        )
    for i, text in enumerate(chunks):
        is_last = i == len(chunks) - 1
        obj: dict[str, object] = {
            "message": {"role": "assistant", "content": text},
            "done": is_last,
        }
        if is_last:
            obj["prompt_eval_count"] = prompt_eval_count
            obj["eval_count"] = eval_count
        lines.append(json.dumps(obj))
    return ("\n".join(lines) + "\n").encode()


def make_evaluator_ndjson(
    verdict: str = "pass",
    new_inferences: list[dict[str, Any]] | None = None,
    violations: list[dict[str, Any]] | None = None,
    decision_log: str = "Response is grounded and clean.",
    experience_updates: list[dict[str, Any]] | None = None,
) -> bytes:
    """Build a minimal Ollama NDJSON body whose content is an evaluator JSON verdict.

    Use this to mock the second Ollama call (evaluator) in any test that exercises
    ``run_turn``.  The returned bytes can be passed to
    ``httpx.Response(200, content=...)``.
    """
    data: dict[str, Any] = {
        "verdict": verdict,
        "new_inferences": new_inferences or [],
        "violations": violations or [],
        "decision_log": decision_log,
    }
    if experience_updates is not None:
        data["experience_updates"] = experience_updates
    return make_ollama_ndjson(json.dumps(data))


def make_extractor_ndjson(
    new_facts: list[dict[str, Any]] | None = None,
    fact_updates: list[dict[str, Any]] | None = None,
    implicit_proposals: list[dict[str, Any]] | None = None,
) -> bytes:
    """Build a minimal Ollama NDJSON body whose content is a fact-extractor JSON result.

    Use this to mock the first Ollama call (extractor) in any Phase 6 test that
    exercises ``run_turn``.  The returned bytes can be passed to
    ``httpx.Response(200, content=...)``.
    """
    data: dict[str, Any] = {
        "new_facts": new_facts or [],
        "fact_updates": fact_updates or [],
        "implicit_proposals": implicit_proposals or [],
    }
    return make_ollama_ndjson(json.dumps(data))


def make_embed_response(vec: list[float] | None = None) -> bytes:
    """Build a minimal Ollama embed API JSON response body."""
    if vec is None:
        vec = [1.0, 0.0, 0.0, 0.0]
    return json.dumps({"embeddings": [vec]}).encode()


@pytest.fixture(autouse=True)
def _clear_active_experiences() -> Any:
    experience_service._session_active_experiences.clear()
    yield
    experience_service._session_active_experiences.clear()


@pytest.fixture(autouse=True)
def _clear_embedding_cache() -> Any:
    database._experience_embedding_cache.clear()
    yield
    database._experience_embedding_cache.clear()


@pytest.fixture
async def character(db: aiosqlite.Connection) -> Character:
    return await create_character(db, name="Alice", modelfile_base="qwen3:7b")


@pytest.fixture
async def session(db: aiosqlite.Connection, character: Character) -> Session:
    return await create_session(db, character_id=character.id)


@pytest.fixture
async def fact(db: aiosqlite.Connection, character: Character) -> Fact:
    return await create_fact(db, character_id=character.id, key="occupation", value="surgeon")


@pytest.fixture
def ollama() -> OllamaClient:
    return OllamaClient(base_url=OLLAMA_BASE_URL)
