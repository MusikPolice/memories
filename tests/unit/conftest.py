"""Unit-test fixtures: pre-populated DB objects and Ollama mock helpers."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
import pytest

from memories import database
from memories.database import create_character, create_session
from memories.models import Character, Session
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


def make_tool_call_response(
    tool_name: str,
    arguments: dict[str, Any],
    prompt_eval_count: int = 10,
    eval_count: int = 5,
    call_id: str | None = None,
) -> bytes:
    """Build a non-streaming Ollama JSON response where the model calls one tool.

    Pass *call_id* to simulate models (e.g. qwen3) that include an ``id`` on
    each tool call; the client must echo it back as ``tool_call_id``.
    """
    tool_call: dict[str, Any] = {"function": {"name": tool_name, "arguments": arguments}}
    if call_id is not None:
        tool_call["id"] = call_id
    obj = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call],
        },
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    return json.dumps(obj).encode()


def make_multi_tool_call_response(
    calls: list[tuple[str, dict[str, Any]]],
    call_ids: list[str] | None = None,
) -> bytes:
    """Build a non-streaming Ollama JSON response where the model calls multiple tools.

    Pass *call_ids* (same length as *calls*) to simulate models that assign IDs
    to each call; the client must echo each one back as ``tool_call_id``.
    """
    tool_calls: list[dict[str, Any]] = []
    for i, (name, args) in enumerate(calls):
        tc: dict[str, Any] = {"function": {"name": name, "arguments": args}}
        if call_ids is not None:
            tc["id"] = call_ids[i]
        tool_calls.append(tc)
    obj = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": tool_calls,
        },
        "done": True,
    }
    return json.dumps(obj).encode()


def make_tool_call_response_with_thinking(
    tool_name: str,
    arguments: dict[str, Any],
    thinking: str = "I should call this tool.",
    prompt_eval_count: int = 10,
    eval_count: int = 5,
) -> bytes:
    """Build a non-streaming Ollama JSON response with a thinking field and one tool call.

    Used to test that thinking tokens are stripped from history before re-sending.
    """
    obj = {
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": thinking,
            "tool_calls": [{"function": {"name": tool_name, "arguments": arguments}}],
        },
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    return json.dumps(obj).encode()


def make_plain_tool_response(content: str, thinking: str = "") -> bytes:
    """Build a non-streaming Ollama JSON response with plain content and no tool calls.

    Pass *thinking* to simulate a final round that both thinks and replies in plain
    text — the shape `chat_with_tools()` must handle when `think=True` is requested
    for the Character LLM.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": None}
    if thinking:
        message["thinking"] = thinking
    obj = {"message": message, "done": True}
    return json.dumps(obj).encode()


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
async def ollama() -> AsyncGenerator[OllamaClient, None]:
    client = OllamaClient(base_url=OLLAMA_BASE_URL)
    yield client
    await client.aclose()
