"""Unit tests for memories.services.tool_gate, SSEEvent, and EventCallback."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
import respx

from memories.services.chat_service import SSEEvent
from memories.services.ollama_client import OllamaClient
from memories.services.tool_gate import (
    await_gate,
    cleanup_gate,
    create_gate,
    resolve_gate,
)
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_plain_tool_response,
    make_tool_call_response,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_REQUIRE_FACT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "require_fact",
        "description": "Request a missing fact from the user",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


async def test_create_and_resolve_gate() -> None:
    create_gate(session_id=1, turn_id=42)
    resolve_gate(session_id=1, turn_id=42, value="Sarah")
    result = await await_gate(session_id=1, turn_id=42)
    assert result == "Sarah"


async def test_none_value_resolves_gate() -> None:
    create_gate(session_id=1, turn_id=1)
    resolve_gate(session_id=1, turn_id=1, value=None)
    result = await await_gate(session_id=1, turn_id=1)
    assert result is None


async def test_multiple_concurrent_gates() -> None:
    create_gate(1, 1)
    create_gate(1, 2)
    create_gate(2, 1)
    resolve_gate(1, 1, value="alpha")
    resolve_gate(1, 2, value="beta")
    resolve_gate(2, 1, value="gamma")
    assert await await_gate(1, 1) == "alpha"
    assert await await_gate(1, 2) == "beta"
    assert await await_gate(2, 1) == "gamma"


async def test_cleanup_removes_gate() -> None:
    create_gate(1, 42)
    cleanup_gate(1, 42)
    with pytest.raises(KeyError):
        await await_gate(1, 42)


async def test_resolve_before_await() -> None:
    create_gate(1, 99)
    resolve_gate(1, 99, value="preloaded")
    result = await await_gate(1, 99)
    assert result == "preloaded"


async def test_double_resolve_raises() -> None:
    create_gate(1, 10)
    resolve_gate(1, 10, value="first")
    with pytest.raises(asyncio.QueueFull):
        resolve_gate(1, 10, value="second")


async def test_cleanup_idempotent() -> None:
    cleanup_gate(99, 99)  # gate does not exist — should not raise


async def test_create_gate_duplicate_raises() -> None:
    create_gate(1, 1)
    with pytest.raises(ValueError, match="already exists"):
        create_gate(1, 1)


# ---------------------------------------------------------------------------
# Step 0b additions
# ---------------------------------------------------------------------------


@respx.mock
async def test_gate_resolves_through_callback_chain(ollama: OllamaClient) -> None:
    """A tool handler that awaits a gate receives the resolved value when a
    concurrent coroutine calls resolve_gate(); chat_with_tools() returns normally."""
    session_id, turn_id = 77, 99
    create_gate(session_id, turn_id)

    received: list[str | None] = []

    async def _require_fact_handler(args: dict[str, Any]) -> str:
        value = await await_gate(session_id, turn_id)
        received.append(value)
        return value if value is not None else "(dismissed)"

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response(
                    "require_fact", {"path": "Character.Identity.Name"}
                ),
            ),
            httpx.Response(200, content=make_plain_tool_response("Your name is Sarah.")),
        ]
    )

    async def _resolver() -> None:
        await asyncio.sleep(0.01)
        resolve_gate(session_id, turn_id, "Sarah")

    results = await asyncio.gather(
        ollama.chat_with_tools(
            "qwen3:7b",
            [{"role": "user", "content": "What is my name?"}],
            [_REQUIRE_FACT_TOOL],
            {"require_fact": _require_fact_handler},
        ),
        _resolver(),
    )

    tool_result = results[0]
    assert received == ["Sarah"]
    assert tool_result.content == "Your name is Sarah."


def test_sse_event_dataclass() -> None:
    """SSEEvent serialises to the expected SSE wire format."""
    event = SSEEvent(event="status", data={"state": "generating"})
    wire = event.to_sse()

    lines = wire.split("\n")
    assert lines[0] == "event: status"
    assert lines[1].startswith("data: ")
    data_payload = json.loads(lines[1][len("data: ") :])
    assert data_payload == {"state": "generating"}
    assert wire.endswith("\n\n")

    # sidechannel event with nested payload
    sc = SSEEvent(
        event="sidechannel",
        data={"type": "require_fact", "path": "Character.Identity.Name"},
    )
    sc_wire = sc.to_sse()
    assert sc_wire.startswith("event: sidechannel\n")
    sc_data = json.loads(sc_wire.split("\n")[1][len("data: ") :])
    assert sc_data["type"] == "require_fact"
