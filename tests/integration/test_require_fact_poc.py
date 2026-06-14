"""Integration PoC tests for the require_fact suspension mechanism (Step 0b).

Verifies:
- The SSE connection survives while a gate awaits user input
- Events emitted before and after gate resolution are both delivered
- Keep-alive ping comments flow during suspension
- The gate is cleaned up in all exit paths (normal, error, dismiss)

The PoC endpoint (POST /api/sessions/{id}/test-require-fact-poc) does not call
a real LLM; it simulates the suspension cycle directly.  All tests use an
in-memory SQLite DB via the ``db`` fixture from tests/conftest.py.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

import memories.services.tool_gate as tool_gate_module
from memories.database import create_character, create_session
from memories.deps import get_db
from memories.main import app

# Fixed turn_id used across tests so the PoC SSE endpoint and accept endpoint
# share a known gate key.
_TURN_ID = 42
_UNKNOWN_SESSION_ID = 999999


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_gates() -> Any:
    tool_gate_module._pending.clear()
    yield
    tool_gate_module._pending.clear()


@pytest.fixture
async def poc_client(db: aiosqlite.Connection) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient wired to the in-memory DB; no Ollama override needed."""

    async def _override_db() -> AsyncGenerator[aiosqlite.Connection, None]:
        yield db

    app.dependency_overrides[get_db] = _override_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


@pytest.fixture
async def session_id(db: aiosqlite.Connection) -> int:
    char = await create_character(db, name="PocChar", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)
    return sess.id


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------


def _poc_url(
    session_id: int,
    turn_id: int = _TURN_ID,
    delay_ms: int = 0,
    fail: bool = False,
) -> str:
    params = f"turn_id={turn_id}&delay_ms={delay_ms}"
    if fail:
        params += "&fail=1"
    return f"/api/sessions/{session_id}/test-require-fact-poc?{params}"


def _respond_url(session_id: int, turn_id: int = _TURN_ID) -> str:
    return f"/api/sessions/{session_id}/turns/{turn_id}/require-fact/respond"


# ---------------------------------------------------------------------------
# SSE consumer helper
# ---------------------------------------------------------------------------


class _SseCollector:
    """Incrementally collects SSE events from a streaming response.

    Sets ``sidechannel_seen`` when a ``require_fact`` sidechannel event arrives.
    Keeps all raw lines (including ``: ping`` comments) for keep-alive tests.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []
        self.raw_lines: list[str] = []
        self.sidechannel_seen: asyncio.Event = asyncio.Event()

    async def consume(self, client: AsyncClient, url: str) -> None:
        async with client.stream("POST", url) as response:
            current: dict[str, str] = {}
            async for line in response.aiter_lines():
                self.raw_lines.append(line)
                if line.startswith("event: "):
                    current = {"event": line[7:]}
                elif line.startswith("data: "):
                    current["data"] = line[6:]
                elif not line:
                    if current:
                        self.events.append(current.copy())
                        if current.get("event") == "sidechannel":
                            try:
                                parsed = json.loads(current.get("data", "{}"))
                                if parsed.get("type") == "require_fact":
                                    self.sidechannel_seen.set()
                            except json.JSONDecodeError:
                                pass
                    current = {}


async def _cancel_task(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_sse_stream_starts_before_suspension(
    poc_client: AsyncClient, session_id: int
) -> None:
    """status(generating) and the require_fact sidechannel arrive before the gate resolves."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)
    finally:
        await _cancel_task(consumer)

    # status(generating) must appear before the sidechannel
    assert any(e.get("event") == "status" for e in collector.events)
    status_events = [e for e in collector.events if e.get("event") == "status"]
    assert json.loads(status_events[0]["data"])["state"] == "generating"

    # sidechannel must carry the required fields
    sc_events = [e for e in collector.events if e.get("event") == "sidechannel"]
    assert len(sc_events) >= 1
    sc = json.loads(sc_events[0]["data"])
    assert sc.get("type") == "require_fact"
    assert "path" in sc
    assert "reason" in sc
    assert "suggested_value" in sc

    # stream must not have ended while gate is still open
    assert not any(e.get("event") == "done" for e in collector.events)


async def test_accept_with_value_resumes_stream(poc_client: AsyncClient, session_id: int) -> None:
    """Accepting with a value resumes the stream; message + done are delivered."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))

    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)

    response = await poc_client.post(_respond_url(session_id), json={"value": "Sarah"})
    assert response.status_code == 200

    await asyncio.wait_for(consumer, timeout=5.0)

    message_events = [e for e in collector.events if e.get("event") == "message"]
    assert len(message_events) == 1
    msg = json.loads(message_events[0]["data"])
    assert msg.get("content") == "Sarah"

    assert collector.events[-1]["event"] == "done"


async def test_dismiss_resolves_gate_with_none(poc_client: AsyncClient, session_id: int) -> None:
    """Dismissing (empty body) resolves the gate with None; stream ends with done."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))

    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)

    response = await poc_client.post(_respond_url(session_id), json={})
    assert response.status_code == 200

    await asyncio.wait_for(consumer, timeout=5.0)

    message_events = [e for e in collector.events if e.get("event") == "message"]
    assert len(message_events) == 1
    msg = json.loads(message_events[0]["data"])
    assert msg.get("content") == "(dismissed)"

    done_events = [e for e in collector.events if e.get("event") == "done"]
    assert len(done_events) == 1


async def test_events_before_suspension_are_delivered(
    poc_client: AsyncClient, session_id: int
) -> None:
    """status(generating) is present in the collected events before gate suspension."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)
    finally:
        await _cancel_task(consumer)

    status_events = [e for e in collector.events if e.get("event") == "status"]
    assert len(status_events) >= 1
    states = [json.loads(e["data"])["state"] for e in status_events]
    assert "generating" in states


async def test_keep_alive_comments_emitted_during_suspension(
    poc_client: AsyncClient, session_id: int
) -> None:
    """At least one `: ping` comment line is emitted while the gate is suspended."""
    collector = _SseCollector()
    consumer = asyncio.create_task(
        collector.consume(poc_client, _poc_url(session_id, delay_ms=100))
    )
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)
        response = await poc_client.post(_respond_url(session_id), json={"value": "Alice"})
        assert response.status_code == 200
        await asyncio.wait_for(consumer, timeout=5.0)
    except Exception:
        await _cancel_task(consumer)
        raise

    ping_lines = [line for line in collector.raw_lines if line.startswith(":")]
    assert (
        len(ping_lines) >= 1
    ), f"expected at least one ping comment, got raw_lines={collector.raw_lines}"


async def test_double_accept_returns_409(poc_client: AsyncClient, session_id: int) -> None:
    """A second call to the accept endpoint returns 409; first resolution stands."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))

    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)

    r1 = await poc_client.post(_respond_url(session_id), json={"value": "Alice"})
    r2 = await poc_client.post(_respond_url(session_id), json={"value": "Bob"})

    assert r1.status_code == 200
    assert r2.status_code == 409

    try:
        await asyncio.wait_for(consumer, timeout=5.0)
    except (TimeoutError, Exception):
        await _cancel_task(consumer)


async def test_accept_unknown_turn_returns_404(poc_client: AsyncClient, session_id: int) -> None:
    """Calling the accept endpoint with no matching gate returns 404."""
    response = await poc_client.post(
        _respond_url(session_id, turn_id=999999), json={"value": "Alice"}
    )
    assert response.status_code == 404


async def test_accept_before_suspension_resolves_immediately(
    poc_client: AsyncClient, session_id: int
) -> None:
    """A pre-queued resolution is consumed by await_gate() without blocking."""
    # Create the gate manually so the accept endpoint can find it
    tool_gate_module.create_gate(session_id, _TURN_ID)
    response = await poc_client.post(_respond_url(session_id), json={"value": "Preloaded"})
    assert response.status_code == 200

    # Now start the SSE stream — the gate value is already queued, so it
    # should resume immediately and emit done without ever blocking.
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))
    await asyncio.wait_for(consumer, timeout=5.0)

    message_events = [e for e in collector.events if e.get("event") == "message"]
    assert len(message_events) == 1
    assert json.loads(message_events[0]["data"])["content"] == "Preloaded"


async def test_gate_absent_after_normal_completion(
    poc_client: AsyncClient, session_id: int
) -> None:
    """The gate is removed from _pending after a successful accept + stream close."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))

    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)
    await poc_client.post(_respond_url(session_id), json={"value": "Alice"})
    await asyncio.wait_for(consumer, timeout=5.0)

    assert (session_id, _TURN_ID) not in tool_gate_module._pending


async def test_gate_absent_after_error(poc_client: AsyncClient, session_id: int) -> None:
    """The gate is removed from _pending even when the PoC endpoint raises an error."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id, fail=True)))
    try:
        await asyncio.wait_for(consumer, timeout=5.0)
    except (TimeoutError, Exception):
        await _cancel_task(consumer)

    assert (session_id, _TURN_ID) not in tool_gate_module._pending


async def test_session_not_found_returns_404(poc_client: AsyncClient) -> None:
    """The PoC SSE endpoint returns 404 immediately for unknown session IDs."""
    response = await poc_client.post(_poc_url(_UNKNOWN_SESSION_ID))
    assert response.status_code == 404


async def test_duplicate_gate_for_same_key_returns_409(
    poc_client: AsyncClient, session_id: int
) -> None:
    """A second PoC SSE request for the same (session_id, turn_id) returns 409."""
    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(poc_client, _poc_url(session_id)))

    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=5.0)

        # Second request for the same gate key — the gate already exists
        response2 = await poc_client.post(_poc_url(session_id))
        assert response2.status_code == 409
    finally:
        await _cancel_task(consumer)
