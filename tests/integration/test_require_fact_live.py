"""Integration tests for require_fact through the real /api/sessions/{id}/messages endpoint.

Replaces test_require_fact_poc.py.  Must use a real uvicorn server rather than
the ASGITransport-backed ``client`` fixture, because ASGITransport buffers the full
HTTP response before yielding any lines — which deadlocks any test that must POST to
the require-fact/respond endpoint while the SSE stream is still open.

Ollama calls are mocked by injecting a ``_ResponseQueue`` transport directly into
the OllamaClient used by the server.  This leaves the ``live_client``'s real TCP
transport untouched so its requests to uvicorn are never intercepted.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket
from collections.abc import AsyncGenerator, Callable

import aiosqlite
import httpx
import pytest
import uvicorn
from httpx import AsyncClient

import memories.services.tool_gate as tool_gate_module
from memories.database import create_character, create_session, get_facts
from memories.deps import get_db, get_ollama
from memories.main import app
from memories.services.ollama_client import OllamaClient
from tests.unit.conftest import (
    make_plain_tool_response,
    make_tool_call_response,
)

OLLAMA_BASE_URL = "http://test-ollama-integration:11434"


# ---------------------------------------------------------------------------
# Minimal mock transport — feeds pre-built responses in order
# ---------------------------------------------------------------------------


class _ResponseQueue(httpx.AsyncBaseTransport):
    """Async transport that returns a pre-built list of responses in order."""

    def __init__(self) -> None:
        self._responses: list[httpx.Response] = []
        self._idx = 0

    def load(self, responses: list[httpx.Response]) -> None:
        self._responses = responses
        self._idx = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._idx >= len(self._responses):
            raise AssertionError(f"_ResponseQueue exhausted at idx={self._idx} for {request.url}")
        resp = self._responses[self._idx]
        self._idx += 1
        return resp

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bound_socket() -> socket.socket:
    """Return a TCP socket already bound to a free loopback port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    return s


async def _wait_until(condition: Callable[[], bool]) -> None:
    while not condition():
        await asyncio.sleep(0.05)


@pytest.fixture
async def live_client(
    db: aiosqlite.Connection,
) -> AsyncGenerator[tuple[AsyncClient, _ResponseQueue], None]:
    """Yields (client, queue).

    ``client`` is backed by a real uvicorn server.  ``queue`` is a
    ``_ResponseQueue`` injected as the transport of the OllamaClient used by
    the server — call ``queue.load([...])`` in each test before issuing requests.

    Using a per-client transport avoids the global-patching problem of
    ``respx.mock()``, which would also intercept ``client``'s real TCP connection
    to uvicorn (see project_asgi_transport_streaming memory note).
    """
    queue = _ResponseQueue()
    ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL)
    await ollama_client._http.aclose()
    ollama_client._http = httpx.AsyncClient(
        transport=queue,
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0),
    )

    async def _override_db() -> AsyncGenerator[aiosqlite.Connection, None]:
        yield db

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_ollama] = lambda: ollama_client

    sock = _bound_socket()
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, loop="none", lifespan="off", log_level="error")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve(sockets=[sock]))

    try:
        await asyncio.wait_for(_wait_until(lambda: server.started), timeout=10.0)
    except TimeoutError as exc:
        server.should_exit = True
        await serve_task
        raise RuntimeError("uvicorn failed to start within 10 seconds") from exc

    async with AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        try:
            yield client, queue
        finally:
            server.should_exit = True
            await serve_task

    app.dependency_overrides.clear()
    await ollama_client._http.aclose()


@pytest.fixture
async def session_id(db: aiosqlite.Connection) -> int:
    char = await create_character(db, name="LiveChar", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)
    return sess.id


# ---------------------------------------------------------------------------
# SSE consumer helper
# ---------------------------------------------------------------------------


class _SseCollector:
    """Incrementally collects SSE events and raw lines from a streaming response."""

    def __init__(self) -> None:
        self.events: list[dict[str, str]] = []
        self.raw_lines: list[str] = []
        self.sidechannel_seen: asyncio.Event = asyncio.Event()

    async def consume(self, client: AsyncClient, url: str) -> None:
        async with client.stream("POST", url, json={"content": "Hello"}) as response:
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
# Shared mock response helpers
# ---------------------------------------------------------------------------


def _mock_world_builder() -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))


def _mock_require_fact_call(
    path: str = "Character.Identity.Name",
    reason: str = "I need my name to introduce myself",
    suggested_value: str = "Elena",
) -> httpx.Response:
    return httpx.Response(
        200,
        content=make_tool_call_response(
            "require_fact",
            {"path": path, "reason": reason, "suggested_value": suggested_value},
        ),
    )


def _mock_plain_reply(content: str = "Hi, I'm Sarah.") -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response(content))


def _mock_eval_pass() -> httpx.Response:
    return httpx.Response(200, content=make_tool_call_response("report_pass", {}))


def _respond_url(session_id: int, turn_id: int = 1) -> str:
    return f"/api/sessions/{session_id}/turns/{turn_id}/require-fact/respond"


def _messages_url(session_id: int) -> str:
    return f"/api/sessions/{session_id}/messages"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_require_fact_sidechannel_emitted_before_done(
    live_client: tuple[AsyncClient, _ResponseQueue],
    session_id: int,
) -> None:
    """A require_fact sidechannel event arrives before any done event."""
    client, queue = live_client
    queue.load(
        [
            _mock_world_builder(),
            _mock_require_fact_call(),
            _mock_plain_reply(),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(client, _messages_url(session_id)))
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

        sc_events = [e for e in collector.events if e.get("event") == "sidechannel"]
        rf_events = [
            e for e in sc_events if json.loads(e.get("data", "{}")).get("type") == "require_fact"
        ]
        assert len(rf_events) >= 1
        sc = json.loads(rf_events[0]["data"])
        assert sc["path"] == "Character.Identity.Name"
        assert "reason" in sc
        assert "suggested_value" in sc
        assert not any(e.get("event") == "done" for e in collector.events)
    finally:
        await _cancel_task(consumer)


async def test_require_fact_accept_resumes_stream_with_value(
    live_client: tuple[AsyncClient, _ResponseQueue],
    session_id: int,
) -> None:
    """Accepting with a value resumes the stream; message + done are delivered."""
    client, queue = live_client
    queue.load(
        [
            _mock_world_builder(),
            _mock_require_fact_call(),
            _mock_plain_reply("Hi, I'm Sarah."),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(client, _messages_url(session_id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

    response = await client.post(_respond_url(session_id), json={"value": "Sarah"})
    assert response.status_code == 200

    await asyncio.wait_for(consumer, timeout=10.0)

    message_events = [e for e in collector.events if e.get("event") == "message"]
    assert len(message_events) == 1
    assert json.loads(message_events[0]["data"])["content"] == "Hi, I'm Sarah."
    assert collector.events[-1]["event"] == "done"


async def test_require_fact_accept_writes_fact_to_db(
    db: aiosqlite.Connection,
    live_client: tuple[AsyncClient, _ResponseQueue],
) -> None:
    """After a successful accept cycle, the fact value is persisted in the DB."""
    client, queue = live_client
    char = await create_character(db, name="FactChar", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    queue.load(
        [
            _mock_world_builder(),
            _mock_require_fact_call(),
            _mock_plain_reply("Hi, I'm Sarah."),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)
    await client.post(_respond_url(sess.id), json={"value": "Sarah"})
    await asyncio.wait_for(consumer, timeout=10.0)

    facts = await get_facts(db, char.id)
    assert facts["Character"]["Identity"]["Name"]["Value"] == "Sarah"


async def test_require_fact_dismiss_resolves_with_none(
    db: aiosqlite.Connection,
    live_client: tuple[AsyncClient, _ResponseQueue],
) -> None:
    """Dismissing (empty body, value defaults to None) — stream completes, path stays unset."""
    client, queue = live_client
    char = await create_character(db, name="DismissChar", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    queue.load(
        [
            _mock_world_builder(),
            _mock_require_fact_call(),
            _mock_plain_reply("I'll proceed without it."),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

    response = await client.post(_respond_url(sess.id), json={})
    assert response.status_code == 200

    await asyncio.wait_for(consumer, timeout=10.0)

    message_events = [e for e in collector.events if e.get("event") == "message"]
    assert len(message_events) == 1
    done_events = [e for e in collector.events if e.get("event") == "done"]
    assert len(done_events) == 1
    facts = await get_facts(db, char.id)
    assert "Name" not in facts.get("Character", {}).get("Identity", {})


async def test_require_fact_keep_alive_during_suspension(
    live_client: tuple[AsyncClient, _ResponseQueue],
    session_id: int,
) -> None:
    """At least one `: ping` comment line is emitted while the gate is suspended."""
    client, queue = live_client
    queue.load(
        [
            _mock_world_builder(),
            _mock_require_fact_call(),
            _mock_plain_reply("Sarah here."),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(client, _messages_url(session_id)))
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)
        await asyncio.sleep(0.15)  # let keep-alive pings accumulate
        await client.post(_respond_url(session_id), json={"value": "Sarah"})
        await asyncio.wait_for(consumer, timeout=10.0)
    except Exception:
        await _cancel_task(consumer)
        raise

    ping_lines = [line for line in collector.raw_lines if line.startswith(":")]
    assert (
        len(ping_lines) >= 1
    ), f"expected at least one ping comment, got raw_lines={collector.raw_lines}"


async def test_require_fact_gate_removed_after_turn_completes(
    live_client: tuple[AsyncClient, _ResponseQueue],
    session_id: int,
) -> None:
    """The gate is removed from _pending after a successful accept + stream close."""
    client, queue = live_client
    queue.load(
        [
            _mock_world_builder(),
            _mock_require_fact_call(),
            _mock_plain_reply("Hi Sarah."),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector()
    consumer = asyncio.create_task(collector.consume(client, _messages_url(session_id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)
    await client.post(_respond_url(session_id), json={"value": "Sarah"})
    await asyncio.wait_for(consumer, timeout=10.0)

    # turn_id is 1 because this is the first turn in a fresh session
    assert (session_id, 1) not in tool_gate_module._pending


async def test_require_fact_respond_unknown_turn_returns_404(
    live_client: tuple[AsyncClient, _ResponseQueue],
    session_id: int,
) -> None:
    """POST to a non-existent gate returns 404."""
    client, _queue = live_client
    response = await client.post(
        f"/api/sessions/{session_id}/turns/9999/require-fact/respond",
        json={"value": "Alice"},
    )
    assert response.status_code == 404
