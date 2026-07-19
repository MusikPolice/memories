"""Integration tests for set_fact approval gates through the real SSE endpoint.

Must use a real uvicorn server (same pattern as test_require_fact_live.py) because
ASGITransport buffers full responses before returning — deadlocking any test that must
POST to the set-fact/respond endpoint while an SSE stream is still open.
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

from memories.database import create_character, create_session, get_facts
from memories.deps import get_db, get_ollama
from memories.main import app
from memories.services.ollama_client import OllamaClient
from tests.unit.conftest import (
    make_multi_tool_call_response,
    make_plain_tool_response,
    make_tool_call_response,
)

OLLAMA_BASE_URL = "http://test-ollama-integration:11434"


# ---------------------------------------------------------------------------
# Transport + helpers (identical pattern to test_require_fact_live.py)
# ---------------------------------------------------------------------------


class _ResponseQueue(httpx.AsyncBaseTransport):
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


def _bound_socket() -> socket.socket:
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


# ---------------------------------------------------------------------------
# SSE consumer
# ---------------------------------------------------------------------------


class _SseCollector:
    def __init__(self, target_sc_type: str) -> None:
        self.events: list[dict[str, str]] = []
        self.sidechannel_seen: asyncio.Event = asyncio.Event()
        self._target = target_sc_type

    async def consume(self, client: AsyncClient, url: str) -> None:
        async with client.stream("POST", url, json={"content": "Hello"}) as response:
            current: dict[str, str] = {}
            async for line in response.aiter_lines():
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
                                if parsed.get("type") == self._target:
                                    self.sidechannel_seen.set()
                            except json.JSONDecodeError:
                                pass
                    current = {}


async def _cancel_task(task: asyncio.Task) -> None:  # type: ignore[type-arg]
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_world_builder() -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))


def _mock_set_fact_pass(path: str, value: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=make_multi_tool_call_response(
            [("set_fact", {"path": path, "value": value}), ("report_pass", {})]
        ),
    )


def _mock_char_reply(content: str = "Sure, I can do that.") -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response(content))


def _mock_eval_pass() -> httpx.Response:
    return httpx.Response(200, content=make_tool_call_response("report_pass", {}))


def _messages_url(session_id: int) -> str:
    return f"/api/sessions/{session_id}/messages"


def _respond_url(session_id: int, turn_id: int = 1) -> str:
    return f"/api/sessions/{session_id}/turns/{turn_id}/set-fact/respond"


_MUT_PATH = "Character.Identity.Occupation"
_IMMU_PATH = "Character.Identity.Name"


# ---------------------------------------------------------------------------
# Tests — Mutable path
# ---------------------------------------------------------------------------


async def test_set_fact_mutable_sidechannel_emitted_before_done(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="MutChar1", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    # The character LLM produces a plain reply; the evaluator is what emits set_fact.
    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply(),
            _mock_set_fact_pass(_MUT_PATH, "surgeon"),
        ]
    )

    collector = _SseCollector("fact_update_mutable")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

        sc_events = [
            e
            for e in collector.events
            if e.get("event") == "sidechannel"
            and json.loads(e.get("data", "{}")).get("type") == "fact_update_mutable"
        ]
        assert len(sc_events) >= 1
        assert not any(e.get("event") == "done" for e in collector.events)
    finally:
        await _cancel_task(consumer)


async def test_set_fact_mutable_accept_delivers_response(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="MutChar2", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply("Certainly."),
            _mock_set_fact_pass(_MUT_PATH, "surgeon"),
        ]
    )

    collector = _SseCollector("fact_update_mutable")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

    resp = await client.post(_respond_url(sess.id), json={"action": "accept"})
    assert resp.status_code == 200

    await asyncio.wait_for(consumer, timeout=10.0)

    message_events = [e for e in collector.events if e.get("event") == "message"]
    assert len(message_events) == 1
    assert json.loads(message_events[0]["data"])["content"] == "Certainly."
    assert collector.events[-1]["event"] == "done"


async def test_set_fact_mutable_accept_writes_fact_to_db(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="MutChar3", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply("I am a surgeon."),
            _mock_set_fact_pass(_MUT_PATH, "surgeon"),
        ]
    )

    collector = _SseCollector("fact_update_mutable")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)
    await client.post(_respond_url(sess.id), json={"action": "accept"})
    await asyncio.wait_for(consumer, timeout=10.0)

    facts = await get_facts(db, char.id)
    assert facts["Character"]["Identity"]["Occupation"]["Value"] == "surgeon"


async def test_set_fact_mutable_reject_triggers_regeneration(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="MutChar4", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    # reject sets needs_regeneration, so the loop runs a second character + evaluator pass.
    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply("Original reply."),
            _mock_set_fact_pass(_MUT_PATH, "surgeon"),
            _mock_char_reply("Regenerated reply."),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector("fact_update_mutable")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

    resp = await client.post(_respond_url(sess.id), json={"action": "reject"})
    assert resp.status_code == 200

    await asyncio.wait_for(consumer, timeout=10.0)

    message_events = [e for e in collector.events if e.get("event") == "message"]
    assert len(message_events) == 1
    assert json.loads(message_events[0]["data"])["content"] == "Regenerated reply."


async def test_set_fact_mutable_edit_writes_user_value(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="MutChar5", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    # edit sets needs_regeneration, so the loop runs a second character + evaluator pass.
    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply("Original reply."),
            _mock_set_fact_pass(_MUT_PATH, "surgeon"),
            _mock_char_reply("I am a nurse."),
            _mock_eval_pass(),
        ]
    )

    collector = _SseCollector("fact_update_mutable")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)
    await client.post(_respond_url(sess.id), json={"action": "edit", "value": "nurse"})
    await asyncio.wait_for(consumer, timeout=10.0)

    facts = await get_facts(db, char.id)
    assert facts["Character"]["Identity"]["Occupation"]["Value"] == "nurse"


# ---------------------------------------------------------------------------
# Tests — Immutable-unset path
# ---------------------------------------------------------------------------


async def test_set_fact_immutable_unset_sidechannel_emitted(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="ImmChar1", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply(),
            _mock_set_fact_pass(_IMMU_PATH, "Elena"),
        ]
    )

    collector = _SseCollector("fact_update_immutable_unset")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

        sc_events = [
            e
            for e in collector.events
            if e.get("event") == "sidechannel"
            and json.loads(e.get("data", "{}")).get("type") == "fact_update_immutable_unset"
        ]
        assert len(sc_events) >= 1
        sc_data = json.loads(sc_events[0]["data"])
        assert sc_data["path"] == _IMMU_PATH
        assert sc_data["proposed"] == "Elena"
    finally:
        await _cancel_task(consumer)


async def test_set_fact_immutable_unset_dismiss_delivers_without_writing(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="ImmChar2", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply("I'll continue."),
            _mock_set_fact_pass(_IMMU_PATH, "Elena"),
        ]
    )

    collector = _SseCollector("fact_update_immutable_unset")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

    resp = await client.post(_respond_url(sess.id), json={"action": "dismiss"})
    assert resp.status_code == 200

    await asyncio.wait_for(consumer, timeout=10.0)

    facts = await get_facts(db, char.id)
    assert "Name" not in facts.get("Character", {}).get("Identity", {})
    done_events = [e for e in collector.events if e.get("event") == "done"]
    assert len(done_events) == 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_set_fact_respond_unknown_turn_returns_404(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, _queue = live_client
    char = await create_character(db, name="ErrChar1", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    response = await client.post(
        f"/api/sessions/{sess.id}/turns/9999/set-fact/respond",
        json={"action": "accept"},
    )
    assert response.status_code == 404


async def test_set_fact_respond_double_resolve_returns_409(
    live_client: tuple[AsyncClient, _ResponseQueue],
    db: aiosqlite.Connection,
) -> None:
    client, queue = live_client
    char = await create_character(db, name="ErrChar2", modelfile_base="qwen3:7b")
    sess = await create_session(db, character_id=char.id)

    queue.load(
        [
            _mock_world_builder(),
            _mock_char_reply(),
            _mock_set_fact_pass(_MUT_PATH, "surgeon"),
        ]
    )

    collector = _SseCollector("fact_update_mutable")
    consumer = asyncio.create_task(collector.consume(client, _messages_url(sess.id)))
    try:
        await asyncio.wait_for(collector.sidechannel_seen.wait(), timeout=10.0)

        r1 = await client.post(_respond_url(sess.id), json={"action": "accept"})
        assert r1.status_code == 200

        r2 = await client.post(_respond_url(sess.id), json={"action": "accept"})
        assert r2.status_code == 409
    finally:
        await _cancel_task(consumer)
