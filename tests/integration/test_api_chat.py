"""Integration tests for the chat SSE endpoint."""

from __future__ import annotations

import json

import aiosqlite
import httpx
import respx
from httpx import AsyncClient

from memories.database import create_fact, get_messages
from memories.models import Character, Session
from tests.unit.conftest import make_ollama_ndjson

_OLLAMA_CHAT_URL = "http://test-ollama-integration:11434/api/chat"


def _mock_ok(content: str = "I am fine.") -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(content))


def _parse_sse(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event: dict[str, str] = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = line[6:]
        if event:
            events.append(event)
    return events


async def test_send_message_content_type_is_event_stream(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    assert "text/event-stream" in response.headers["content-type"]


async def test_send_message_emits_status_event_first(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    assert events[0]["event"] == "status"
    assert json.loads(events[0]["data"])["state"] == "generating"


async def test_send_message_emits_message_event(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok("Hello from character."))
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    message_events = [e for e in events if e.get("event") == "message"]
    assert len(message_events) == 1
    data = json.loads(message_events[0]["data"])
    assert data["role"] == "assistant"
    assert data["content"] == "Hello from character."


async def test_send_message_emits_done_event_last(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    assert events[-1]["event"] == "done"


async def test_send_message_stores_user_message(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "User input"})
    messages = await get_messages(db, session.id)
    assert any(m.role == "user" and m.content == "User input" for m in messages)


async def test_send_message_stores_assistant_response(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok("Character says hello."))
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    messages = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "Character says hello." for m in messages)


async def test_ollama_receives_system_message_with_facts(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    await create_fact(db, character_id=character.id, key="birthplace", value="Reykjavik")
    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok())
        await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Where am I from?"}
        )
    body = json.loads(route.calls[0].request.content)
    system_content: str = body["messages"][0]["content"]
    assert "birthplace" in system_content
    assert "Reykjavik" in system_content


async def test_ollama_receives_prior_history(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok("First reply."))
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "First message"})

    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_ok("Second reply."))
        await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Second message"}
        )

    body = json.loads(route.calls[0].request.content)
    roles = [m["role"] for m in body["messages"]]
    assert "user" in roles
    assert "assistant" in roles


async def test_send_to_unknown_session_404(client: AsyncClient) -> None:
    response = await client.post("/api/sessions/9999/messages", json={"content": "Hello"})
    assert response.status_code == 404


async def test_send_to_ended_session_409(
    client: AsyncClient, character: Character, session: Session
) -> None:
    await client.post(f"/api/sessions/{session.id}/end")
    response = await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    assert response.status_code == 409
