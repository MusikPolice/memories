"""Unit tests for memories.services.chat_service.run_turn.

These tests use a real in-memory SQLite database (via the shared ``db``
fixture) and a mocked Ollama HTTP layer (via respx).  They test the
orchestration contract of run_turn — what it reads, what it writes, and in
what order — not the HTTP or SQL layers themselves.
"""

from __future__ import annotations

import json

import aiosqlite
import httpx
import pytest
import respx

from memories.database import create_fact, get_messages
from memories.exceptions import NotFoundError
from memories.models import Character, Session
from memories.services.chat_service import run_turn
from memories.services.ollama_client import OllamaClient, OllamaConnectionError
from tests.unit.conftest import OLLAMA_BASE_URL, make_ollama_ndjson

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def _mock_ok(content: str = "I am fine, thank you.") -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(content))


# ---------------------------------------------------------------------------
# Tests: what run_turn sends to Ollama
# ---------------------------------------------------------------------------


async def test_system_message_is_first_in_ollama_request(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(return_value=_mock_ok())
        await run_turn(db, session.id, "Hello", ollama)

    body = json.loads(route.calls[0].request.content)
    assert body["messages"][0]["role"] == "system"


async def test_history_included_in_ollama_request(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """After the first turn the second Ollama request must include prior history."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_mock_ok("First response."))
        await run_turn(db, session.id, "Turn one", ollama)

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(return_value=_mock_ok("Second response."))
        await run_turn(db, session.id, "Turn two", ollama)

    body = json.loads(route.calls[0].request.content)
    roles = [m["role"] for m in body["messages"]]
    assert "user" in roles
    assert "assistant" in roles


async def test_history_ordered_by_turn_id(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_mock_ok("Reply one."))
        await run_turn(db, session.id, "First message", ollama)

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(return_value=_mock_ok("Reply two."))
        await run_turn(db, session.id, "Second message", ollama)

    body = json.loads(route.calls[0].request.content)
    # Skip the system message; check remaining messages are user/assistant/user…
    conversation = [m for m in body["messages"] if m["role"] != "system"]
    assert conversation[0]["content"] == "First message"
    assert conversation[1]["content"] == "Reply one."
    assert conversation[2]["content"] == "Second message"


async def test_new_user_message_appended_last(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(return_value=_mock_ok())
        await run_turn(db, session.id, "My specific question", ollama)

    body = json.loads(route.calls[0].request.content)
    assert body["messages"][-1]["role"] == "user"
    assert body["messages"][-1]["content"] == "My specific question"


async def test_facts_reflected_in_system_message(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await create_fact(db, character_id=character.id, key="birthplace", value="Reykjavik")

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(return_value=_mock_ok())
        await run_turn(db, session.id, "Where are you from?", ollama)

    body = json.loads(route.calls[0].request.content)
    system_content: str = body["messages"][0]["content"]
    assert "birthplace" in system_content
    assert "Reykjavik" in system_content


# ---------------------------------------------------------------------------
# Tests: what run_turn writes to the DB
# ---------------------------------------------------------------------------


async def test_user_message_stored_before_llm_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """The user message must be persisted even if the Ollama call subsequently fails."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(OllamaConnectionError):
            await run_turn(db, session.id, "Hi there", ollama)

    messages = await get_messages(db, session.id)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hi there"


async def test_assistant_message_stored_after_llm_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_mock_ok("Stored assistant reply."))
        await run_turn(db, session.id, "Hello", ollama)

    messages = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "Stored assistant reply." for m in messages)


async def test_turn_ids_increment(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_mock_ok("First."))
        await run_turn(db, session.id, "Message one", ollama)

    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_mock_ok("Second."))
        await run_turn(db, session.id, "Message two", ollama)

    messages = await get_messages(db, session.id)
    turn_ids = [m.turn_id for m in messages]
    # Two turns → turn_ids 1, 1, 2, 2 (user+assistant per turn)
    assert sorted(set(turn_ids)) == [1, 2]


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


async def test_run_turn_raises_on_unknown_session(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    with pytest.raises(NotFoundError):
        await run_turn(db, session_id=9999, user_content="Hello", ollama=ollama)


async def test_run_turn_raises_on_ended_session(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import end_session

    await end_session(db, session.id)
    with pytest.raises(NotFoundError):
        await run_turn(db, session.id, "Hello", ollama)
