"""Integration tests for the decisions API endpoint."""

from __future__ import annotations

import aiosqlite
import httpx
import respx
from httpx import AsyncClient

from memories.models import Character, Session
from tests.unit.conftest import make_evaluator_ndjson, make_ollama_ndjson

_OLLAMA_CHAT_URL = "http://test-ollama-integration:11434/api/chat"


def _mock_turn(character_content: str = "I am fine.") -> list[httpx.Response]:
    return [
        httpx.Response(200, content=make_ollama_ndjson(character_content)),
        httpx.Response(200, content=make_evaluator_ndjson()),
    ]


async def test_get_decisions_initially_empty(
    client: AsyncClient, character: Character, session: Session
) -> None:
    response = await client.get(f"/api/sessions/{session.id}/decisions")
    assert response.status_code == 200
    assert response.json() == []


async def test_get_decisions_after_one_turn(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    response = await client.get(f"/api/sessions/{session.id}/decisions")
    assert response.status_code == 200
    assert len(response.json()) == 1


async def test_get_decisions_contains_verdict_field(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    response = await client.get(f"/api/sessions/{session.id}/decisions")
    data = response.json()
    assert "verdict" in data[0]


async def test_get_decisions_contains_reasoning_field(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    response = await client.get(f"/api/sessions/{session.id}/decisions")
    data = response.json()
    assert "reasoning" in data[0]
    assert data[0]["reasoning"]


async def test_get_decisions_ordered_by_turn_id_desc(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("First reply."))
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Turn one"})
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("Second reply."))
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Turn two"})
    response = await client.get(f"/api/sessions/{session.id}/decisions")
    data = response.json()
    assert len(data) == 2
    assert data[0]["turn_id"] == 2
    assert data[1]["turn_id"] == 1


async def test_get_decisions_unknown_session_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/sessions/9999/decisions")
    assert response.status_code == 404


async def test_get_decisions_includes_violations_for_implication(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    violations = [
        {"type": "implication", "description": "implied a sibling", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_ollama_ndjson("I have a sister.")),
                httpx.Response(
                    200, content=make_evaluator_ndjson("implication", violations=violations)
                ),
            ]
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Family?"})
    response = await client.get(f"/api/sessions/{session.id}/decisions")
    data = response.json()
    assert data[0]["violations"] is not None
