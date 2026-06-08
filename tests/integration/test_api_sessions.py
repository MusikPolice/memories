"""Integration tests for the sessions API."""

from __future__ import annotations

import json

import aiosqlite
import httpx
import respx
from httpx import AsyncClient

from memories.models import Character, Session
from memories.services.experience_service import add_active_experiences, get_active_experiences
from tests.unit.conftest import make_ollama_ndjson

_CHAT_URL = "http://test-ollama-integration:11434/api/chat"


def _make_session_end_chat(
    closing_journal: str = "A quiet and reflective session.",
    proposed_experiences: list[dict] | None = None,
) -> httpx.Response:
    """Build a mock Ollama chat response that returns valid session-end JSON."""
    data = {
        "closing_journal": closing_journal,
        "proposed_experiences": proposed_experiences or [],
    }
    return httpx.Response(200, content=make_ollama_ndjson(json.dumps(data)))


async def _create_session(client: AsyncClient, char_id: int) -> int:
    """Helper: POST /api/sessions/ and return the new session id."""
    r = await client.post("/api/sessions/", json={"character_id": char_id})
    return r.json()["session"]["id"]


async def _create_char(client: AsyncClient, name: str = "Alice") -> int:
    r = await client.post("/api/characters/", json={"name": name, "modelfile_base": "qwen3:7b"})
    return r.json()["id"]


async def test_start_session_201(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    response = await client.post("/api/sessions/", json={"character_id": char_id})
    assert response.status_code == 201
    data = response.json()
    assert "session" in data
    assert "id" in data["session"]
    assert "previous_journal" in data


async def test_start_session_no_previous_journal_is_null(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    response = await client.post("/api/sessions/", json={"character_id": char_id})
    assert response.json()["previous_journal"] is None


async def test_start_session_returns_previous_journal_when_present(
    client: AsyncClient,
) -> None:
    char_id = await _create_char(client)
    first_session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat("Great first session!"))
        await client.post(f"/api/sessions/{first_session_id}/end")
    response = await client.post("/api/sessions/", json={"character_id": char_id})
    assert response.json()["previous_journal"] == "Great first session!"


async def test_start_session_skips_empty_sessions_for_previous_journal(
    client: AsyncClient,
) -> None:
    char_id = await _create_char(client)
    first_session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat("Anchored journal."))
        await client.post(f"/api/sessions/{first_session_id}/end")
    # Second session ends without a journal (e.g. evaluator failure)
    second_session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(200, content=make_ollama_ndjson("not json"))
        )
        await client.post(f"/api/sessions/{second_session_id}/end")
    # Third session should get the first session's journal (second has none)
    response = await client.post("/api/sessions/", json={"character_id": char_id})
    assert response.json()["previous_journal"] == "Anchored journal."


async def test_start_session_unknown_character_404(client: AsyncClient) -> None:
    response = await client.post("/api/sessions/", json={"character_id": 9999})
    assert response.status_code == 404


async def test_end_session_returns_200(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat())
        response = await client.post(f"/api/sessions/{session_id}/end")
    assert response.status_code == 200
    data = response.json()
    assert "session" in data
    assert "closing_journal" in data
    assert "proposed_experiences" in data
    assert data["session"]["ended_at"] is not None


async def test_end_unknown_session_404(client: AsyncClient) -> None:
    response = await client.post("/api/sessions/9999/end")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Phase 5 additions — session-end evaluator
# ---------------------------------------------------------------------------


async def test_end_session_response_has_session(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat())
        response = await client.post(f"/api/sessions/{session_id}/end")
    data = response.json()
    assert "session" in data
    assert data["session"]["ended_at"] is not None


async def test_end_session_response_has_closing_journal(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat())
        response = await client.post(f"/api/sessions/{session_id}/end")
    assert "closing_journal" in response.json()


async def test_end_session_response_has_proposed_experiences(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat())
        response = await client.post(f"/api/sessions/{session_id}/end")
    data = response.json()
    assert "proposed_experiences" in data
    assert isinstance(data["proposed_experiences"], list)


async def test_end_session_stores_closing_journal_in_db(
    db: aiosqlite.Connection, client: AsyncClient
) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat("Great session today!"))
        await client.post(f"/api/sessions/{session_id}/end")
    row = await (
        await db.execute("SELECT closing_journal FROM sessions WHERE id = ?", (session_id,))
    ).fetchone()
    assert row is not None
    assert row[0] == "Great session today!"


async def test_end_session_already_ended_returns_409(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat())
        await client.post(f"/api/sessions/{session_id}/end")
    response = await client.post(f"/api/sessions/{session_id}/end")
    assert response.status_code == 409


async def test_end_session_unknown_session_returns_404(client: AsyncClient) -> None:
    response = await client.post("/api/sessions/99999/end")
    assert response.status_code == 404


async def test_end_session_no_messages_returns_empty_proposals(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            return_value=_make_session_end_chat(
                closing_journal="Empty session.", proposed_experiences=[]
            )
        )
        response = await client.post(f"/api/sessions/{session_id}/end")
    assert response.json()["proposed_experiences"] == []


async def test_end_session_evaluator_parse_failure_returns_empty_proposals(
    client: AsyncClient,
) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(200, content=make_ollama_ndjson("this is not json"))
        )
        response = await client.post(f"/api/sessions/{session_id}/end")
    assert response.status_code == 200
    assert response.json()["proposed_experiences"] == []
    assert response.json()["closing_journal"] == ""


async def test_end_session_evaluator_parse_failure_still_ends_session(
    client: AsyncClient,
) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(200, content=make_ollama_ndjson("garbage"))
        )
        response = await client.post(f"/api/sessions/{session_id}/end")
    assert response.json()["session"]["ended_at"] is not None


async def test_end_session_calls_session_end_evaluator_ollama(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat())
        await client.post(f"/api/sessions/{session_id}/end")
    assert route.called


async def test_end_session_proposed_experience_has_statement(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    proposals = [
        {"statement": "User lives in Chicago", "source": "told_by_user", "turn_reference": 1}
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat("Good session.", proposals))
        response = await client.post(f"/api/sessions/{session_id}/end")
    proposals_data = response.json()["proposed_experiences"]
    assert len(proposals_data) == 1
    assert proposals_data[0]["statement"] == "User lives in Chicago"


async def test_end_session_proposed_experience_has_source(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    proposals = [
        {"statement": "Chicago statement", "source": "told_by_user", "turn_reference": 1},
        {"statement": "Observed behavior", "source": "observed", "turn_reference": 2},
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat("Good session.", proposals))
        response = await client.post(f"/api/sessions/{session_id}/end")
    for proposal in response.json()["proposed_experiences"]:
        assert proposal["source"] in ("told_by_user", "observed")


async def test_end_session_clears_active_experiences_for_session(
    client: AsyncClient, character: Character, session: Session
) -> None:
    add_active_experiences(session.id, [])  # seed the dict entry
    with respx.mock:
        respx.post(_CHAT_URL).mock(return_value=_make_session_end_chat())
        await client.post(f"/api/sessions/{session.id}/end")
    assert get_active_experiences(session.id) == []


# ---------------------------------------------------------------------------


async def test_get_session_messages_initially_empty(client: AsyncClient) -> None:
    char_id = await _create_char(client)
    session_id = await _create_session(client, char_id)
    response = await client.get(f"/api/sessions/{session_id}/messages")
    assert response.status_code == 200
    assert response.json() == []
