"""Integration tests for the implication accept/ignore endpoints."""

from __future__ import annotations

import aiosqlite
import httpx
import pytest
import respx
from httpx import AsyncClient

from memories.database import get_facts, get_inferences, get_messages
from memories.models import Character, Session
from tests.unit.conftest import make_evaluator_ndjson, make_ollama_ndjson

_OLLAMA_CHAT_URL = "http://test-ollama-integration:11434/api/chat"

_VIOLATION = {
    "type": "implication",
    "description": "Character implied having a sister",
    "suggested_fact": {"key": "siblings", "value": "one sister"},
}
_INFERENCE_VIOLATION = {
    "type": "implication",
    "description": "Character works long hours (inference)",
    "suggested_fact": None,
}


def _implication_turn(
    character_content: str = "I have a sister, actually.",
) -> list[httpx.Response]:
    """Mock a turn that produces an implication verdict."""
    return [
        httpx.Response(200, content=make_ollama_ndjson(character_content)),
        httpx.Response(
            200,
            content=make_evaluator_ndjson("implication", violations=[_VIOLATION]),
        ),
    ]


def _pass_turn(character_content: str = "I am an only child.") -> list[httpx.Response]:
    """Mock a regeneration turn that produces a pass verdict."""
    return [
        httpx.Response(200, content=make_ollama_ndjson(character_content)),
        httpx.Response(200, content=make_evaluator_ndjson()),
    ]


@pytest.fixture
async def implication_session(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> tuple[Session, int]:
    """Set up a session with one completed implication-verdict turn."""
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_implication_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Any siblings?"})
    msgs = await get_messages(db, session.id)
    assistant_msg = next(m for m in msgs if m.role == "assistant")
    return session, assistant_msg.turn_id


# ---------------------------------------------------------------------------
# Accept implication
# ---------------------------------------------------------------------------


async def test_accept_implication_creates_fact_in_db(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, turn_id = implication_session
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn())
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister"},
        )
    facts = await get_facts(db, character.id)
    assert any(f.key == "siblings" and f.value == "one sister" for f in facts)


async def test_accept_implication_returns_200_with_content(
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, turn_id = implication_session
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn("I am an only child."))
        response = await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "none"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "turn_id" in data


async def test_accept_implication_content_differs_from_original(
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, turn_id = implication_session
    regenerated = "Actually, I have no siblings."
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn(regenerated))
        response = await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "none"},
        )
    assert response.json()["content"] == regenerated


async def test_accept_implication_clears_ungrounded_implications_on_message(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, turn_id = implication_session
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn())
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "none"},
        )
    msgs = await get_messages(db, session.id)
    assistant_msg = next(m for m in msgs if m.role == "assistant" and m.turn_id == turn_id)
    assert assistant_msg.ungrounded_implications is None


async def test_edit_implication_uses_user_provided_value(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, turn_id = implication_session
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn())
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "two brothers"},  # different from suggestion
        )
    facts = await get_facts(db, character.id)
    assert any(f.key == "siblings" and f.value == "two brothers" for f in facts)


# ---------------------------------------------------------------------------
# Ignore implication
# ---------------------------------------------------------------------------


async def test_ignore_implication_returns_204(
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, turn_id = implication_session
    response = await client.post(f"/api/sessions/{session.id}/turns/{turn_id}/ignore-implication")
    assert response.status_code == 204


async def test_ignore_implication_message_ungrounded_remains_set(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, turn_id = implication_session
    await client.post(f"/api/sessions/{session.id}/turns/{turn_id}/ignore-implication")
    msgs = await get_messages(db, session.id)
    assistant_msg = next(m for m in msgs if m.role == "assistant" and m.turn_id == turn_id)
    assert assistant_msg.ungrounded_implications is not None


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


async def test_accept_implication_unknown_session_returns_404(
    client: AsyncClient,
) -> None:
    response = await client.post(
        "/api/sessions/9999/turns/1/accept-implication",
        json={"key": "x", "value": "y"},
    )
    assert response.status_code == 404


async def test_accept_implication_unknown_turn_returns_404(
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    session, _ = implication_session
    response = await client.post(
        f"/api/sessions/{session.id}/turns/9999/accept-implication",
        json={"key": "x", "value": "y"},
    )
    assert response.status_code == 404


async def test_accept_implication_on_clean_turn_returns_422(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """A turn with no ungrounded implications cannot be accepted."""
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_ollama_ndjson("Clean response.")),
                httpx.Response(200, content=make_evaluator_ndjson()),
            ]
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    msgs = await get_messages(db, session.id)
    turn_id = next(m for m in msgs if m.role == "assistant").turn_id
    response = await client.post(
        f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
        json={"key": "x", "value": "y"},
    )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Accept/ignore inference
# ---------------------------------------------------------------------------


@pytest.fixture
async def probabilistic_session(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> tuple[Session, int]:
    """Set up a session with one completed new_inference_probabilistic verdict turn."""
    inferences = [
        {
            "inference_type": "probabilistic",
            "statement": "Alice works long hours",
            "derivation": "occupation=surgeon",
            "source_fact_ids": [1],
            "source_inference_ids": [],
        }
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_ollama_ndjson("I work very long hours.")),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson(
                        "new_inference_probabilistic", new_inferences=inferences
                    ),
                ),
            ]
        )
        await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Your schedule?"}
        )
    msgs = await get_messages(db, session.id)
    turn_id = next(m for m in msgs if m.role == "assistant").turn_id
    return session, turn_id


async def test_accept_inference_creates_inference_in_db(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    probabilistic_session: tuple[Session, int],
) -> None:
    session, turn_id = probabilistic_session
    response = await client.post(
        f"/api/sessions/{session.id}/turns/{turn_id}/accept-inference",
        json={
            "statement": "Alice works long hours",
            "derivation": "occupation=surgeon",
            "source_fact_ids": [1],
            "inference_type": "probabilistic",
        },
    )
    assert response.status_code == 201
    inferences = await get_inferences(db, character.id)
    assert any(i.statement == "Alice works long hours" for i in inferences)


async def test_accept_inference_returns_201_with_inference(
    client: AsyncClient,
    character: Character,
    probabilistic_session: tuple[Session, int],
) -> None:
    session, turn_id = probabilistic_session
    response = await client.post(
        f"/api/sessions/{session.id}/turns/{turn_id}/accept-inference",
        json={
            "statement": "Alice works long hours",
            "derivation": "occupation=surgeon",
            "source_fact_ids": [],
            "inference_type": "probabilistic",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert "statement" in data


async def test_ignore_inference_returns_204(
    client: AsyncClient,
    character: Character,
    probabilistic_session: tuple[Session, int],
) -> None:
    session, turn_id = probabilistic_session
    response = await client.post(f"/api/sessions/{session.id}/turns/{turn_id}/ignore-inference")
    assert response.status_code == 204


async def test_ignore_inference_does_not_create_inference_row(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    probabilistic_session: tuple[Session, int],
) -> None:
    session, turn_id = probabilistic_session
    await client.post(f"/api/sessions/{session.id}/turns/{turn_id}/ignore-inference")
    inferences = await get_inferences(db, character.id)
    assert inferences == []
