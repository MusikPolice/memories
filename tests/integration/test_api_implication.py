"""Integration tests for the implication accept/ignore endpoints."""

from __future__ import annotations

import json

import aiosqlite
import httpx
import pytest
import respx
from httpx import AsyncClient

from memories.database import (
    create_inference,
    get_decisions,
    get_facts,
    get_inferences,
    get_messages,
)
from memories.models import Character, Session
from tests.unit.conftest import make_evaluator_ndjson, make_extractor_ndjson, make_ollama_ndjson

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
        httpx.Response(200, content=make_extractor_ndjson()),
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


async def test_accept_implication_stores_new_decision(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """After accepting, a new decision row is stored with the regenerated verdict."""
    session, turn_id = implication_session
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn("I am an only child."))
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "none"},
        )
    decisions = await get_decisions(db, session.id)
    # Two decisions: one from the original turn (implication), one from the regen (pass)
    assert len(decisions) == 2
    regen_decision = next(d for d in decisions if d.reasoning != "Clean.")
    assert regen_decision.verdict == "pass"
    assert regen_decision.turn_id == turn_id


async def test_accept_implication_duplicate_key_updates_existing_fact(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """Accepting an implication for an existing key updates the fact rather than failing."""
    session, turn_id = implication_session
    # Accept once to create the fact
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn())
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister"},
        )

    # Trigger a second implication turn where the evaluator proposes a CHANGED
    # value for the same key.  The filter only strips violations whose suggested
    # value already matches an existing fact; a different value must still surface.
    changed_violation = {
        "type": "implication",
        "description": "Character now mentions two brothers instead",
        "suggested_fact": {"key": "siblings", "value": "two brothers"},
    }
    second_turn_side_effect = [
        httpx.Response(200, content=make_extractor_ndjson()),
        httpx.Response(200, content=make_ollama_ndjson("Actually I have two brothers.")),
        httpx.Response(
            200, content=make_evaluator_ndjson("implication", violations=[changed_violation])
        ),
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=second_turn_side_effect)
        await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Tell me about your sister."}
        )
    msgs = await get_messages(db, session.id)
    second_assistant = next(
        m for m in reversed(msgs) if m.role == "assistant" and m.ungrounded_implications is not None
    )
    second_turn_id = second_assistant.turn_id

    # Accept with a different value — should update, not fail with 500
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/turns/{second_turn_id}/accept-implication",
            json={"key": "siblings", "value": "two brothers"},
        )
    assert response.status_code == 200
    facts = await get_facts(db, character.id)
    siblings = next(f for f in facts if f.key == "siblings")
    assert siblings.value == "two brothers"


async def test_accept_implication_regen_ungrounded_returns_ungrounded_flag(
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """If the regenerated response is itself ungrounded, the endpoint returns ungrounded=True."""
    session, turn_id = implication_session
    second_violation = {
        "type": "implication",
        "description": "implied a favourite colour",
        "suggested_fact": {"key": "favourite_colour", "value": "blue"},
    }
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_ollama_ndjson("I like blue things.")),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson("implication", violations=[second_violation]),
                ),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "none"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data.get("ungrounded") is True
    assert "violations" in data


async def test_accept_implication_regeneration_includes_inferences(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """Inferences must appear in the regenerated system prompt after accepting an implication."""
    session, turn_id = implication_session
    await create_inference(
        db,
        character_id=character.id,
        statement="Alice was born in 1991",
        derivation="age=33, year=2024",
    )
    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn())
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister"},
        )
    body = json.loads(route.calls[0].request.content)
    system_content = body["messages"][0]["content"]
    assert "Alice was born in 1991" in system_content


# ---------------------------------------------------------------------------
# regenerate=False — accept without LLM regeneration
# ---------------------------------------------------------------------------


async def test_accept_implication_no_regen_creates_fact_without_llm_call(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """With regenerate=False the fact is saved but no Ollama call is made."""
    session, turn_id = implication_session
    with respx.mock:
        # No mocked Ollama routes; any call would raise an error
        response = await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister", "regenerate": False},
        )
    assert response.status_code == 200
    facts = await get_facts(db, character.id)
    assert any(f.key == "siblings" and f.value == "one sister" for f in facts)


async def test_accept_implication_no_regen_returns_original_content(
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """With regenerate=False the response contains the original assistant message."""
    session, turn_id = implication_session
    with respx.mock:
        response = await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister", "regenerate": False},
        )
    data = response.json()
    assert data["content"] == "I have a sister, actually."
    assert data["turn_id"] == turn_id


async def test_accept_implication_no_regen_clears_ungrounded_flag(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """With regenerate=False the ungrounded_implications flag is still cleared."""
    session, turn_id = implication_session
    with respx.mock:
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister", "regenerate": False},
        )
    msgs = await get_messages(db, session.id)
    assistant_msg = next(m for m in msgs if m.role == "assistant" and m.turn_id == turn_id)
    assert assistant_msg.ungrounded_implications is None


async def test_accept_implication_no_regen_does_not_store_extra_decision(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    implication_session: tuple[Session, int],
) -> None:
    """With regenerate=False no second decision row is written."""
    session, turn_id = implication_session
    with respx.mock:
        await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister", "regenerate": False},
        )
    decisions = await get_decisions(db, session.id)
    # Only the original implication decision — no regeneration decision
    assert len(decisions) == 1


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


async def test_accept_second_implication_on_same_turn_succeeds(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Regression: accepting the second of two violations on the same turn must not return 422.

    The first accept clears ungrounded_implications on the stored message; the second
    accept was previously blocked by a guard that checked that field.
    """
    v1 = {
        "type": "implication",
        "description": "implied a sister",
        "suggested_fact": {"key": "siblings", "value": "one sister"},
    }
    v2 = {
        "type": "implication",
        "description": "implied eye colour",
        "suggested_fact": {"key": "eye_colour", "value": "brown"},
    }

    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson()),
                httpx.Response(200, content=make_ollama_ndjson("I have brown eyes and a sister.")),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson("implication", violations=[v1, v2]),
                ),
            ]
        )
        await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Describe yourself."}
        )

    msgs = await get_messages(db, session.id)
    turn_id = next(m for m in msgs if m.role == "assistant").turn_id

    # First accept — succeeds and clears ungrounded_implications on the stored message
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn("I have a sister."))
        r1 = await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "siblings", "value": "one sister"},
        )
    assert r1.status_code == 200

    # Second accept on the same turn — previously returned 422
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_pass_turn("I have brown eyes."))
        r2 = await client.post(
            f"/api/sessions/{session.id}/turns/{turn_id}/accept-implication",
            json={"key": "eye_colour", "value": "brown"},
        )
    assert r2.status_code == 200
    facts = await get_facts(db, character.id)
    assert any(f.key == "eye_colour" and f.value == "brown" for f in facts)


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
                httpx.Response(200, content=make_extractor_ndjson()),
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


async def test_accept_inference_depth_computed_from_source_inference_ids(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    probabilistic_session: tuple[Session, int],
) -> None:
    """Depth is computed from source_inference_ids; depth-1 source → stored at depth 2."""
    session, turn_id = probabilistic_session
    base = await create_inference(
        db,
        character_id=character.id,
        statement="Base inference at depth 1",
        derivation="base",
        depth=1,
    )
    response = await client.post(
        f"/api/sessions/{session.id}/turns/{turn_id}/accept-inference",
        json={
            "statement": "Derived inference",
            "derivation": "from base",
            "source_fact_ids": [],
            "source_inference_ids": [base.id],
            "inference_type": "probabilistic",
        },
    )
    assert response.status_code == 201
    assert response.json()["depth"] == 2


async def test_accept_inference_exceeding_depth_cap_returns_422(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    probabilistic_session: tuple[Session, int],
) -> None:
    """Accepting an inference that would exceed MAX_INFERENCE_DEPTH returns 422."""
    from memories.services.inference_service import MAX_INFERENCE_DEPTH

    session, turn_id = probabilistic_session
    at_max = await create_inference(
        db,
        character_id=character.id,
        statement="At max depth",
        derivation="base",
        depth=MAX_INFERENCE_DEPTH,
    )
    response = await client.post(
        f"/api/sessions/{session.id}/turns/{turn_id}/accept-inference",
        json={
            "statement": "Would exceed depth cap",
            "derivation": "from at_max",
            "source_fact_ids": [],
            "source_inference_ids": [at_max.id],
            "inference_type": "probabilistic",
        },
    )
    assert response.status_code == 422


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
