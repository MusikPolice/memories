"""Integration tests for Phase 6 extraction resolution endpoints.

Three endpoints added to implication.py:
  POST .../turns/{turn_id}/undo-user-fact
  POST .../turns/{turn_id}/accept-implicit-fact
  POST .../turns/{turn_id}/ignore-implicit-fact

Tests 47-65.  All endpoints return 404 before Phase 6 is implemented, so
tests that assert 200/201 will fail.  Tests that assert 404/409 are written
with specific error-message checks so they also fail against the generic
routing-level 404.
"""

from __future__ import annotations

import aiosqlite
import pytest
import respx
from httpx import AsyncClient

from memories.database import (
    create_fact,
    create_inference,
    get_facts,
    update_fact,
)
from memories.models import Character, Session

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def fact(db: aiosqlite.Connection, character: Character):
    return await create_fact(
        db,
        character_id=character.id,
        key="home_city",
        value="Reykjavik",
        category="user",
        mutability="low",
    )


@pytest.fixture
async def fact_with_inference(db: aiosqlite.Connection, character: Character):
    """A fact with a downstream inference — used for cascade tests."""
    f = await create_fact(
        db,
        character_id=character.id,
        key="home_city",
        value="Reykjavik",
        category="user",
        mutability="low",
    )
    inf = await create_inference(
        db,
        character_id=character.id,
        statement="User is probably familiar with Icelandic culture",
        derivation=f"home_city=Reykjavik [fact:{f.id}]",
        source_fact_ids=[f.id],
    )
    return f, inf


# ---------------------------------------------------------------------------
# undo-user-fact
# ---------------------------------------------------------------------------


async def test_undo_user_fact_returns_200(
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """POST .../undo-user-fact with valid fact_id and restore_value → 200."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/undo-user-fact",
        json={"fact_id": fact.id, "restore_value": "Oslo"},
    )
    assert response.status_code == 200


async def test_undo_user_fact_restores_value_in_db(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """After undo, GET /facts shows restore_value."""
    # First update the fact to simulate a Tier 2 auto-update
    await update_fact(db, fact.id, value="Chicago")

    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/undo-user-fact",
        json={"fact_id": fact.id, "restore_value": "Reykjavik"},
    )
    assert response.status_code == 200

    facts = await get_facts(db, character.id)
    restored = next(f for f in facts if f.id == fact.id)
    assert restored.value == "Reykjavik"


async def test_undo_user_fact_preserves_category_and_mutability(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """Category and mutability unchanged after undo."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/undo-user-fact",
        json={"fact_id": fact.id, "restore_value": "Oslo"},
    )
    assert response.status_code == 200

    facts = await get_facts(db, character.id)
    restored = next(f for f in facts if f.id == fact.id)
    assert restored.category == "user"
    assert restored.mutability == "low"


async def test_undo_user_fact_response_includes_fact_and_stale_inferences(
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """Response has fact key and stale_inferences list."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/undo-user-fact",
        json={"fact_id": fact.id, "restore_value": "Oslo"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "fact" in body
    assert "stale_inferences" in body
    assert isinstance(body["stale_inferences"], list)


async def test_undo_user_fact_triggers_cascade(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
    fact_with_inference,
) -> None:
    """Fact with downstream inferences → stale_inferences non-empty in response."""
    fact, inference = fact_with_inference
    # Change the fact value first so the cascade has something to find
    await update_fact(db, fact.id, value="Chicago")

    with respx.mock:
        # Cascade requires an LLM call per inference; stub with a "stale" verdict
        import httpx

        from tests.unit.conftest import make_ollama_ndjson

        stale_eval = httpx.Response(
            200,
            content=make_ollama_ndjson('{"verdict":"stale","reasoning":"location changed"}'),
        )
        respx.post("http://test-ollama-integration:11434/api/chat").mock(return_value=stale_eval)
        response = await client.post(
            f"/api/sessions/{session.id}/turns/1/undo-user-fact",
            json={"fact_id": fact.id, "restore_value": "Reykjavik"},
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body["stale_inferences"]) > 0


async def test_undo_user_fact_unknown_session_returns_404(
    client: AsyncClient,
    character: Character,
    fact,
) -> None:
    """Non-existent session_id → 404 with session-specific message."""
    response = await client.post(
        "/api/sessions/99999/turns/1/undo-user-fact",
        json={"fact_id": fact.id, "restore_value": "Oslo"},
    )
    assert response.status_code == 404
    body = response.json()
    # Real implementation returns "Session 99999 not found" not generic "Not Found"
    assert "session" in body.get("detail", "").lower()


async def test_undo_user_fact_unknown_fact_returns_404(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Non-existent fact_id → 404 with fact-specific message."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/undo-user-fact",
        json={"fact_id": 99999, "restore_value": "Oslo"},
    )
    assert response.status_code == 404
    body = response.json()
    assert "fact" in body.get("detail", "").lower()


async def test_undo_user_fact_wrong_character_returns_404(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Fact belongs to different character → 404."""
    from memories.database import create_character
    from memories.database import create_fact as _cf

    other = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    other_fact = await _cf(db, character_id=other.id, key="other_key", value="other_val")

    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/undo-user-fact",
        json={"fact_id": other_fact.id, "restore_value": "Oslo"},
    )
    assert response.status_code == 404
    body = response.json()
    assert "fact" in body.get("detail", "").lower()


async def test_undo_user_fact_ended_session_returns_409(
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """Ended session → 409."""
    await client.post(f"/api/sessions/{session.id}/end")
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/undo-user-fact",
        json={"fact_id": fact.id, "restore_value": "Oslo"},
    )
    assert response.status_code == 409


# ---------------------------------------------------------------------------
# accept-implicit-fact
# ---------------------------------------------------------------------------


async def test_accept_implicit_fact_tier3_creates_new_fact(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """POST .../accept-implicit-fact without existing_fact_id → 201; GET /facts includes it."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
        json={
            "key": "current_mood",
            "value": "anxious",
            "category": "user",
            "mutability": "high",
        },
    )
    assert response.status_code == 201

    facts = await get_facts(db, character.id)
    assert any(f.key == "current_mood" and f.value == "anxious" for f in facts)


async def test_accept_implicit_fact_tier3_response_includes_fact(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Response body has fact key with created Fact."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
        json={
            "key": "current_mood",
            "value": "anxious",
            "category": "user",
            "mutability": "high",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert "fact" in body
    assert body["fact"]["key"] == "current_mood"


async def test_accept_implicit_fact_tier3_stale_inferences_is_empty(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Tier 3 acceptance (new fact) → stale_inferences is empty."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
        json={
            "key": "current_mood",
            "value": "anxious",
            "category": "user",
            "mutability": "high",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body.get("stale_inferences", []) == []


async def test_accept_implicit_fact_tier4_updates_existing_fact(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """POST .../accept-implicit-fact with existing_fact_id → GET /facts shows new value."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
        json={
            "key": "home_city",
            "value": "Chicago",
            "category": "user",
            "mutability": "low",
            "existing_fact_id": fact.id,
        },
    )
    assert response.status_code == 200

    facts = await get_facts(db, character.id)
    updated = next(f for f in facts if f.id == fact.id)
    assert updated.value == "Chicago"


async def test_accept_implicit_fact_tier4_triggers_cascade(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
    fact_with_inference,
) -> None:
    """Tier 4 acceptance on fact with downstream inferences → stale_inferences non-empty."""
    fact, inference = fact_with_inference

    with respx.mock:
        import httpx

        from tests.unit.conftest import make_ollama_ndjson

        stale_eval = httpx.Response(
            200,
            content=make_ollama_ndjson('{"verdict":"stale","reasoning":"city changed"}'),
        )
        respx.post("http://test-ollama-integration:11434/api/chat").mock(return_value=stale_eval)
        response = await client.post(
            f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
            json={
                "key": "home_city",
                "value": "Chicago",
                "category": "user",
                "mutability": "low",
                "existing_fact_id": fact.id,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert len(body.get("stale_inferences", [])) > 0


async def test_accept_implicit_fact_tier4_wrong_character_returns_404(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """existing_fact_id belongs to different character → 404."""
    from memories.database import create_character
    from memories.database import create_fact as _cf

    other = await create_character(db, name="Carol", modelfile_base="qwen3:7b")
    other_fact = await _cf(db, character_id=other.id, key="other_key", value="other_val")

    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
        json={
            "key": "other_key",
            "value": "new_val",
            "category": "user",
            "mutability": "low",
            "existing_fact_id": other_fact.id,
        },
    )
    assert response.status_code == 404


async def test_accept_implicit_fact_ended_session_returns_409(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Ended session → 409."""
    await client.post(f"/api/sessions/{session.id}/end")
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
        json={
            "key": "current_mood",
            "value": "tired",
            "category": "user",
            "mutability": "high",
        },
    )
    assert response.status_code == 409


# ---------------------------------------------------------------------------
# ignore-implicit-fact
# ---------------------------------------------------------------------------


async def test_ignore_implicit_fact_returns_204(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """POST .../ignore-implicit-fact → 204."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/ignore-implicit-fact",
        json={"key": "current_mood"},
    )
    assert response.status_code == 204


async def test_ignore_implicit_fact_does_not_modify_db(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """After ignore, GET /facts unchanged."""
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/ignore-implicit-fact",
        json={"key": "home_city"},
    )
    assert response.status_code == 204

    facts = await get_facts(db, character.id)
    home = next(f for f in facts if f.id == fact.id)
    assert home.value == "Reykjavik"


async def test_accept_implicit_fact_tier3_duplicate_key_updates_existing(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
    fact,
) -> None:
    """Tier 3 accept with a key that already exists → 200 with updated value (not 500)."""
    # The 'fact' fixture creates home_city=Reykjavik with category='user'
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/accept-implicit-fact",
        json={
            "key": "home_city",
            "value": "Chicago",
            "category": "user",
            "mutability": "low",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["fact"]["value"] == "Chicago"

    facts = await get_facts(db, character.id)
    home = next(f for f in facts if f.key == "home_city")
    assert home.value == "Chicago"


async def test_ignore_implicit_fact_ended_session_returns_409(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Ended session → 409."""
    await client.post(f"/api/sessions/{session.id}/end")
    response = await client.post(
        f"/api/sessions/{session.id}/turns/1/ignore-implicit-fact",
        json={"key": "current_mood"},
    )
    assert response.status_code == 409
