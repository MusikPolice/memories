"""Integration tests for the inference generation and management API endpoints."""

from __future__ import annotations

import json

import aiosqlite
import httpx
import respx
from httpx import AsyncClient

from memories.database import create_inference, get_inferences
from memories.models import Character, Fact
from tests.unit.conftest import make_ollama_ndjson

_OLLAMA_CHAT_URL = "http://test-ollama-integration:11434/api/chat"


def _mock_eager_pass_response(items: list[dict]) -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(json.dumps(items)))


def _mock_revalidation_response(holds: bool) -> httpx.Response:
    return httpx.Response(
        200, content=make_ollama_ndjson(json.dumps({"holds": holds, "reason": "test"}))
    )


_DEFAULT_EAGER_ITEM = {
    "inference_type": "logical",
    "statement": "Alice was born in 1993",
    "derivation": "age=33, current_year=2026",
    "source_fact_ids": [],
    "source_inference_ids": [],
}

# ---------------------------------------------------------------------------
# Eager pass endpoint — POST /api/characters/{id}/inferences/generate
# ---------------------------------------------------------------------------


async def test_generate_inferences_returns_200(
    client: AsyncClient, character: Character, fact: Fact
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            return_value=_mock_eager_pass_response([_DEFAULT_EAGER_ITEM])
        )
        response = await client.post(f"/api/characters/{character.id}/inferences/generate")
    assert response.status_code == 200


async def test_generate_inferences_returns_new_inferences_list(
    client: AsyncClient, character: Character, fact: Fact
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            return_value=_mock_eager_pass_response([_DEFAULT_EAGER_ITEM])
        )
        response = await client.post(f"/api/characters/{character.id}/inferences/generate")
    data = response.json()
    assert "new_inferences" in data
    assert isinstance(data["new_inferences"], list)


async def test_generate_inferences_stores_to_db(
    client: AsyncClient, character: Character, fact: Fact, db: aiosqlite.Connection
) -> None:
    item = dict(_DEFAULT_EAGER_ITEM, source_fact_ids=[fact.id])
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_eager_pass_response([item]))
        await client.post(f"/api/characters/{character.id}/inferences/generate")
    stored = await get_inferences(db, character.id)
    assert len(stored) == 1
    assert stored[0].status == "active"


async def test_generate_inferences_unknown_character_returns_404(
    client: AsyncClient,
) -> None:
    response = await client.post("/api/characters/9999/inferences/generate")
    assert response.status_code == 404


async def test_generate_inferences_respects_depth_cap(
    client: AsyncClient, character: Character, fact: Fact, db: aiosqlite.Connection
) -> None:
    from memories.services.inference_service import MAX_INFERENCE_DEPTH

    existing = await create_inference(
        db,
        character_id=character.id,
        statement="Deep existing",
        derivation="d",
        depth=MAX_INFERENCE_DEPTH,
    )
    deep_item = {
        "inference_type": "logical",
        "statement": "Exceeds depth cap",
        "derivation": "from deep",
        "source_fact_ids": [],
        "source_inference_ids": [existing.id],
    }
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_eager_pass_response([deep_item]))
        response = await client.post(f"/api/characters/{character.id}/inferences/generate")
    data = response.json()
    assert len(data["new_inferences"]) == 0
    stored = await get_inferences(db, character.id)
    assert not any(s.statement == "Exceeds depth cap" for s in stored)


async def test_generate_inferences_applies_breadth_cap(
    client: AsyncClient, character: Character, fact: Fact, db: aiosqlite.Connection
) -> None:
    from memories.services.inference_service import MAX_INFERENCE_BREADTH

    items = [
        {
            "inference_type": "logical",
            "statement": f"Inference {i}",
            "derivation": "d",
            "source_fact_ids": [fact.id],
            "source_inference_ids": [],
        }
        for i in range(MAX_INFERENCE_BREADTH + 2)
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_eager_pass_response(items))
        response = await client.post(f"/api/characters/{character.id}/inferences/generate")
    data = response.json()
    assert len(data["new_inferences"]) == MAX_INFERENCE_BREADTH


async def test_generate_inferences_empty_response_returns_empty_list(
    client: AsyncClient, character: Character, fact: Fact
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_eager_pass_response([]))
        response = await client.post(f"/api/characters/{character.id}/inferences/generate")
    data = response.json()
    assert response.status_code == 200
    assert data["new_inferences"] == []


async def test_generate_inferences_on_parse_error_returns_warning(
    client: AsyncClient, character: Character, fact: Fact
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            return_value=httpx.Response(
                200, content=make_ollama_ndjson("This is not valid JSON for an array")
            )
        )
        response = await client.post(f"/api/characters/{character.id}/inferences/generate")
    data = response.json()
    assert response.status_code == 200
    assert data["new_inferences"] == []
    assert "warning" in data


# ---------------------------------------------------------------------------
# Revalidate endpoint — POST /api/characters/{id}/inferences/revalidate
# ---------------------------------------------------------------------------


async def test_revalidate_returns_200(
    client: AsyncClient, character: Character, fact: Fact, db: aiosqlite.Connection
) -> None:
    await create_inference(
        db,
        character_id=character.id,
        statement="Dep on fact",
        derivation="d",
        source_fact_ids=[fact.id],
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_revalidation_response(False))
        response = await client.post(
            f"/api/characters/{character.id}/inferences/revalidate",
            json={"changed_fact_id": fact.id},
        )
    assert response.status_code == 200


async def test_revalidate_returns_stale_inferences(
    client: AsyncClient, character: Character, fact: Fact, db: aiosqlite.Connection
) -> None:
    await create_inference(
        db,
        character_id=character.id,
        statement="Dep on fact",
        derivation="d",
        source_fact_ids=[fact.id],
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_revalidation_response(False))
        response = await client.post(
            f"/api/characters/{character.id}/inferences/revalidate",
            json={"changed_fact_id": fact.id},
        )
    data = response.json()
    assert "stale_inferences" in data
    assert isinstance(data["stale_inferences"], list)


async def test_revalidate_marks_stale_in_db(
    client: AsyncClient, character: Character, fact: Fact, db: aiosqlite.Connection
) -> None:
    inf = await create_inference(
        db,
        character_id=character.id,
        statement="Will become stale",
        derivation="d",
        source_fact_ids=[fact.id],
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_revalidation_response(False))
        await client.post(
            f"/api/characters/{character.id}/inferences/revalidate",
            json={"changed_fact_id": fact.id},
        )
    stale = await get_inferences(db, character.id, status="stale")
    assert any(s.id == inf.id for s in stale)


async def test_revalidate_does_not_affect_unrelated_inferences(
    client: AsyncClient, character: Character, fact: Fact, db: aiosqlite.Connection
) -> None:
    unrelated = await create_inference(
        db,
        character_id=character.id,
        statement="Unrelated inference",
        derivation="d",
        source_fact_ids=[999],
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(return_value=_mock_revalidation_response(False))
        await client.post(
            f"/api/characters/{character.id}/inferences/revalidate",
            json={"changed_fact_id": fact.id},
        )
    active = await get_inferences(db, character.id, status="active")
    assert any(a.id == unrelated.id for a in active)


async def test_revalidate_unknown_character_returns_404(client: AsyncClient) -> None:
    response = await client.post(
        "/api/characters/9999/inferences/revalidate",
        json={"changed_fact_id": 1},
    )
    assert response.status_code == 404


async def test_revalidate_unknown_fact_returns_404(
    client: AsyncClient, character: Character
) -> None:
    response = await client.post(
        f"/api/characters/{character.id}/inferences/revalidate",
        json={"changed_fact_id": 9999},
    )
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Inference management — DELETE and PATCH
# ---------------------------------------------------------------------------


async def test_delete_inference_returns_204(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    inf = await create_inference(
        db, character_id=character.id, statement="To delete", derivation="d"
    )
    response = await client.delete(f"/api/characters/{character.id}/inferences/{inf.id}")
    assert response.status_code == 204


async def test_delete_inference_removes_from_db(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    inf = await create_inference(
        db, character_id=character.id, statement="Will be deleted", derivation="d"
    )
    await client.delete(f"/api/characters/{character.id}/inferences/{inf.id}")
    active = await get_inferences(db, character.id, status="active")
    assert not any(a.id == inf.id for a in active)


async def test_delete_inference_unknown_id_returns_404(
    client: AsyncClient, character: Character
) -> None:
    response = await client.delete(f"/api/characters/{character.id}/inferences/9999")
    assert response.status_code == 404


async def test_patch_inference_status_to_active_returns_200(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    inf = await create_inference(
        db, character_id=character.id, statement="Will become active", derivation="d"
    )
    # First mark as stale
    from memories.database import update_inference_status

    await update_inference_status(db, inf.id, "stale")

    response = await client.patch(
        f"/api/characters/{character.id}/inferences/{inf.id}",
        json={"status": "active"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "active"


async def test_patch_inference_status_to_stale_returns_200(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    inf = await create_inference(
        db, character_id=character.id, statement="Will become stale", derivation="d"
    )
    response = await client.patch(
        f"/api/characters/{character.id}/inferences/{inf.id}",
        json={"status": "stale"},
    )
    assert response.status_code == 200


async def test_patch_inference_status_updates_db(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    inf = await create_inference(
        db, character_id=character.id, statement="Test inference", derivation="d"
    )
    await client.patch(
        f"/api/characters/{character.id}/inferences/{inf.id}",
        json={"status": "invalidated"},
    )
    row = await (
        await db.execute("SELECT status FROM inferences WHERE id = ?", (inf.id,))
    ).fetchone()
    assert row[0] == "invalidated"


async def test_patch_inference_unknown_id_returns_404(
    client: AsyncClient, character: Character
) -> None:
    response = await client.patch(
        f"/api/characters/{character.id}/inferences/9999",
        json={"status": "active"},
    )
    assert response.status_code == 404
