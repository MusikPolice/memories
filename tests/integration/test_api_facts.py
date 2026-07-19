"""Integration tests for the facts API (Facts v2 blob model)."""

from __future__ import annotations

import aiosqlite
from httpx import AsyncClient

from memories.database import create_inference, get_facts, set_facts
from memories.models import Character

# ---------------------------------------------------------------------------
# GET /{character_id}/facts — masked blob read
# ---------------------------------------------------------------------------


async def test_get_facts_returns_empty_blob_for_new_character(
    client: AsyncClient, character: Character
) -> None:
    response = await client.get(f"/api/characters/{character.id}/facts")
    assert response.status_code == 200
    assert response.json() == {}


async def test_get_facts_returns_masked_blob(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    blob = {"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}
    await set_facts(db, character.id, blob)
    response = await client.get(f"/api/characters/{character.id}/facts")
    assert response.status_code == 200
    assert response.json() == blob


async def test_get_facts_unknown_character_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/characters/9999/facts")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# PUT /{character_id}/facts — by-path value write
# ---------------------------------------------------------------------------


async def test_put_fact_writes_string_leaf_and_returns_blob(
    client: AsyncClient, character: Character
) -> None:
    response = await client.put(
        f"/api/characters/{character.id}/facts",
        json={"path": "Character.Identity.Name", "value": "Sarah"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["Character"]["Identity"]["Name"]["Value"] == "Sarah"


async def test_put_fact_coerces_integer_leaf(client: AsyncClient, character: Character) -> None:
    response = await client.put(
        f"/api/characters/{character.id}/facts",
        json={"path": "Character.Identity.Age", "value": "34"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["Character"]["Identity"]["Age"]["Value"] == 34


async def test_put_fact_coerces_enum_case_insensitively(
    client: AsyncClient, character: Character
) -> None:
    response = await client.put(
        f"/api/characters/{character.id}/facts",
        json={"path": "Character.State-Of-Mind.Mood", "value": "anxious"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Anxious"


async def test_put_fact_invalid_enum_returns_422(client: AsyncClient, character: Character) -> None:
    response = await client.put(
        f"/api/characters/{character.id}/facts",
        json={"path": "Character.State-Of-Mind.Mood", "value": "ecstatic"},
    )
    assert response.status_code == 422


async def test_put_fact_invalid_integer_returns_422(
    client: AsyncClient, character: Character
) -> None:
    response = await client.put(
        f"/api/characters/{character.id}/facts",
        json={"path": "Character.Identity.Age", "value": "abc"},
    )
    assert response.status_code == 422


async def test_put_fact_unknown_path_returns_422(client: AsyncClient, character: Character) -> None:
    response = await client.put(
        f"/api/characters/{character.id}/facts",
        json={"path": "Nope.Not.Here", "value": "x"},
    )
    assert response.status_code == 422


async def test_put_fact_unknown_character_returns_404(client: AsyncClient) -> None:
    response = await client.put(
        "/api/characters/9999/facts",
        json={"path": "Character.Identity.Name", "value": "Sarah"},
    )
    assert response.status_code == 404


async def test_put_fact_persists_to_db(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    await client.put(
        f"/api/characters/{character.id}/facts",
        json={"path": "Character.Identity.Name", "value": "Sarah"},
    )
    blob = await get_facts(db, character.id)
    assert blob["Character"]["Identity"]["Name"]["Value"] == "Sarah"


# ---------------------------------------------------------------------------
# Inferences endpoint (unchanged in Step 8 — kept)
# ---------------------------------------------------------------------------


async def test_list_inferences_empty_for_new_character(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.get(f"/api/characters/{char_id}/inferences")
    assert response.status_code == 200
    assert response.json() == []


async def test_list_inferences_returns_active_inferences(
    client: AsyncClient, db: aiosqlite.Connection
) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await create_inference(
        db,
        character_id=char_id,
        statement="Works long hours",
        derivation="occupation=surgeon",
        source_fact_ids=[],
        inference_type="probabilistic",
    )

    response = await client.get(f"/api/characters/{char_id}/inferences")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["statement"] == "Works long hours"
    assert data[0]["inference_type"] == "probabilistic"


async def test_list_inferences_unknown_character_404(client: AsyncClient) -> None:
    response = await client.get("/api/characters/9999/inferences")
    assert response.status_code == 404
