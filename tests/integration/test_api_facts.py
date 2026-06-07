"""Integration tests for the facts API."""

from __future__ import annotations

import aiosqlite
from httpx import AsyncClient

from memories.database import create_inference, get_inferences
from memories.models import Character


async def test_add_fact_201(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"}
    )
    assert response.status_code == 201
    assert response.json()["key"] == "age"


async def test_add_fact_duplicate_key_409(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "31"}
    )
    assert response.status_code == 409


async def test_list_facts_for_character(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.get(f"/api/characters/{char_id}/facts")
    assert response.status_code == 200
    assert len(response.json()) == 1


async def test_update_fact_200(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    fact_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"}
    )
    fact_id = fact_resp.json()["id"]
    response = await client.put(f"/api/characters/{char_id}/facts/{fact_id}", json={"value": "31"})
    assert response.status_code == 200
    assert response.json()["value"] == "31"


async def test_update_nonexistent_fact_404(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.put(f"/api/characters/{char_id}/facts/99999", json={"value": "x"})
    assert response.status_code == 404


async def test_delete_fact_returns_200(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    fact_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"}
    )
    fact_id = fact_resp.json()["id"]
    response = await client.delete(f"/api/characters/{char_id}/facts/{fact_id}")
    assert response.status_code == 200


async def test_delete_nonexistent_fact_404(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.delete(f"/api/characters/{char_id}/facts/99999")
    assert response.status_code == 404


async def test_facts_for_unknown_character_404(client: AsyncClient) -> None:
    response = await client.get("/api/characters/9999/facts")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Inferences endpoint
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
    from memories.database import create_inference

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


# ---------------------------------------------------------------------------
# Phase 3 — DELETE fact cascade
# ---------------------------------------------------------------------------


async def test_delete_fact_response_has_invalidated_inferences_key(
    client: AsyncClient,
) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    fact_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"}
    )
    fact_id = fact_resp.json()["id"]
    response = await client.delete(f"/api/characters/{char_id}/facts/{fact_id}")
    assert "invalidated_inferences" in response.json()
    assert isinstance(response.json()["invalidated_inferences"], list)


async def test_delete_fact_cascade_marks_dependent_inference_invalidated(
    client: AsyncClient,
    character: Character,
    db: aiosqlite.Connection,
) -> None:
    # Create a fact and an inference that depends on it
    fact_resp = await client.post(
        f"/api/characters/{character.id}/facts", json={"key": "age", "value": "33"}
    )
    fact_id = fact_resp.json()["id"]
    inf = await create_inference(
        db,
        character_id=character.id,
        statement="Alice was born in 1993",
        derivation="age=33",
        source_fact_ids=[fact_id],
    )

    response = await client.delete(f"/api/characters/{character.id}/facts/{fact_id}")
    data = response.json()

    assert any(i["id"] == inf.id for i in data["invalidated_inferences"])
    invalidated_db = await get_inferences(db, character.id, status="invalidated")
    assert any(i.id == inf.id for i in invalidated_db)


async def test_delete_fact_cascade_leaves_unrelated_inference_active(
    client: AsyncClient,
    character: Character,
    db: aiosqlite.Connection,
) -> None:
    # Create a fact to delete, and an inference that references a DIFFERENT fact
    fact_resp = await client.post(
        f"/api/characters/{character.id}/facts", json={"key": "age", "value": "33"}
    )
    fact_id = fact_resp.json()["id"]
    unrelated = await create_inference(
        db,
        character_id=character.id,
        statement="Unrelated inference",
        derivation="d",
        source_fact_ids=[999],  # different fact id
    )

    await client.delete(f"/api/characters/{character.id}/facts/{fact_id}")

    active = await get_inferences(db, character.id, status="active")
    assert any(a.id == unrelated.id for a in active)


async def test_delete_fact_no_dependents_returns_empty_list(
    client: AsyncClient,
    character: Character,
    db: aiosqlite.Connection,
) -> None:
    fact_resp = await client.post(
        f"/api/characters/{character.id}/facts", json={"key": "age", "value": "33"}
    )
    fact_id = fact_resp.json()["id"]
    # Create an inference that does NOT depend on this fact
    await create_inference(
        db,
        character_id=character.id,
        statement="Independent inference",
        derivation="d",
        source_fact_ids=[999],
    )

    response = await client.delete(f"/api/characters/{character.id}/facts/{fact_id}")
    assert response.json()["invalidated_inferences"] == []


# ---------------------------------------------------------------------------
# Phase 4 additions — category, mutability, and ID-based endpoints
# ---------------------------------------------------------------------------


async def test_create_fact_with_category_returns_201_with_category(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Jon", "category": "user"},
    )
    assert response.status_code == 201
    assert response.json()["category"] == "user"


async def test_create_fact_with_mutability_returns_201_with_mutability(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "mood", "value": "cheerful", "mutability": "high"},
    )
    assert response.status_code == 201
    assert response.json()["mutability"] == "high"


async def test_create_fact_default_category_in_response(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "occupation", "value": "doctor"}
    )
    assert response.status_code == 201
    assert response.json()["category"] == "character"


async def test_create_fact_default_mutability_in_response(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "occupation", "value": "doctor"}
    )
    assert response.status_code == 201
    assert response.json()["mutability"] == "immutable"


async def test_create_fact_invalid_category_returns_422(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "x", "value": "y", "category": "invalid"},
    )
    assert response.status_code == 422


async def test_create_fact_invalid_mutability_returns_422(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "x", "value": "y", "mutability": "invalid"},
    )
    assert response.status_code == 422


async def test_list_facts_includes_category_field(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.get(f"/api/characters/{char_id}/facts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "category" in data[0]


async def test_list_facts_includes_mutability_field(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.get(f"/api/characters/{char_id}/facts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert "mutability" in data[0]


async def test_update_fact_value_preserves_category(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Alice", "category": "user"},
    )
    fact_id = create_resp.json()["id"]
    await client.put(f"/api/characters/{char_id}/facts/{fact_id}", json={"value": "Jon"})
    facts = (await client.get(f"/api/characters/{char_id}/facts")).json()
    assert facts[0]["category"] == "user"


async def test_update_fact_value_preserves_mutability(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "mood", "value": "cheerful", "mutability": "high"},
    )
    fact_id = create_resp.json()["id"]
    await client.put(f"/api/characters/{char_id}/facts/{fact_id}", json={"value": "anxious"})
    facts = (await client.get(f"/api/characters/{char_id}/facts")).json()
    assert facts[0]["mutability"] == "high"


async def test_update_fact_with_new_category_via_put(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "city", "value": "Chicago"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.put(
        f"/api/characters/{char_id}/facts/{fact_id}",
        json={"value": "Chicago", "category": "setting"},
    )
    assert response.status_code == 200
    assert response.json()["category"] == "setting"


async def test_update_fact_with_new_mutability_via_put(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "clothing", "value": "dark coat"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.put(
        f"/api/characters/{char_id}/facts/{fact_id}",
        json={"value": "dark coat", "mutability": "low"},
    )
    assert response.status_code == 200
    assert response.json()["mutability"] == "low"


async def test_patch_fact_mutability_returns_200(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "mood", "value": "cheerful"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.patch(
        f"/api/characters/{char_id}/facts/{fact_id}", json={"mutability": "high"}
    )
    assert response.status_code == 200
    assert response.json()["mutability"] == "high"


async def test_patch_fact_mutability_updates_db(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "mood", "value": "cheerful"}
    )
    fact_id = create_resp.json()["id"]
    await client.patch(f"/api/characters/{char_id}/facts/{fact_id}", json={"mutability": "high"})
    facts = (await client.get(f"/api/characters/{char_id}/facts")).json()
    assert facts[0]["mutability"] == "high"


async def test_patch_fact_category_returns_200(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "city", "value": "Chicago"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.patch(
        f"/api/characters/{char_id}/facts/{fact_id}", json={"category": "setting"}
    )
    assert response.status_code == 200
    assert response.json()["category"] == "setting"


async def test_patch_fact_category_updates_db(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "city", "value": "Chicago"}
    )
    fact_id = create_resp.json()["id"]
    await client.patch(f"/api/characters/{char_id}/facts/{fact_id}", json={"category": "setting"})
    facts = (await client.get(f"/api/characters/{char_id}/facts")).json()
    assert facts[0]["category"] == "setting"


async def test_patch_fact_value_not_changed_by_patch(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "mood", "value": "cheerful"}
    )
    fact_id = create_resp.json()["id"]
    await client.patch(f"/api/characters/{char_id}/facts/{fact_id}", json={"mutability": "high"})
    facts = (await client.get(f"/api/characters/{char_id}/facts")).json()
    assert facts[0]["value"] == "cheerful"


async def test_patch_fact_empty_body_returns_422(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "mood", "value": "cheerful"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.patch(f"/api/characters/{char_id}/facts/{fact_id}", json={})
    assert response.status_code == 422


async def test_patch_fact_unknown_id_returns_404(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.patch(
        f"/api/characters/{char_id}/facts/99999", json={"mutability": "high"}
    )
    assert response.status_code == 404


async def test_patch_fact_invalid_mutability_returns_422(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "mood", "value": "cheerful"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.patch(
        f"/api/characters/{char_id}/facts/{fact_id}", json={"mutability": "invalid"}
    )
    assert response.status_code == 422


async def test_patch_fact_invalid_category_returns_422(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "mood", "value": "cheerful"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.patch(
        f"/api/characters/{char_id}/facts/{fact_id}", json={"category": "invalid"}
    )
    assert response.status_code == 422


async def test_create_two_facts_same_key_different_categories(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    resp1 = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Jon", "category": "user"},
    )
    resp2 = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Elara", "category": "character"},
    )
    assert resp1.status_code == 201
    assert resp2.status_code == 201
    facts = (await client.get(f"/api/characters/{char_id}/facts")).json()
    name_facts = [f for f in facts if f["key"] == "name"]
    assert len(name_facts) == 2


async def test_create_two_facts_same_key_same_category_returns_409(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Jon", "category": "user"},
    )
    resp2 = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Bob", "category": "user"},
    )
    assert resp2.status_code == 409


async def test_put_fact_wrong_character_returns_404(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    # Create a second character and a fact for that character
    char2_resp = await client.post(
        "/api/characters/", json={"name": "Bob", "modelfile_base": "qwen3:7b"}
    )
    char2_id = char2_resp.json()["id"]
    fact_resp = await client.post(
        f"/api/characters/{char2_id}/facts", json={"key": "age", "value": "25"}
    )
    fact_id = fact_resp.json()["id"]
    # Try to update via character 1 — the fact belongs to character 2
    response = await client.put(
        f"/api/characters/{character.id}/facts/{fact_id}", json={"value": "30"}
    )
    assert response.status_code == 404


async def test_delete_fact_by_id_returns_200(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    create_resp = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"}
    )
    fact_id = create_resp.json()["id"]
    response = await client.delete(f"/api/characters/{char_id}/facts/{fact_id}")
    assert response.status_code == 200


async def test_patch_fact_category_conflict_returns_409(client: AsyncClient) -> None:
    """Patching a fact's category to one where (category, key) already exists returns 409."""
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    # Create user/name = "Jon" and character/name = "Elara"
    await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Jon", "category": "user"},
    )
    char_fact = await client.post(
        f"/api/characters/{char_id}/facts",
        json={"key": "name", "value": "Elara", "category": "character"},
    )
    elara_id = char_fact.json()["id"]
    # Try to re-categorise character/name → user, which would collide with user/name
    response = await client.patch(
        f"/api/characters/{char_id}/facts/{elara_id}", json={"category": "user"}
    )
    assert response.status_code == 409
