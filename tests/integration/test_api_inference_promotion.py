"""Integration tests for the inference promote endpoint (Phase 4)."""

from __future__ import annotations

import aiosqlite
import pytest
from httpx import AsyncClient

from memories.database import create_inference, get_facts, get_inferences
from memories.models import Character, Inference

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def inference(db: aiosqlite.Connection, character: Character) -> Inference:
    return await create_inference(
        db,
        character_id=character.id,
        statement="Elara was born in 1993",
        derivation="age=33, current_year=2026",
        source_fact_ids=[],
        source_inference_ids=[],
        inference_type="logical",
    )


@pytest.fixture
async def downstream_inference(
    db: aiosqlite.Connection, character: Character, inference: Inference
) -> Inference:
    """An inference that derives from `inference`."""
    return await create_inference(
        db,
        character_id=character.id,
        statement="Elara was a teenager during 9/11",
        derivation="born in 1993 (inference:1)",
        source_fact_ids=[],
        source_inference_ids=[inference.id],
        inference_type="logical",
        depth=2,
    )


@pytest.fixture
async def unrelated_inference(db: aiosqlite.Connection, character: Character) -> Inference:
    """An active inference unrelated to the promoted one."""
    return await create_inference(
        db,
        character_id=character.id,
        statement="Elara likely works long hours",
        derivation="occupation=surgeon",
        source_fact_ids=[],
        source_inference_ids=[],
        inference_type="probabilistic",
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _promote_url(char_id: int, inf_id: int) -> str:
    return f"/api/characters/{char_id}/inferences/{inf_id}/promote"


# ---------------------------------------------------------------------------
# Phase 4 inference promotion tests
# ---------------------------------------------------------------------------


async def test_promote_inference_returns_201(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201


async def test_promote_inference_response_contains_fact(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201
    assert "fact" in response.json()


async def test_promote_inference_fact_has_correct_key_and_value(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201
    fact = response.json()["fact"]
    assert fact["key"] == "birth_year"
    assert fact["value"] == "1993"


async def test_promote_inference_default_category_is_character(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201
    assert response.json()["fact"]["category"] == "character"


async def test_promote_inference_default_mutability_is_immutable(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201
    assert response.json()["fact"]["mutability"] == "immutable"


async def test_promote_inference_with_custom_category(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993", "category": "setting"},
    )
    assert response.status_code == 201
    assert response.json()["fact"]["category"] == "setting"


async def test_promote_inference_with_custom_mutability(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993", "mutability": "high"},
    )
    assert response.status_code == 201
    assert response.json()["fact"]["mutability"] == "high"


async def test_promote_inference_fact_stored_in_db(
    client: AsyncClient, character: Character, inference: Inference, db: aiosqlite.Connection
) -> None:
    await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    facts = await get_facts(db, character.id)
    assert any(f.key == "birth_year" for f in facts)


async def test_promote_inference_deletes_source_inference(
    client: AsyncClient, character: Character, inference: Inference, db: aiosqlite.Connection
) -> None:
    await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    all_inferences = await get_inferences(db, character.id, status="all")
    assert not any(i.id == inference.id for i in all_inferences)


async def test_promote_inference_response_contains_stale_inferences_list(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201
    assert "stale_inferences" in response.json()
    assert isinstance(response.json()["stale_inferences"], list)


async def test_promote_inference_marks_downstream_inference_stale(
    client: AsyncClient,
    character: Character,
    inference: Inference,
    downstream_inference: Inference,
    db: aiosqlite.Connection,
) -> None:
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201
    stale_ids = [i["id"] for i in response.json()["stale_inferences"]]
    assert downstream_inference.id in stale_ids

    stale_in_db = await get_inferences(db, character.id, status="stale")
    assert any(i.id == downstream_inference.id for i in stale_in_db)


async def test_promote_inference_marks_transitive_downstream_stale(
    client: AsyncClient,
    character: Character,
    inference: Inference,
    downstream_inference: Inference,
    db: aiosqlite.Connection,
) -> None:
    """Grandchild inference (A→B→C) must be marked stale when A is promoted."""
    grandchild = await create_inference(
        db,
        character_id=character.id,
        statement="Elara missed the Columbine shooting by two years",
        derivation="was a teenager during 9/11 (inference:2)",
        source_fact_ids=[],
        source_inference_ids=[downstream_inference.id],
        inference_type="logical",
        depth=3,
    )

    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 201

    stale_ids = [i["id"] for i in response.json()["stale_inferences"]]
    assert downstream_inference.id in stale_ids, "direct child should be stale"
    assert grandchild.id in stale_ids, "grandchild should also be stale (transitive cascade)"

    stale_in_db = await get_inferences(db, character.id, status="stale")
    stale_db_ids = {i.id for i in stale_in_db}
    assert downstream_inference.id in stale_db_ids
    assert grandchild.id in stale_db_ids


async def test_promote_inference_non_downstream_inference_stays_active(
    client: AsyncClient,
    character: Character,
    inference: Inference,
    unrelated_inference: Inference,
    db: aiosqlite.Connection,
) -> None:
    await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    active = await get_inferences(db, character.id, status="active")
    assert any(i.id == unrelated_inference.id for i in active)


async def test_promote_inference_unknown_character_returns_404(
    client: AsyncClient, inference: Inference
) -> None:
    response = await client.post(
        _promote_url(99999, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 404


async def test_promote_inference_unknown_inference_returns_404(
    client: AsyncClient, character: Character
) -> None:
    response = await client.post(
        _promote_url(character.id, 99999),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 404


async def test_promote_inference_from_different_character_returns_404(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    # Create a second character
    char2_resp = await client.post(
        "/api/characters/", json={"name": "Bob", "modelfile_base": "qwen3:7b"}
    )
    char2_id = char2_resp.json()["id"]
    # Try to promote character 1's inference via character 2's endpoint
    response = await client.post(
        _promote_url(char2_id, inference.id),
        json={"key": "birth_year", "value": "1993"},
    )
    assert response.status_code == 404


async def test_promote_inference_key_already_exists_in_same_category_returns_409(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    # Create a fact with the same key and category first
    await client.post(
        f"/api/characters/{character.id}/facts",
        json={"key": "birth_year", "value": "1990", "category": "character"},
    )
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993", "category": "character"},
    )
    assert response.status_code == 409


async def test_promote_inference_key_exists_in_different_category_succeeds(
    client: AsyncClient, character: Character, inference: Inference
) -> None:
    # Create a "user" fact with the same key
    await client.post(
        f"/api/characters/{character.id}/facts",
        json={"key": "birth_year", "value": "1990", "category": "user"},
    )
    # Promote inference as "character" category — different category, no conflict
    response = await client.post(
        _promote_url(character.id, inference.id),
        json={"key": "birth_year", "value": "1993", "category": "character"},
    )
    assert response.status_code == 201


async def test_promote_inference_stale_inference_can_be_promoted(
    client: AsyncClient, character: Character, db: aiosqlite.Connection
) -> None:
    # Create an inference and mark it stale
    stale_inf = await create_inference(
        db,
        character_id=character.id,
        statement="Stale inference",
        derivation="some derivation",
        source_fact_ids=[],
        inference_type="logical",
    )
    await db.execute("UPDATE inferences SET status = 'stale' WHERE id = ?", (stale_inf.id,))
    await db.commit()

    response = await client.post(
        _promote_url(character.id, stale_inf.id),
        json={"key": "stale_key", "value": "stale_value"},
    )
    assert response.status_code == 201
