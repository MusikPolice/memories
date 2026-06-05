"""Integration tests for the inferences repository."""

from __future__ import annotations

import aiosqlite
import pytest

from memories.database import (
    create_character,
    create_inference,
    delete_inference,
    get_inference,
    get_inferences,
    update_inference_status,
)
from memories.exceptions import NotFoundError


async def test_create_inference_stores_statement_and_derivation(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db,
        character_id=char.id,
        statement="Alice was born in 1991",
        derivation="age=33, current_year=2024",
        source_fact_ids=[1],
    )
    assert inf.statement == "Alice was born in 1991"
    assert inf.derivation == "age=33, current_year=2024"


async def test_create_inference_default_status_is_active(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db, character_id=char.id, statement="Test", derivation="from facts"
    )
    assert inf.status == "active"


async def test_create_inference_default_depth_is_one(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db, character_id=char.id, statement="Test", derivation="from facts"
    )
    assert inf.depth == 1


async def test_source_fact_ids_stored_and_retrieved_as_list(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db,
        character_id=char.id,
        statement="Test",
        derivation="from facts",
        source_fact_ids=[1, 2],
    )
    assert inf.source_fact_ids == [1, 2]


async def test_source_inference_ids_empty_list_when_not_set(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db, character_id=char.id, statement="Test", derivation="from facts"
    )
    assert inf.source_inference_ids == []


async def test_get_inferences_returns_active_only_by_default(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_inference(db, character_id=char.id, statement="Active", derivation="d")
    # Force-insert a stale one
    await db.execute(
        "INSERT INTO inferences "
        "(character_id, statement, derivation, depth, inference_type, status) "
        "VALUES (?, ?, ?, 1, 'logical', 'stale')",
        (char.id, "Stale inference", "d"),
    )
    await db.commit()
    inferences = await get_inferences(db, char.id)
    assert all(i.status == "active" for i in inferences)
    assert len(inferences) == 1


async def test_inferences_isolated_per_character(db: aiosqlite.Connection) -> None:
    char_a = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    char_b = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    await create_inference(
        db, character_id=char_a.id, statement="Alice's inference", derivation="d"
    )
    inferences_b = await get_inferences(db, char_b.id)
    assert inferences_b == []


async def test_inference_type_logical_stored_correctly(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db,
        character_id=char.id,
        statement="Test",
        derivation="d",
        inference_type="logical",
    )
    assert inf.inference_type == "logical"


async def test_inference_type_probabilistic_stored_correctly(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db,
        character_id=char.id,
        statement="Test",
        derivation="d",
        inference_type="probabilistic",
    )
    assert inf.inference_type == "probabilistic"


# ---------------------------------------------------------------------------
# Phase 3 additions — update_inference_status, get_inference, delete_inference
# ---------------------------------------------------------------------------


async def test_update_inference_status_to_stale(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(db, character_id=char.id, statement="Test", derivation="d")
    await update_inference_status(db, inf.id, "stale")
    row = await (
        await db.execute("SELECT status FROM inferences WHERE id = ?", (inf.id,))
    ).fetchone()
    assert row[0] == "stale"


async def test_update_inference_status_to_invalidated(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(db, character_id=char.id, statement="Test", derivation="d")
    await update_inference_status(db, inf.id, "invalidated")
    row = await (
        await db.execute("SELECT status FROM inferences WHERE id = ?", (inf.id,))
    ).fetchone()
    assert row[0] == "invalidated"


async def test_update_inference_status_to_active(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(db, character_id=char.id, statement="Test", derivation="d")
    await update_inference_status(db, inf.id, "stale")
    await update_inference_status(db, inf.id, "active")
    active = await get_inferences(db, char.id, status="active")
    assert any(a.id == inf.id for a in active)


async def test_update_inference_status_returns_updated_inference(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(db, character_id=char.id, statement="Test", derivation="d")
    updated = await update_inference_status(db, inf.id, "stale")
    assert updated.status == "stale"
    assert updated.id == inf.id


async def test_update_inference_status_nonexistent_raises(db: aiosqlite.Connection) -> None:
    with pytest.raises(NotFoundError):
        await update_inference_status(db, 9999, "stale")


async def test_get_inference_by_id_returns_correct_inference(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(
        db, character_id=char.id, statement="Specific statement", derivation="d"
    )
    fetched = await get_inference(db, inf.id)
    assert fetched is not None
    assert fetched.id == inf.id
    assert fetched.statement == "Specific statement"


async def test_get_inference_by_id_returns_none_for_unknown(db: aiosqlite.Connection) -> None:
    result = await get_inference(db, 9999)
    assert result is None


async def test_delete_inference_removes_row(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    inf = await create_inference(db, character_id=char.id, statement="To delete", derivation="d")
    await delete_inference(db, inf.id)
    active = await get_inferences(db, char.id, status="active")
    assert not any(a.id == inf.id for a in active)
    assert await get_inference(db, inf.id) is None


async def test_delete_inference_nonexistent_raises(db: aiosqlite.Connection) -> None:
    with pytest.raises(NotFoundError):
        await delete_inference(db, 9999)


async def test_get_inferences_stale_status_filter(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    active_inf = await create_inference(
        db, character_id=char.id, statement="Active one", derivation="d"
    )
    stale_inf = await create_inference(
        db, character_id=char.id, statement="Stale one", derivation="d"
    )
    await update_inference_status(db, stale_inf.id, "stale")

    stale_list = await get_inferences(db, char.id, status="stale")
    assert all(s.status == "stale" for s in stale_list)
    assert any(s.id == stale_inf.id for s in stale_list)
    assert not any(s.id == active_inf.id for s in stale_list)


async def test_get_inferences_invalidated_status_filter(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    active_inf = await create_inference(
        db, character_id=char.id, statement="Active one", derivation="d"
    )
    inv_inf = await create_inference(
        db, character_id=char.id, statement="Invalidated one", derivation="d"
    )
    await update_inference_status(db, inv_inf.id, "invalidated")

    inv_list = await get_inferences(db, char.id, status="invalidated")
    assert all(i.status == "invalidated" for i in inv_list)
    assert any(i.id == inv_inf.id for i in inv_list)
    assert not any(i.id == active_inf.id for i in inv_list)
