"""Integration tests for the inferences repository."""

from __future__ import annotations

import aiosqlite

from memories.database import (
    create_character,
    create_inference,
    get_inferences,
)


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
