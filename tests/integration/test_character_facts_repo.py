"""Integration tests for the character_facts blob repository."""

from __future__ import annotations

import aiosqlite

from memories.database import (
    create_character,
    get_facts,
    patch_fact,
    set_facts,
)
from memories.models import Character


async def test_get_facts_returns_empty_dict_for_new_character(
    db: aiosqlite.Connection, character: Character
) -> None:
    result = await get_facts(db, character.id)
    assert result == {}


async def test_get_facts_returns_empty_dict_for_unknown_character_id(
    db: aiosqlite.Connection,
) -> None:
    result = await get_facts(db, 999999)
    assert result == {}


async def test_set_facts_then_get_facts_round_trips(
    db: aiosqlite.Connection, character: Character
) -> None:
    blob = {"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}
    await set_facts(db, character.id, blob)
    result = await get_facts(db, character.id)
    assert result["Character"]["Identity"]["Name"]["Value"] == "Sarah"


async def test_set_facts_replaces_existing_blob(
    db: aiosqlite.Connection, character: Character
) -> None:
    await set_facts(db, character.id, {"Character": {"Identity": {"Name": {"Value": "First"}}}})
    await set_facts(db, character.id, {"Character": {"State-Of-Mind": {"Mood": {"Value": "Calm"}}}})
    result = await get_facts(db, character.id)
    assert "Identity" not in result.get("Character", {})
    assert result["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Calm"


async def test_get_facts_keeps_valid_paths_drops_invalid_after_masking(
    db: aiosqlite.Connection, character: Character
) -> None:
    blob = {
        "Character": {"Identity": {"Name": {"Value": "Elena"}}},
        "BadPath": {"Nested": {"Value": "dropped"}},
    }
    await set_facts(db, character.id, blob)
    result = await get_facts(db, character.id)
    assert result["Character"]["Identity"]["Name"]["Value"] == "Elena"
    assert "BadPath" not in result


async def test_patch_fact_updates_single_nested_key(
    db: aiosqlite.Connection, character: Character
) -> None:
    await set_facts(
        db,
        character.id,
        {
            "Character": {
                "Identity": {"Name": {"Value": "Old"}},
                "State-Of-Mind": {"Mood": {"Value": "Calm"}},
            }
        },
    )
    await patch_fact(db, character.id, ("Character", "Identity", "Name"), "Elena")
    result = await get_facts(db, character.id)
    assert result["Character"]["Identity"]["Name"]["Value"] == "Elena"
    assert result["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Calm"


async def test_patch_fact_on_character_with_no_row_creates_row(
    db: aiosqlite.Connection, character: Character
) -> None:
    await patch_fact(db, character.id, ("Character", "Identity", "Name"), "Elena")
    result = await get_facts(db, character.id)
    assert result["Character"]["Identity"]["Name"]["Value"] == "Elena"


async def test_patch_fact_overwrite_existing_leaf(
    db: aiosqlite.Connection, character: Character
) -> None:
    await patch_fact(db, character.id, ("Character", "Identity", "Name"), "First")
    await patch_fact(db, character.id, ("Character", "Identity", "Name"), "Second")
    result = await get_facts(db, character.id)
    assert result["Character"]["Identity"]["Name"]["Value"] == "Second"


async def test_set_facts_multiple_characters_independent(
    db: aiosqlite.Connection,
) -> None:
    char_a = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    char_b = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    await set_facts(db, char_a.id, {"Character": {"Identity": {"Name": {"Value": "Alice"}}}})
    await set_facts(db, char_b.id, {"Character": {"Identity": {"Name": {"Value": "Bob"}}}})
    result_a = await get_facts(db, char_a.id)
    result_b = await get_facts(db, char_b.id)
    assert result_a["Character"]["Identity"]["Name"]["Value"] == "Alice"
    assert result_b["Character"]["Identity"]["Name"]["Value"] == "Bob"


async def test_get_facts_does_not_expose_updated_at(
    db: aiosqlite.Connection, character: Character
) -> None:
    await set_facts(db, character.id, {"Character": {"Identity": {"Name": {"Value": "Sarah"}}}})
    result = await get_facts(db, character.id)
    assert "updated_at" not in result
