"""Integration tests for the facts repository."""

import sqlite3

import aiosqlite
import pytest

from memories.database import (
    create_character,
    create_fact,
    delete_fact,
    get_fact_by_category_key,
    get_facts,
    patch_fact,
    update_fact,
)
from memories.exceptions import NotFoundError


async def test_create_fact_stores_key_and_value(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="occupation", value="doctor")
    assert fact.key == "occupation"
    assert fact.value == "doctor"


async def test_create_fact_duplicate_key_raises(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_fact(db, character_id=char.id, key="occupation", value="doctor")
    with pytest.raises(sqlite3.IntegrityError):
        await create_fact(db, character_id=char.id, key="occupation", value="surgeon")


async def test_get_facts_returns_only_own_character(db: aiosqlite.Connection) -> None:
    alice = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    bob = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    await create_fact(db, character_id=alice.id, key="age", value="30")
    await create_fact(db, character_id=bob.id, key="age", value="40")
    alice_facts = await get_facts(db, alice.id)
    assert len(alice_facts) == 1
    assert alice_facts[0].value == "30"


async def test_update_fact_changes_value(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_fact(db, character_id=char.id, key="occupation", value="doctor")
    await update_fact(db, character_id=char.id, key="occupation", value="surgeon")
    facts = await get_facts(db, char.id)
    assert facts[0].value == "surgeon"


async def test_update_nonexistent_fact_raises(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    with pytest.raises(NotFoundError):
        await update_fact(db, character_id=char.id, key="nonexistent", value="x")


async def test_delete_fact_removes_it(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_fact(db, character_id=char.id, key="occupation", value="doctor")
    await delete_fact(db, character_id=char.id, key="occupation")
    facts = await get_facts(db, char.id)
    assert facts == []


async def test_delete_nonexistent_fact_raises(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    with pytest.raises(NotFoundError):
        await delete_fact(db, character_id=char.id, key="nonexistent")


# ---------------------------------------------------------------------------
# Phase 4 additions — category, mutability, and ID-based operations
# ---------------------------------------------------------------------------


async def test_create_fact_default_category_is_character(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="occupation", value="doctor")
    assert fact.category == "character"


async def test_create_fact_default_mutability_is_immutable(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="occupation", value="doctor")
    assert fact.mutability == "immutable"


async def test_create_fact_with_user_category(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="name", value="Jon", category="user")
    assert fact.category == "user"


async def test_create_fact_with_setting_category(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(
        db, character_id=char.id, key="location", value="Chicago", category="setting"
    )
    assert fact.category == "setting"


async def test_create_fact_with_low_mutability(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(
        db, character_id=char.id, key="clothing", value="dark coat", mutability="low"
    )
    assert fact.mutability == "low"


async def test_create_fact_with_high_mutability(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(
        db, character_id=char.id, key="mood", value="cheerful", mutability="high"
    )
    assert fact.mutability == "high"


async def test_update_fact_value_does_not_change_category(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="name", value="Alice", category="user")
    updated = await update_fact(db, fact_id=fact.id, value="Bob")
    assert updated.category == "user"


async def test_update_fact_value_does_not_change_mutability(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(
        db, character_id=char.id, key="mood", value="cheerful", mutability="high"
    )
    updated = await update_fact(db, fact_id=fact.id, value="anxious")
    assert updated.mutability == "high"


async def test_update_fact_category_changes_correctly(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="location", value="Chicago")
    updated = await update_fact(db, fact_id=fact.id, value="Chicago", category="setting")
    assert updated.category == "setting"


async def test_update_fact_mutability_changes_correctly(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="clothing", value="dark coat")
    updated = await update_fact(db, fact_id=fact.id, value="dark coat", mutability="low")
    assert updated.mutability == "low"


async def test_update_fact_all_three_fields_simultaneously(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(db, character_id=char.id, key="name", value="Alice")
    updated = await update_fact(
        db, fact_id=fact.id, value="Jon", category="user", mutability="high"
    )
    assert updated.value == "Jon"
    assert updated.category == "user"
    assert updated.mutability == "high"


async def test_patch_fact_mutability_only(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(
        db, character_id=char.id, key="mood", value="cheerful", category="character"
    )
    updated = await patch_fact(db, fact_id=fact.id, mutability="high")
    assert updated.mutability == "high"
    assert updated.value == "cheerful"
    assert updated.category == "character"


async def test_patch_fact_category_only(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact = await create_fact(
        db, character_id=char.id, key="name", value="Alice", mutability="immutable"
    )
    updated = await patch_fact(db, fact_id=fact.id, category="user")
    assert updated.category == "user"
    assert updated.value == "Alice"
    assert updated.mutability == "immutable"


async def test_patch_fact_raises_not_found_for_unknown_id(db: aiosqlite.Connection) -> None:
    with pytest.raises(NotFoundError):
        await patch_fact(db, fact_id=99999, mutability="high")


async def test_same_key_allowed_in_different_categories(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    fact_user = await create_fact(
        db, character_id=char.id, key="name", value="Jon", category="user"
    )
    fact_char = await create_fact(
        db, character_id=char.id, key="name", value="Elara", category="character"
    )
    facts = await get_facts(db, char.id)
    name_facts = [f for f in facts if f.key == "name"]
    assert len(name_facts) == 2
    assert fact_user.id != fact_char.id


async def test_same_key_same_category_raises_integrity_error(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_fact(db, character_id=char.id, key="name", value="Alice", category="user")
    with pytest.raises(sqlite3.IntegrityError):
        await create_fact(db, character_id=char.id, key="name", value="Bob", category="user")


async def test_get_fact_by_category_key_returns_matching_fact(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_fact(db, character_id=char.id, key="name", value="Jon", category="user")
    await create_fact(db, character_id=char.id, key="name", value="Elara", category="character")
    result = await get_fact_by_category_key(db, character_id=char.id, category="user", key="name")
    assert result is not None
    assert result.value == "Jon"
    assert result.category == "user"


async def test_get_fact_by_category_key_returns_none_for_missing(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_fact(db, character_id=char.id, key="name", value="Jon", category="user")
    result = await get_fact_by_category_key(
        db, character_id=char.id, category="setting", key="name"
    )
    assert result is None
