"""Integration tests for the facts repository."""

import sqlite3

import aiosqlite
import pytest

from memories.database import create_character, create_fact, delete_fact, get_facts, update_fact
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
