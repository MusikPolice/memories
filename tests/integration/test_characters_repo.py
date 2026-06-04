"""Integration tests for the characters repository."""

import aiosqlite

from memories.database import create_character, get_character, list_characters


async def test_create_character_returns_with_id(db: aiosqlite.Connection) -> None:
    character = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    assert character.id > 0


async def test_get_character_by_id(db: aiosqlite.Connection) -> None:
    created = await create_character(db, name="Carol", modelfile_base="qwen3:7b")
    fetched = await get_character(db, created.id)
    assert fetched is not None
    assert fetched.name == "Carol"
    assert fetched.id == created.id


async def test_get_nonexistent_character_returns_none(db: aiosqlite.Connection) -> None:
    result = await get_character(db, 9999)
    assert result is None


async def test_list_characters_empty(db: aiosqlite.Connection) -> None:
    characters = await list_characters(db)
    assert characters == []


async def test_list_characters_multiple(db: aiosqlite.Connection) -> None:
    await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    characters = await list_characters(db)
    assert len(characters) == 2
    names = {c.name for c in characters}
    assert names == {"Alice", "Bob"}
