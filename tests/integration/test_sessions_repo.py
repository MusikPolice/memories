"""Integration tests for the sessions repository."""

import aiosqlite

from memories.database import create_character, create_session, end_session, get_session


async def test_create_session_sets_character_id(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    assert session.character_id == char.id


async def test_create_session_creates_initial_segment(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    cursor = await db.execute(
        "SELECT * FROM segments WHERE session_id = ? AND boundary_reason = 'session_start'",
        (session.id,),
    )
    row = await cursor.fetchone()
    assert row is not None


async def test_end_session_sets_ended_at(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    ended = await end_session(db, session.id)
    assert ended.ended_at is not None


async def test_get_session_by_id(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    created = await create_session(db, character_id=char.id)
    fetched = await get_session(db, created.id)
    assert fetched is not None
    assert fetched.id == created.id


async def test_get_nonexistent_session_returns_none(db: aiosqlite.Connection) -> None:
    result = await get_session(db, 9999)
    assert result is None
