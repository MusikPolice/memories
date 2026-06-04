"""Integration tests for database initialisation."""

import aiosqlite

from memories.database import init_db

_EXPECTED_TABLES = {
    "characters",
    "sessions",
    "facts",
    "inferences",
    "experiences",
    "decisions",
    "segments",
    "messages",
}


async def test_all_tables_created(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = await cursor.fetchall()
    table_names = {row[0] for row in rows}
    assert _EXPECTED_TABLES.issubset(table_names)


async def test_init_is_idempotent(db: aiosqlite.Connection) -> None:
    await init_db(db)
