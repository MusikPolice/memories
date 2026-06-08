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


async def test_foreign_keys_enabled(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA foreign_keys")
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1


# ---------------------------------------------------------------------------
# Phase 4 additions — category and mutability columns on facts table
# ---------------------------------------------------------------------------


async def test_facts_table_has_category_column(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(facts)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "category" in col_names


async def test_facts_table_has_mutability_column(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(facts)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "mutability" in col_names


async def test_category_column_default_is_character(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(facts)")
    rows = await cursor.fetchall()
    # PRAGMA table_info columns: cid(0), name(1), type(2), notnull(3), dflt_value(4), pk(5)
    category_row = next((r for r in rows if r[1] == "category"), None)
    assert category_row is not None, "category column not found"
    dflt_value = category_row[4]
    assert dflt_value is not None, "category column has no default"
    assert "character" in dflt_value


async def test_mutability_column_default_is_immutable(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(facts)")
    rows = await cursor.fetchall()
    mutability_row = next((r for r in rows if r[1] == "mutability"), None)
    assert mutability_row is not None, "mutability column not found"
    dflt_value = mutability_row[4]
    assert dflt_value is not None, "mutability column has no default"
    assert "immutable" in dflt_value


async def test_facts_uniqueness_constraint_is_category_scoped(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='facts'")
    row = await cursor.fetchone()
    assert row is not None
    table_sql: str = row[0]
    # The unique constraint must reference the category column
    assert "category" in table_sql, "Expected 'category' in facts table DDL unique constraint"
    assert "UNIQUE" in table_sql.upper(), "Expected UNIQUE constraint in facts table DDL"


# ---------------------------------------------------------------------------
# Phase 5 additions — experiences table and sessions.closing_journal
# ---------------------------------------------------------------------------


async def test_experiences_table_exists(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(experiences)")
    rows = await cursor.fetchall()
    assert len(rows) > 0, "experiences table should have columns"


async def test_experiences_table_has_embedding_column(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(experiences)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "embedding" in col_names


async def test_experiences_table_has_approved_at_column(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(experiences)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "approved_at" in col_names


async def test_sessions_table_has_closing_journal_column(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(sessions)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "closing_journal" in col_names
