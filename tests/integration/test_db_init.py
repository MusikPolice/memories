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
    "character_facts",
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


# ---------------------------------------------------------------------------
# Step 2 additions — character_facts, updated decisions, updated messages,
# removed segments
# ---------------------------------------------------------------------------


async def test_segments_table_does_not_exist(db: aiosqlite.Connection) -> None:
    cursor = await db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='segments'"
    )
    row = await cursor.fetchone()
    assert row is None


async def test_character_facts_table_exists(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(character_facts)")
    rows = await cursor.fetchall()
    assert len(rows) > 0, "character_facts table should have columns"


async def test_character_facts_table_columns(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(character_facts)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "character_id" in col_names
    assert "facts_json" in col_names
    assert "updated_at" in col_names


async def test_character_facts_json_default_is_empty_object(db: aiosqlite.Connection) -> None:
    await db.execute(
        "INSERT INTO characters (name, modelfile_base) VALUES (?, ?)", ("Test", "qwen3:7b")
    )
    await db.commit()
    cursor = await db.execute("SELECT last_insert_rowid()")
    row = await cursor.fetchone()
    assert row is not None
    char_id = row[0]
    await db.execute("INSERT INTO character_facts (character_id) VALUES (?)", (char_id,))
    await db.commit()
    cursor = await db.execute(
        "SELECT facts_json FROM character_facts WHERE character_id = ?", (char_id,)
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == "{}"


async def test_decisions_table_columns(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(decisions)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "pass_name" in col_names
    assert "tool_name" in col_names
    assert "tool_args" in col_names
    assert "user_input" in col_names
    assert "reasoning" not in col_names
    assert "verdict" not in col_names
    assert "violations" not in col_names


async def test_messages_table_removed_columns(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(messages)")
    rows = await cursor.fetchall()
    col_names = [row[1] for row in rows]
    assert "segment_id" not in col_names
    assert "captured_by" not in col_names
    assert "ungrounded_implications" not in col_names
