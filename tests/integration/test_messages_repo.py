"""Integration tests for the messages repository."""

import aiosqlite
import pytest

from memories.database import (
    create_character,
    create_session,
    get_messages,
    replace_message_content,
    store_message,
)
from memories.exceptions import NotFoundError


async def test_store_user_message(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    msg = await store_message(
        db,
        session_id=session.id,
        character_id=char.id,
        role="user",
        content="Hello",
        turn_id=1,
    )
    assert msg.role == "user"
    assert msg.content == "Hello"


async def test_store_assistant_message(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    msg = await store_message(
        db,
        session_id=session.id,
        character_id=char.id,
        role="assistant",
        content="Hi there!",
        turn_id=1,
    )
    assert msg.role == "assistant"


async def test_get_messages_ordered_by_turn_id(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    await store_message(
        db,
        session_id=session.id,
        character_id=char.id,
        role="assistant",
        content="Second",
        turn_id=2,
    )
    await store_message(
        db,
        session_id=session.id,
        character_id=char.id,
        role="user",
        content="First",
        turn_id=1,
    )
    messages = await get_messages(db, session.id)
    assert messages[0].turn_id == 1
    assert messages[1].turn_id == 2


async def test_messages_isolated_per_session(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session_a = await create_session(db, character_id=char.id)
    session_b = await create_session(db, character_id=char.id)
    await store_message(
        db,
        session_id=session_a.id,
        character_id=char.id,
        role="user",
        content="Message A",
        turn_id=1,
    )
    messages_b = await get_messages(db, session_b.id)
    assert messages_b == []


async def test_stored_message_has_no_removed_attributes(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    msg = await store_message(
        db,
        session_id=session.id,
        character_id=char.id,
        role="user",
        content="Hello",
        turn_id=1,
    )
    assert not hasattr(msg, "segment_id")
    assert not hasattr(msg, "ungrounded_implications")
    assert not hasattr(msg, "captured_by")


async def test_replace_message_content_updates_content(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    await store_message(
        db,
        session_id=session.id,
        character_id=char.id,
        role="user",
        content="Hi",
        turn_id=1,
    )
    await store_message(
        db,
        session_id=session.id,
        character_id=char.id,
        role="assistant",
        content="Old content",
        turn_id=1,
    )
    updated = await replace_message_content(
        db, session_id=session.id, turn_id=1, new_content="New content"
    )
    assert updated.content == "New content"
    messages = await get_messages(db, session.id)
    assistant_msg = next(m for m in messages if m.role == "assistant")
    assert assistant_msg.content == "New content"


async def test_replace_message_content_nonexistent_turn_raises(
    db: aiosqlite.Connection,
) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    with pytest.raises(NotFoundError):
        await replace_message_content(db, session_id=session.id, turn_id=99, new_content="x")
