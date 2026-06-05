"""Integration tests for the messages repository."""

import aiosqlite
import pytest

from memories.database import (
    create_character,
    create_session,
    get_active_segment,
    get_messages,
    replace_message_content,
    store_message,
)
from memories.exceptions import NotFoundError


async def test_store_user_message(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    segment = await get_active_segment(db, session.id)
    msg = await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
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
    segment = await get_active_segment(db, session.id)
    msg = await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
        character_id=char.id,
        role="assistant",
        content="Hi there!",
        turn_id=1,
    )
    assert msg.role == "assistant"


async def test_get_messages_ordered_by_turn_id(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    segment = await get_active_segment(db, session.id)
    await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
        character_id=char.id,
        role="assistant",
        content="Second",
        turn_id=2,
    )
    await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
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
    seg_a = await get_active_segment(db, session_a.id)
    await store_message(
        db,
        session_id=session_a.id,
        segment_id=seg_a.id,
        character_id=char.id,
        role="user",
        content="Message A",
        turn_id=1,
    )
    messages_b = await get_messages(db, session_b.id)
    assert messages_b == []


async def test_messages_reference_segment(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    segment = await get_active_segment(db, session.id)
    msg = await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
        character_id=char.id,
        role="user",
        content="Hello",
        turn_id=1,
    )
    assert msg.segment_id == segment.id


# ---------------------------------------------------------------------------
# replace_message_content
# ---------------------------------------------------------------------------


async def test_replace_message_content_updates_text(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    segment = await get_active_segment(db, session.id)
    await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
        character_id=char.id,
        role="user",
        content="Hi",
        turn_id=1,
    )
    await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
        character_id=char.id,
        role="assistant",
        content="Old content",
        turn_id=1,
        ungrounded_implications=[
            {"type": "implication", "description": "x", "suggested_fact": None}
        ],
    )
    updated = await replace_message_content(
        db, session_id=session.id, turn_id=1, new_content="New content"
    )
    assert updated.content == "New content"


async def test_replace_message_content_clears_ungrounded(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    segment = await get_active_segment(db, session.id)
    await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
        character_id=char.id,
        role="user",
        content="Hi",
        turn_id=1,
    )
    await store_message(
        db,
        session_id=session.id,
        segment_id=segment.id,
        character_id=char.id,
        role="assistant",
        content="Old",
        turn_id=1,
        ungrounded_implications=[
            {"type": "implication", "description": "x", "suggested_fact": None}
        ],
    )
    updated = await replace_message_content(db, session_id=session.id, turn_id=1, new_content="New")
    assert updated.ungrounded_implications is None


async def test_replace_message_content_nonexistent_turn_raises(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    with pytest.raises(NotFoundError):
        await replace_message_content(db, session_id=session.id, turn_id=99, new_content="x")
