"""Chat turn orchestration."""

from __future__ import annotations

import aiosqlite

from memories.database import (
    get_active_segment,
    get_character,
    get_facts,
    get_messages,
    get_session,
    next_turn_id,
    store_message,
)
from memories.exceptions import NotFoundError
from memories.services.ollama_client import OllamaClient
from memories.services.prompt_builder import build_system_prompt


async def run_turn(
    db: aiosqlite.Connection,
    session_id: int,
    user_content: str,
    ollama: OllamaClient,
    think: bool = False,
) -> tuple[str, str, int]:
    """Execute one conversation turn.

    Returns ``(response_content, thinking_text, turn_id)``.  *thinking_text* is
    the model's reasoning monologue; empty string if the model did not think.
    """
    session = await get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")
    if session.ended_at is not None:
        raise NotFoundError(f"Session {session_id} has ended")

    character = await get_character(db, session.character_id)
    assert character is not None

    facts = await get_facts(db, session.character_id)
    system_prompt = build_system_prompt(character, facts)
    history = await get_messages(db, session_id)
    segment = await get_active_segment(db, session_id)
    turn_id = await next_turn_id(db, session_id)

    await store_message(
        db,
        session_id=session_id,
        segment_id=segment.id,
        character_id=session.character_id,
        role="user",
        content=user_content,
        turn_id=turn_id,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        messages.append({"role": msg.role, "content": msg.content})
    messages.append({"role": "user", "content": user_content})

    model = character.current_model_name or character.modelfile_base
    content, metadata = await ollama.chat(model, messages, think=think)
    thinking: str = str(metadata.get("thinking", ""))

    await store_message(
        db,
        session_id=session_id,
        segment_id=segment.id,
        character_id=session.character_id,
        role="assistant",
        content=content,
        turn_id=turn_id,
    )

    return content, thinking, turn_id
