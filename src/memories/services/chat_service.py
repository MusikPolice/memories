"""Chat turn orchestration."""

from __future__ import annotations

import os
from typing import Any

import aiosqlite

from memories.database import (
    create_inference,
    get_active_segment,
    get_character,
    get_facts,
    get_messages,
    get_session,
    next_turn_id,
    store_decision,
    store_message,
)
from memories.exceptions import NotFoundError, SessionEndedError
from memories.services.evaluator import (
    ContradictionNotification,
    EvaluatorParseError,
    EvaluatorResult,
    run_evaluator,
)
from memories.services.ollama_client import OllamaClient
from memories.services.prompt_builder import build_system_prompt

MAX_CONTRADICTION_RETRIES: int = int(os.getenv("MAX_CONTRADICTION_RETRIES", "3"))


async def run_turn(
    db: aiosqlite.Connection,
    session_id: int,
    user_content: str,
    ollama: OllamaClient,
    think: bool = False,
) -> tuple[str, str, int, EvaluatorResult]:
    """Execute one conversation turn.

    Returns ``(response_content, thinking_text, turn_id, evaluator_result)``.
    The assistant message is stored only after the evaluator confirms the
    response is not a contradiction (or retries are exhausted).
    """
    session = await get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")
    if session.ended_at is not None:
        raise SessionEndedError(f"Session {session_id} has ended")

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

    base_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        base_messages.append({"role": msg.role, "content": msg.content})
    base_messages.append({"role": "user", "content": user_content})

    model = character.current_model_name or character.modelfile_base

    # --- contradiction loop ---
    contradiction_notifications: list[ContradictionNotification] = []
    contradiction_hints: list[str] = []
    char_content = ""
    char_thinking = ""
    eval_result: EvaluatorResult | None = None

    for attempt in range(MAX_CONTRADICTION_RETRIES + 1):
        # Build messages for this attempt (append system note on retry)
        char_messages = list(base_messages)
        if contradiction_hints:
            note = (
                "[SYSTEM NOTE: Your previous response contained a contradiction. "
                + "; ".join(contradiction_hints)
                + ". Please revise your response so it does not contradict any established facts.]"
            )
            char_messages.append({"role": "user", "content": note})

        raw_content, metadata = await ollama.chat(model, char_messages, think=think)
        char_content = raw_content
        char_thinking = str(metadata.get("thinking", ""))

        try:
            ev = await run_evaluator(
                character,
                facts,
                user_content,
                char_content,
                ollama,
                contradiction_hints=contradiction_hints or None,
            )
        except EvaluatorParseError:
            ev = EvaluatorResult(
                verdict="pass",
                decision_log="(evaluator parse error — response delivered unverified)",
            )

        if ev.verdict != "contradiction":
            eval_result = ev
            break

        # Collect contradiction notifications
        for v in ev.violations:
            if v.type == "contradiction":
                contradiction_notifications.append(
                    ContradictionNotification(iteration=attempt + 1, description=v.description)
                )
                contradiction_hints.append(v.description)

        if attempt == MAX_CONTRADICTION_RETRIES:
            ev.max_retries_exceeded = True
            eval_result = ev
            break

    assert eval_result is not None
    eval_result.contradiction_notifications = contradiction_notifications

    # Determine ungrounded_implications to store with the assistant message
    ungrounded: list[dict[str, Any]] | None = None
    if eval_result.verdict in ("implication", "new_inference_probabilistic"):
        ungrounded = [v.model_dump() for v in eval_result.violations]

    await store_message(
        db,
        session_id=session_id,
        segment_id=segment.id,
        character_id=session.character_id,
        role="assistant",
        content=char_content,
        turn_id=turn_id,
        ungrounded_implications=ungrounded,
    )

    # Auto-promote logical inferences
    if eval_result.verdict == "new_inference_logical":
        for inf in eval_result.new_inferences:
            await create_inference(
                db,
                character_id=session.character_id,
                statement=inf.statement,
                derivation=inf.derivation,
                source_fact_ids=inf.source_fact_ids,
                source_inference_ids=inf.source_inference_ids,
                inference_type=inf.inference_type,
            )

    # Log decision
    violations_for_log = (
        [v.model_dump() for v in eval_result.violations] if eval_result.violations else None
    )
    await store_decision(
        db,
        character_id=session.character_id,
        session_id=session_id,
        turn_id=turn_id,
        reasoning=eval_result.decision_log,
        verdict=eval_result.verdict,
        violations=violations_for_log,
    )

    return char_content, char_thinking, turn_id, eval_result
