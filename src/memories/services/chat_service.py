"""Chat turn orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, cast

import aiosqlite

from memories.database import (
    create_inference,
    get_character,
    get_facts,
    get_inferences,
    get_messages,
    get_session,
    next_turn_id,
    store_decision,
    store_message,
)
from memories.exceptions import NotFoundError, SessionEndedError
from memories.models import Character, Experience, Inference
from memories.services.evaluator import (
    ContradictionNotification,
    EvaluatorParseError,
    EvaluatorResult,
    run_evaluator,
)
from memories.services.experience_service import (
    TOP_K_EXPERIENCES,
    add_active_experiences,
    clear_active_experiences,
    retrieve_experiences,
)
from memories.services.inference_service import MAX_INFERENCE_DEPTH, compute_depth
from memories.services.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from memories.services.prompt_builder import build_system_prompt
from memories.services.sse_events import EventCallback as EventCallback
from memories.services.sse_events import SSEEvent as SSEEvent
from memories.services.world_builder import run_world_builder

_log = logging.getLogger(__name__)

MAX_CONTRADICTION_RETRIES: int = int(os.getenv("MAX_CONTRADICTION_RETRIES", "3"))


async def run_contradiction_loop(
    model: str,
    base_messages: list[dict[str, str]],
    character: Character,
    facts_blob: dict[str, Any],
    user_content: str,
    ollama: OllamaClient,
    think: bool = False,
    max_retries: int = MAX_CONTRADICTION_RETRIES,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
    on_event: EventCallback = None,
) -> tuple[str, str, EvaluatorResult]:
    """Run the character LLM + evaluator, retrying until no contradictions remain.

    Returns ``(content, thinking, eval_result)``.
    ``eval_result.contradiction_notifications`` accumulates one entry per contradiction
    found during the loop.  ``eval_result.max_retries_exceeded`` is set if the loop
    exhausted all retries without a clean response.
    """
    contradiction_notifications: list[ContradictionNotification] = []
    contradiction_hints: list[str] = []
    content = ""
    thinking = ""
    eval_result: EvaluatorResult | None = None

    for attempt in range(max_retries + 1):
        messages = list(base_messages)
        if contradiction_hints:
            note = (
                "[SYSTEM NOTE: Your previous response contained a contradiction. "
                + "; ".join(contradiction_hints)
                + ". Please revise your response so it does not contradict any established facts.]"
            )
            messages.append({"role": "user", "content": note})

        if attempt == 0 and on_event is not None:
            await on_event(SSEEvent(event="status", data={"state": "generating"}))
        raw_content, metadata = await ollama.chat(model, messages, think=think)
        content = raw_content
        thinking = str(metadata.get("thinking", ""))

        if attempt == 0 and on_event is not None:
            await on_event(SSEEvent(event="status", data={"state": "reviewing"}))
        try:
            ev = await run_evaluator(
                character,
                facts_blob,
                user_content,
                content,
                ollama,
                contradiction_hints=contradiction_hints or None,
                inferences=inferences or None,
                experiences=experiences or None,
            )
        except EvaluatorParseError:
            _log.warning(
                "evaluator parse error on attempt %d — delivering response unverified", attempt + 1
            )
            ev = EvaluatorResult(
                verdict="pass",
                decision_log="(evaluator parse error — response delivered unverified)",
            )

        if ev.verdict != "contradiction":
            eval_result = ev
            break

        for v in ev.violations:
            if v.type == "contradiction":
                _log.info("contradiction on attempt %d: %s", attempt + 1, v.description)
                contradiction_notifications.append(
                    ContradictionNotification(iteration=attempt + 1, description=v.description)
                )
                contradiction_hints.append(v.description)

        if attempt == max_retries:
            ev.max_retries_exceeded = True
            eval_result = ev
            break

    assert eval_result is not None
    eval_result.contradiction_notifications = contradiction_notifications
    return content, thinking, eval_result


async def run_turn(
    db: aiosqlite.Connection,
    session_id: int,
    user_content: str,
    ollama: OllamaClient,
    think: bool = False,
    on_event: EventCallback = None,
) -> tuple[str, str, int, EvaluatorResult, dict[int, float]]:
    """Execute one conversation turn.

    Returns ``(response_content, thinking_text, turn_id, evaluator_result,
    experience_scores)``.
    The assistant message is stored only after the evaluator confirms the
    response is not a contradiction (or retries are exhausted).
    """
    session = await get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")
    if session.ended_at is not None:
        raise SessionEndedError(f"Session {session_id} has ended")

    # Parallelize all DB reads that depend only on session, not on each other.
    character, facts_blob, inferences, history, turn_id = await asyncio.gather(
        get_character(db, session.character_id),
        get_facts(db, session.character_id),
        get_inferences(db, session.character_id),
        get_messages(db, session_id),
        next_turn_id(db, session_id),
    )
    assert character is not None

    # --- Parallel: experience retrieval (embed) + World Builder (LLM + DB writes) ---
    # Neither depends on the other: embed only needs user_content; the World
    # Builder needs inferences which are already loaded above.
    async def _run_world_builder_safe() -> dict[str, Any]:
        try:
            return await run_world_builder(
                db,
                character,
                session_id,
                turn_id,
                user_content,
                facts_blob,
                inferences,
                ollama,
                on_event=on_event,
            )
        except (OllamaConnectionError, OllamaResponseError) as exc:
            _log.warning("world builder failed: %s", exc)
            return cast(dict[str, Any], facts_blob)

    (active, experience_scores), facts_blob = await asyncio.gather(
        retrieve_experiences(
            db, session.character_id, user_content, ollama, top_k=TOP_K_EXPERIENCES
        ),
        _run_world_builder_safe(),
    )

    # Process experience results
    clear_active_experiences(session_id)
    if active:
        add_active_experiences(session_id, active)
        _log.info("session=%d turn=%d retrieved %d experience(s)", session_id, turn_id, len(active))

    system_prompt = build_system_prompt(character, facts_blob, inferences, active or None)

    await store_message(
        db,
        session_id=session_id,
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

    char_content, char_thinking, eval_result = await run_contradiction_loop(
        model,
        base_messages,
        character,
        facts_blob,
        user_content,
        ollama,
        think=think,
        inferences=inferences,
        experiences=active or None,
        on_event=on_event,
    )

    await store_message(
        db,
        session_id=session_id,
        character_id=session.character_id,
        role="assistant",
        content=char_content,
        turn_id=turn_id,
    )

    # Auto-promote logical inferences with depth cap.
    # Append each stored inference to the snapshot so subsequent depth
    # computations in the same batch see the correct chain depth.
    if eval_result.verdict == "new_inference_logical":
        for inf in eval_result.new_inferences:
            if inf.inference_type != "logical":
                continue
            depth = compute_depth(inf.source_inference_ids, inferences)
            if depth > MAX_INFERENCE_DEPTH:
                continue
            stored = await create_inference(
                db,
                character_id=session.character_id,
                statement=inf.statement,
                derivation=inf.derivation,
                source_inference_ids=inf.source_inference_ids,
                source_fact_paths=inf.source_fact_paths,
                inference_type=inf.inference_type,
                depth=depth,
            )
            inferences.append(stored)

    await store_decision(
        db,
        character_id=session.character_id,
        session_id=session_id,
        turn_id=turn_id,
        pass_name="character_evaluator",  # nosec B106
        tool_name="evaluator_verdict",
        tool_args={"verdict": eval_result.verdict},
        user_input=None,
    )

    if eval_result.max_retries_exceeded:
        _log.warning(
            "session=%d turn=%d contradiction retries exhausted — delivering response unverified",
            session_id,
            turn_id,
        )
    _log.info(
        "session=%d turn=%d verdict=%s violations=%d",
        session_id,
        turn_id,
        eval_result.verdict,
        len(eval_result.violations),
    )
    return char_content, char_thinking, turn_id, eval_result, experience_scores
