"""Chat turn orchestration."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import aiosqlite

from memories.database import (
    create_fact,
    create_inference,
    delete_experience,
    get_active_segment,
    get_character,
    get_facts,
    get_inferences,
    get_messages,
    get_session,
    next_turn_id,
    store_decision,
    store_message,
    update_fact,
)
from memories.exceptions import NotFoundError, SessionEndedError
from memories.models import Character, Experience, Fact, Inference
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
    remove_active_experience,
    retrieve_experiences,
)
from memories.services.extraction_service import (
    ExtractionParseError,
    ExtractionResult,
    run_fact_extractor,
)
from memories.services.inference_service import MAX_INFERENCE_DEPTH, compute_depth
from memories.services.ollama_client import OllamaClient, OllamaConnectionError
from memories.services.prompt_builder import build_system_prompt

_log = logging.getLogger(__name__)

MAX_CONTRADICTION_RETRIES: int = int(os.getenv("MAX_CONTRADICTION_RETRIES", "3"))

StatusCallback = Callable[[str], Awaitable[None]] | None


async def run_contradiction_loop(
    model: str,
    base_messages: list[dict[str, str]],
    character: Character,
    facts: list[Fact],
    user_content: str,
    ollama: OllamaClient,
    think: bool = False,
    max_retries: int = MAX_CONTRADICTION_RETRIES,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
    on_status: StatusCallback = None,
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

        if attempt == 0 and on_status is not None:
            await on_status("generating")
        raw_content, metadata = await ollama.chat(model, messages, think=think)
        content = raw_content
        thinking = str(metadata.get("thinking", ""))

        if attempt == 0 and on_status is not None:
            await on_status("reviewing")
        try:
            ev = await run_evaluator(
                character,
                facts,
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
    on_status: StatusCallback = None,
) -> tuple[str, str, int, EvaluatorResult, dict[int, float], ExtractionResult]:
    """Execute one conversation turn.

    Returns ``(response_content, thinking_text, turn_id, evaluator_result,
    experience_scores, extraction_result)``.
    The assistant message is stored only after the evaluator confirms the
    response is not a contradiction (or retries are exhausted).
    """
    session = await get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Session {session_id} not found")
    if session.ended_at is not None:
        raise SessionEndedError(f"Session {session_id} has ended")

    # Parallelize all DB reads that depend only on session, not on each other.
    # history and segment are loaded here rather than after extraction so they
    # share the same gather pass; both are read-only and unaffected by extraction writes.
    character, facts, inferences, history, segment, turn_id = await asyncio.gather(
        get_character(db, session.character_id),
        get_facts(db, session.character_id),
        get_inferences(db, session.character_id),
        get_messages(db, session_id),
        get_active_segment(db, session_id),
        next_turn_id(db, session_id),
    )
    assert character is not None

    # --- Parallel: experience retrieval (embed) + fact extraction (LLM) ---
    # Neither depends on the other: embed only needs user_content; extraction
    # needs facts/inferences which are already loaded above.
    async def _run_extraction_safe() -> ExtractionResult:
        try:
            return await run_fact_extractor(user_content, character, facts, inferences, ollama)
        except (ExtractionParseError, OllamaConnectionError) as exc:
            _log.warning("fact extraction failed: %s", exc)
            return ExtractionResult()

    (active, experience_scores), extraction_result = await asyncio.gather(
        retrieve_experiences(
            db, session.character_id, user_content, ollama, top_k=TOP_K_EXPERIENCES
        ),
        _run_extraction_safe(),
    )

    # Process experience results
    clear_active_experiences(session_id)
    if active:
        add_active_experiences(session_id, active)
        _log.info("session=%d turn=%d retrieved %d experience(s)", session_id, turn_id, len(active))

    # Process extraction results (DB writes happen sequentially after gather)
    for extracted in extraction_result.new_facts:
        try:
            created = await create_fact(
                db,
                character_id=session.character_id,
                key=extracted.key,
                value=extracted.value,
                category=extracted.category,
                mutability=extracted.mutability,
            )
            extraction_result.applied_fact_ids[extracted.key] = created.id
        except aiosqlite.IntegrityError:
            _log.debug("extraction tier1: skipping duplicate fact key=%s", extracted.key)
    for fact_upd in extraction_result.fact_updates:
        try:
            await update_fact(db, fact_id=fact_upd.fact_id, value=fact_upd.new_value)
        except NotFoundError:
            _log.warning(
                "extraction tier2: fact_id=%d not found, skipping update", fact_upd.fact_id
            )
    if extraction_result.new_facts or extraction_result.fact_updates:
        facts = await get_facts(db, session.character_id)

    system_prompt = build_system_prompt(character, facts, inferences, active or None)

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

    char_content, char_thinking, eval_result = await run_contradiction_loop(
        model,
        base_messages,
        character,
        facts,
        user_content,
        ollama,
        think=think,
        inferences=inferences,
        experiences=active or None,
        on_status=on_status,
    )

    # Handle experience_update verdict: delete contradicted experiences
    if eval_result.verdict == "experience_update":
        if not eval_result.experience_updates:
            _log.warning(
                "session=%d experience_update verdict returned no experience_updates",
                session_id,
            )
        for upd in eval_result.experience_updates:
            try:
                await delete_experience(db, upd.contradicted_experience_id)
                remove_active_experience(session_id, upd.contradicted_experience_id)
                _log.info(
                    "session=%d deleted contradicted experience %d",
                    session_id,
                    upd.contradicted_experience_id,
                )
            except NotFoundError:
                _log.warning(
                    "experience_update referenced unknown experience %d",
                    upd.contradicted_experience_id,
                )

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

    # Auto-promote logical inferences with depth cap.
    # Runs for both "new_inference_logical" and "experience_update" (orthogonal signals).
    # Only logical inferences are auto-promoted; probabilistic ones require user review
    # and are silently discarded when they appear alongside experience_update.
    # Append each stored inference to the snapshot so subsequent depth
    # computations in the same batch see the correct chain depth.
    if eval_result.verdict in ("new_inference_logical", "experience_update"):
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
                source_fact_ids=inf.source_fact_ids,
                source_inference_ids=inf.source_inference_ids,
                inference_type=inf.inference_type,
                depth=depth,
            )
            inferences.append(stored)

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
    return char_content, char_thinking, turn_id, eval_result, experience_scores, extraction_result
