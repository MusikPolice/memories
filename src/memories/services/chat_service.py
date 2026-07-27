"""Chat turn orchestration."""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from typing import Any, cast

import aiosqlite

from memories.database import (
    get_character,
    get_facts,
    get_inferences,
    get_messages,
    get_session,
    next_turn_id,
    store_decision,
    store_message,
)
from memories.database import set_facts as db_set_facts
from memories.exceptions import NotFoundError, SessionEndedError
from memories.models import Character, Experience, Inference
from memories.schema_loader import _collect_leaves, check_write_permitted, load_schema
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
from memories.services.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from memories.services.prompt_builder import build_system_prompt
from memories.services.sse_events import EventCallback as EventCallback
from memories.services.sse_events import SSEEvent as SSEEvent
from memories.services.tool_gate import await_gate, cleanup_gate, create_gate
from memories.services.world_builder import run_world_builder

_log = logging.getLogger(__name__)

MAX_CONTRADICTION_RETRIES: int = int(os.getenv("MAX_CONTRADICTION_RETRIES", "3"))

_REQUIRE_FACT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "require_fact",
        "description": (
            "Request a value for an Immutable schema path that has no value yet and "
            "that you cannot produce a coherent response without. Call this INSTEAD "
            "of generating prose when you hit this situation — never invent a value "
            "for an unset Immutable path. The user will confirm, edit, or decline; "
            "you will resume with whatever they decide as the result of this call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Dot-notation schema path, e.g. 'Character.Identity.Name'.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you need this value to respond coherently.",
                },
                "suggested_value": {
                    "type": "string",
                    "description": (
                        "Optional plausible value to pre-fill for the user to confirm or edit."
                    ),
                },
            },
            "required": ["path", "reason"],
        },
    },
}


def _set_leaf(blob: dict[str, Any], path: str, value: str | int | float | bool | None) -> None:
    parts = path.split(".")
    node = blob
    for key in parts[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[parts[-1]] = {"Value": value}


def _lookup_leaf_value(blob: dict[str, Any], path: str) -> str | int | float | bool | None:
    node: Any = blob
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node.get("Value") if isinstance(node, dict) else None


async def run_contradiction_loop(
    db: aiosqlite.Connection,
    session_id: int,
    turn_id: int,
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

    schema = load_schema()
    leaves_by_path = dict(_collect_leaves(schema))

    async def _handle_require_fact(args: dict[str, Any]) -> str:
        nonlocal facts_blob
        path = str(args.get("path", ""))
        reason = str(args.get("reason", ""))
        suggested_value = args.get("suggested_value")

        try:
            mutability = check_write_permitted(path, schema)
        except ValueError as exc:
            return f"Error: {exc}"
        if mutability != "Immutable":
            return (
                f"Error: {path} is {mutability}, not Immutable. require_fact is only "
                "for Immutable paths with no value yet — respond using your best "
                "judgement; the evaluator will record any implied value afterward."
            )
        leaf = leaves_by_path[path]
        current = _lookup_leaf_value(facts_blob, path)
        if current is not None:
            return (
                f"Error: {path} is already set to {current!r}. Use that value "
                "directly instead of calling require_fact."
            )

        if on_event is not None:
            await on_event(
                SSEEvent(
                    event="sidechannel",
                    data={
                        "type": "require_fact",
                        "turn_id": turn_id,
                        "path": path,
                        "reason": reason,
                        "suggested_value": suggested_value,
                    },
                )
            )

        _log.info(
            "require_fact awaiting user input session=%d turn=%d path=%s",
            session_id,
            turn_id,
            path,
        )
        value = await await_gate(session_id, turn_id)
        _log.info(
            "require_fact resolved session=%d turn=%d path=%s provided=%s",
            session_id,
            turn_id,
            path,
            value is not None,
        )

        tool_args: dict[str, Any] = {"path": path, "reason": reason}
        if suggested_value is not None:
            tool_args["suggested_value"] = suggested_value
        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_llm",  # nosec B106
            tool_name="require_fact",
            tool_args=tool_args,
            user_input={"value": value},
        )

        if value is None:
            return (
                f"No value was provided for {path}. Do not invent one — respond "
                "without depending on it, or acknowledge that it is not yet known."
            )

        coerced: str | int | float | bool | None = value
        if leaf["Type"] == "Enum":
            match = next((c for c in leaf["Constraint"] if c.lower() == value.lower()), None)
            if match is None:
                _log.warning(
                    "require_fact: user value %r for %s does not match any Enum "
                    "constraint — storing verbatim",
                    value,
                    path,
                )
            coerced = match if match is not None else value
        elif leaf["Type"] == "Integer":
            try:
                coerced = int(value)
            except ValueError:
                _log.warning(
                    "require_fact: user value %r for %s is not a valid integer — storing verbatim",
                    value,
                    path,
                )

        working_blob = copy.deepcopy(facts_blob)
        _set_leaf(working_blob, path, coerced)
        await db_set_facts(db, character.id, working_blob)
        facts_blob = working_blob

        return f"{path} = {coerced!r}. Use this value now."

    # Work on a private copy so the per-iteration system-prompt rebuild (below) never
    # mutates the caller's list.
    base_messages = list(base_messages)

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
        char_result = await ollama.chat_with_tools(
            model,
            messages,
            [_REQUIRE_FACT_TOOL],
            {"require_fact": _handle_require_fact},
            think=think,
        )
        if char_result.cap_reached:
            _log.warning(
                "character LLM tool-call cap reached on attempt %d — delivering "
                "whatever content was produced (likely empty)",
                attempt + 1,
            )
        content = char_result.content
        thinking = char_result.thinking

        if attempt == 0 and on_event is not None:
            await on_event(SSEEvent(event="status", data={"state": "reviewing"}))
        try:
            ev, facts_blob = await run_evaluator(
                db,
                character,
                session_id,
                turn_id,
                facts_blob,
                user_content,
                content,
                ollama,
                contradiction_hints=contradiction_hints or None,
                inferences=inferences or None,
                experiences=experiences or None,
                on_event=on_event,
            )
        except EvaluatorParseError:
            _log.warning(
                "evaluator parse error on attempt %d — delivering response unverified", attempt + 1
            )
            ev = EvaluatorResult(
                verdict="pass",
                decision_log="(evaluator parse error — response delivered unverified)",
            )

        if ev.verdict != "contradiction" and not ev.needs_regeneration:
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

        # We are regenerating (contradiction or needs_regeneration).  The evaluator may
        # have written fact updates into facts_blob during an edit/accept, so rebuild the
        # system prompt from the current blob before the next Character-LLM attempt —
        # otherwise the regenerated response renders a stale world state.
        if base_messages and base_messages[0].get("role") == "system":
            base_messages[0] = {
                "role": "system",
                "content": build_system_prompt(character, facts_blob, inferences, experiences),
            }

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

    _log.info(
        "turn start session=%d turn=%d think=%s msg=%r",
        session_id,
        turn_id,
        think,
        user_content[:120],
    )

    create_gate(session_id, turn_id)
    try:
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
            _log.info(
                "session=%d turn=%d retrieved %d experience(s)", session_id, turn_id, len(active)
            )

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
            db,
            session_id,
            turn_id,
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

        if eval_result.max_retries_exceeded:
            _log.warning(
                "session=%d turn=%d contradiction retries exhausted — "
                "delivering response unverified",
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
    finally:
        cleanup_gate(session_id, turn_id)
