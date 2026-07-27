"""Evaluator LLM service.

Drives a tool-calling loop after each character response to check it against
established Facts.  Returns a structured verdict that chat_service uses to
decide whether to deliver, regenerate, or surface a notification.
"""

from __future__ import annotations

import copy
import json
import logging
from typing import Any

import aiosqlite
from pydantic import BaseModel

from memories.database import create_inference, store_decision
from memories.database import set_facts as db_set_facts
from memories.models import Character, Experience, Inference
from memories.schema_loader import (
    _collect_leaves,
    check_write_permitted,
    load_schema,
    render_current_fact_values,
    render_schema_for_prompt,
)
from memories.services.ollama_client import MAX_TOOL_CALL_ROUNDS, OllamaClient, ToolHandler
from memories.services.sse_events import EventCallback, SSEEvent
from memories.services.tool_gate import await_gate

_log = logging.getLogger(__name__)

_TERMINAL_TOOLS = frozenset({"report_pass", "report_contradiction"})


class EvaluatorParseError(Exception):
    """Raised when the evaluator returns unparseable or invalid JSON."""


class NewInference(BaseModel):
    inference_type: str
    statement: str
    derivation: str
    source_fact_paths: list[str] = []
    source_inference_ids: list[int] = []


class Violation(BaseModel):
    type: str
    description: str


class ContradictionNotification(BaseModel):
    iteration: int
    description: str


class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    decision_log: str
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False
    needs_regeneration: bool = False


_SET_FACT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_fact",
        "description": (
            "Record a value the character's response implies for an existing schema "
            "path. The server enforces the path's mutability tier: Fluid values apply "
            "immediately; an Immutable path that is already set returns an error "
            "instructing you to call report_contradiction instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Dot-notation schema path, e.g. 'Character.State-Of-Mind.Mood'.",
                },
                "value": {"type": "string", "description": "The value implied by the response."},
            },
            "required": ["path", "value"],
        },
    },
}

_PROPOSE_INFERENCE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "propose_inference",
        "description": (
            "Record a detail the character's response asserts or implies that has no "
            "matching schema path. This is recorded as the character's belief, written "
            "immediately with no approval step — not a Fact."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "statement": {"type": "string", "description": "The inferred belief."},
                "derivation": {
                    "type": "string",
                    "description": "Brief explanation of how this follows from known facts.",
                },
                "source_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Schema paths this inference was derived from, if any.",
                },
            },
            "required": ["statement", "derivation"],
        },
    },
}

_REPORT_CONTRADICTION_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_contradiction",
        "description": (
            "End the evaluation: the character's response conflicts with an Immutable "
            "fact that is already set. The response will be discarded and regenerated."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "What immutable fact was contradicted and how.",
                }
            },
            "required": ["description"],
        },
    },
}

_REPORT_PASS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_pass",
        "description": "End the evaluation: the response is consistent with all established facts.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_EVALUATOR_TOOLS: list[dict[str, Any]] = [
    _SET_FACT_TOOL,
    _PROPOSE_INFERENCE_TOOL,
    _REPORT_CONTRADICTION_TOOL,
    _REPORT_PASS_TOOL,
]


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


def build_evaluator_prompt(
    character: Character,
    facts_blob: dict[str, Any],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
    """Build the user-facing content for the evaluator tool-calling loop."""
    parts: list[str] = [f"Character: {character.name}"]

    parts.append("")
    parts.append(render_schema_for_prompt())

    parts.append("")
    parts.append(render_current_fact_values(facts_blob))

    parts.append("\n## Established Inferences (id: statement)")
    if inferences:
        for inf in inferences:
            parts.append(f"[{inf.id}] {inf.statement}  (from: {inf.derivation})")
    else:
        parts.append("(no inferences established yet)")

    parts.append("\n## Active Experiences (id: statement  [source])")
    if experiences:
        for exp in experiences:
            parts.append(f"[{exp.id}] {exp.statement}  [{exp.source}]")
    else:
        parts.append("(no active experiences this session)")

    parts.append(f'\n## Conversation Context\nUser said: "{user_message}"')
    parts.append(f"\n## Character Response to Evaluate\n{character_response}")

    if contradiction_hints:
        parts.append("\n## Previously Flagged Contradictions")
        for hint in contradiction_hints:
            parts.append(f"- {hint}")

    parts.append(
        """
## Your Task
Analyze the character's response against the Fact Schema and Current Fact Values above.
You have four tools available:

- set_fact(path, value): the response implies a value for an existing schema path that
  is not already correctly recorded. Call this for every such path.
- propose_inference(statement, derivation, source_paths): the response asserts or
  implies a detail that has NO matching schema path. This records the character's
  belief, not a fact — it is written immediately with no approval step.
- report_contradiction(description): the response conflicts with an IMMUTABLE fact
  that is already set. This ends the evaluation; do not call any other tool after it.
- report_pass(): the response needs no further action beyond whatever set_fact and
  propose_inference calls you already made. This ends the evaluation; do not call any
  other tool after it.

Call set_fact and/or propose_inference as many times as needed — batch them into a
single response when possible — then ALWAYS finish by calling exactly one of
report_contradiction or report_pass. Never respond with plain text instead of a tool
call.

If a set_fact call returns an error because the path is Immutable and already set to a
conflicting value, call report_contradiction with a description of the conflict — do
not retry set_fact for that path."""
    )

    return "\n".join(parts)


async def run_evaluator(
    db: aiosqlite.Connection,
    character: Character,
    session_id: int,
    turn_id: int,
    facts_blob: dict[str, Any],
    user_message: str,
    character_response: str,
    ollama: OllamaClient,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
    on_event: EventCallback = None,
    max_rounds: int = MAX_TOOL_CALL_ROUNDS,
) -> tuple[EvaluatorResult, dict[str, Any]]:
    """Run the evaluator tool-call loop. Returns (result, updated_facts_blob).

    The returned blob reflects any Fluid set_fact writes made during this call;
    callers that retry (e.g. run_contradiction_loop across contradiction attempts)
    must pass the returned blob into the next call so writes from earlier attempts
    are not lost on the next full-blob replace.
    """
    schema = load_schema()
    leaves_by_path = dict(_collect_leaves(schema))
    working_blob: dict[str, Any] = copy.deepcopy(facts_blob)
    _regeneration_needed: list[bool] = [False]

    prompt = build_evaluator_prompt(
        character,
        facts_blob,
        user_message,
        character_response,
        contradiction_hints,
        inferences,
        experiences,
    )
    model = character.current_model_name or character.modelfile_base
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a strict fact-checker for a character roleplay system. "
                "Use the tools provided to record fact updates and inferences, then "
                "conclude with exactly one terminal tool call."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    async def _handle_set_fact(args: dict[str, Any]) -> str:
        path = str(args.get("path", ""))
        value = args.get("value")
        try:
            mutability = check_write_permitted(path, schema)
        except ValueError as exc:
            return f"Error: {exc}"
        leaf = leaves_by_path[path]

        if mutability == "Fluid":
            coerced = value
            if leaf["Type"] == "Enum":
                match = next(
                    (c for c in leaf["Constraint"] if c.lower() == str(value).lower()),
                    None,
                )
                if match is None:
                    return (
                        f"Error: {path}: {value!r} is not a valid value. "
                        f"Valid values: {', '.join(leaf['Constraint'])}"
                    )
                coerced = match
            elif leaf["Type"] == "Integer":
                try:
                    coerced = int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return f"Error: {path}: {value!r} is not a valid integer"
            _set_leaf(working_blob, path, coerced)
            await db_set_facts(db, character.id, working_blob)
            await store_decision(
                db,
                character_id=character.id,
                session_id=session_id,
                turn_id=turn_id,
                pass_name="character_evaluator",  # nosec B106
                tool_name="set_fact",
                tool_args={"path": path, "value": coerced},
                user_input=None,
            )
            if on_event is not None:
                await on_event(
                    SSEEvent(
                        event="sidechannel",
                        data={
                            "type": "fact_update_fluid",
                            "turn_id": turn_id,
                            "path": path,
                            "value": coerced,
                        },
                    )
                )
            _log.info("evaluator set_fact (fluid) turn=%d %s=%r", turn_id, path, coerced)
            return f"Wrote {path} = {coerced!r}."

        if mutability == "Immutable":
            current = _lookup_leaf_value(working_blob, path)
            if current is not None:
                return (
                    f"Error: {path} is Immutable and already set to {current!r}. "
                    "You may not change it. If the character's response conflicts with "
                    "this, call report_contradiction instead."
                )
            # Immutable, unset — validate the proposed value before suspending
            if leaf["Type"] == "Enum":
                proposed_match = next(
                    (c for c in leaf["Constraint"] if c.lower() == str(value).lower()),
                    None,
                )
                if proposed_match is None:
                    return (
                        f"Error: {path}: {value!r} is not a valid value. "
                        f"Valid values: {', '.join(leaf['Constraint'])}"
                    )
                proposed: str | int | float | bool | None = proposed_match
            elif leaf["Type"] == "Integer":
                try:
                    proposed = int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return f"Error: {path}: {value!r} is not a valid integer"
            else:
                proposed = value

            if on_event is not None:
                await on_event(
                    SSEEvent(
                        event="sidechannel",
                        data={
                            "type": "fact_update_immutable_unset",
                            "turn_id": turn_id,
                            "path": path,
                            "proposed": proposed,
                        },
                    )
                )
            _log.info(
                "evaluator set_fact approval surfaced turn=%d path=%s tier=Immutable",
                turn_id,
                path,
            )
            raw = await await_gate(session_id, turn_id)
            decision: dict[str, str | None] = (
                json.loads(raw) if raw is not None else {"action": "dismiss"}
            )
            action = decision.get("action", "dismiss")
            _log.info(
                "evaluator set_fact resolved turn=%d path=%s action=%s value=%r",
                turn_id,
                path,
                action,
                decision.get("value"),
            )
            await store_decision(
                db,
                character_id=character.id,
                session_id=session_id,
                turn_id=turn_id,
                pass_name="character_evaluator",  # nosec B106
                tool_name="set_fact",
                tool_args={"path": path, "value": str(value)},
                user_input={"action": action, "value": decision.get("value")},
            )
            if action == "accept":
                _set_leaf(working_blob, path, proposed)
                await db_set_facts(db, character.id, working_blob)
                return f"Wrote {path} = {proposed!r}. Value is now locked immutably."
            if action == "edit":
                user_val_raw = str(decision.get("value") or "")
                if leaf["Type"] == "Enum":
                    em = next(
                        (c for c in leaf["Constraint"] if c.lower() == user_val_raw.lower()),
                        None,
                    )
                    if em is None:
                        _log.warning(
                            "set_fact approval: user edit %r for %s has no Enum match "
                            "— storing verbatim",
                            user_val_raw,
                            path,
                        )
                    user_val: str | int | float | bool | None = (
                        em if em is not None else user_val_raw
                    )
                elif leaf["Type"] == "Integer":
                    try:
                        user_val = int(user_val_raw)
                    except ValueError:
                        _log.warning(
                            "set_fact approval: user edit %r for %s is not a valid "
                            "integer — storing verbatim",
                            user_val_raw,
                            path,
                        )
                        user_val = user_val_raw
                else:
                    user_val = user_val_raw
                _set_leaf(working_blob, path, user_val)
                await db_set_facts(db, character.id, working_blob)
                _regeneration_needed[0] = True
                return f"Wrote {path} = {user_val!r}. Response will be regenerated with this value."
            # dismiss
            return f"No value recorded for {path}. Do not rely on the invented value."

        # Mutable — validate the proposed value before suspending
        if leaf["Type"] == "Enum":
            proposed_match = next(
                (c for c in leaf["Constraint"] if c.lower() == str(value).lower()),
                None,
            )
            if proposed_match is None:
                return (
                    f"Error: {path}: {value!r} is not a valid value. "
                    f"Valid values: {', '.join(leaf['Constraint'])}"
                )
            proposed = proposed_match
        elif leaf["Type"] == "Integer":
            try:
                proposed = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return f"Error: {path}: {value!r} is not a valid integer"
        else:
            proposed = value

        if on_event is not None:
            await on_event(
                SSEEvent(
                    event="sidechannel",
                    data={
                        "type": "fact_update_mutable",
                        "turn_id": turn_id,
                        "path": path,
                        "proposed": proposed,
                    },
                )
            )
        _log.info(
            "evaluator set_fact approval surfaced turn=%d path=%s tier=Mutable",
            turn_id,
            path,
        )
        raw = await await_gate(session_id, turn_id)
        decision = json.loads(raw) if raw is not None else {"action": "reject"}
        action = decision.get("action", "reject")
        _log.info(
            "evaluator set_fact resolved turn=%d path=%s action=%s value=%r",
            turn_id,
            path,
            action,
            decision.get("value"),
        )
        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_evaluator",  # nosec B106
            tool_name="set_fact",
            tool_args={"path": path, "value": str(value)},
            user_input={"action": action, "value": decision.get("value")},
        )
        if action == "accept":
            _set_leaf(working_blob, path, proposed)
            await db_set_facts(db, character.id, working_blob)
            return f"Wrote {path} = {proposed!r}."
        if action == "edit":
            user_val_raw = str(decision.get("value") or "")
            if leaf["Type"] == "Enum":
                em = next(
                    (c for c in leaf["Constraint"] if c.lower() == user_val_raw.lower()),
                    None,
                )
                if em is None:
                    _log.warning(
                        "set_fact approval: user edit %r for %s has no Enum match "
                        "— storing verbatim",
                        user_val_raw,
                        path,
                    )
                user_val = em if em is not None else user_val_raw
            elif leaf["Type"] == "Integer":
                try:
                    user_val = int(user_val_raw)
                except ValueError:
                    _log.warning(
                        "set_fact approval: user edit %r for %s is not a valid "
                        "integer — storing verbatim",
                        user_val_raw,
                        path,
                    )
                    user_val = user_val_raw
            else:
                user_val = user_val_raw
            _set_leaf(working_blob, path, user_val)
            await db_set_facts(db, character.id, working_blob)
            _regeneration_needed[0] = True
            return f"Wrote {path} = {user_val!r}. Response will be regenerated with this value."
        # reject
        _regeneration_needed[0] = True
        return "Change rejected. Response will be regenerated without this update."

    async def _handle_propose_inference(args: dict[str, Any]) -> str:
        statement = str(args.get("statement", ""))
        derivation = str(args.get("derivation", ""))
        source_paths = [str(p) for p in (args.get("source_paths") or [])]
        stored = await create_inference(
            db,
            character_id=character.id,
            statement=statement,
            derivation=derivation,
            source_fact_paths=source_paths,
            source_inference_ids=[],
            inference_type="logical",
            depth=1,
        )
        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_evaluator",  # nosec B106
            tool_name="propose_inference",
            tool_args={
                "statement": statement,
                "derivation": derivation,
                "source_paths": source_paths,
            },
            user_input=None,
        )
        _log.info(
            "evaluator propose_inference turn=%d id=%d stmt=%r",
            turn_id,
            stored.id,
            statement[:120],
        )
        if on_event is not None:
            await on_event(
                SSEEvent(
                    event="sidechannel",
                    data={
                        "type": "inference_proposed",
                        "turn_id": turn_id,
                        "inference": stored.model_dump(mode="json"),
                    },
                )
            )
        return f"Recorded inference: {statement!r}"

    async def _handle_report_contradiction(args: dict[str, Any]) -> str:
        description = str(args.get("description", ""))
        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_evaluator",  # nosec B106
            tool_name="report_contradiction",
            tool_args={"description": description},
            user_input=None,
        )
        _log.info("evaluator report_contradiction turn=%d: %s", turn_id, description)
        return "Contradiction recorded."

    async def _handle_report_pass(_args: dict[str, Any]) -> str:
        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_evaluator",  # nosec B106
            tool_name="report_pass",
            tool_args={},
            user_input=None,
        )
        return "Pass recorded."

    handlers: dict[str, ToolHandler] = {
        "set_fact": _handle_set_fact,
        "propose_inference": _handle_propose_inference,
        "report_contradiction": _handle_report_contradiction,
        "report_pass": _handle_report_pass,
    }

    result = await ollama.chat_with_tools(
        model,
        messages,
        _EVALUATOR_TOOLS,
        handlers,
        max_rounds=max_rounds,
        terminal_tools=_TERMINAL_TOOLS,
    )

    if result.terminal_call is None:
        # Exhausted max_rounds without a terminal tool — give one more round with an
        # explicit instruction before giving up.
        nudged_history = list(result.history)
        nudged_history.append(
            {
                "role": "system",
                "content": (
                    "You must now call report_pass() or report_contradiction(description) "
                    "to conclude this evaluation. No other tool call will be processed."
                ),
            }
        )
        result = await ollama.chat_with_tools(
            model,
            nudged_history,
            _EVALUATOR_TOOLS,
            handlers,
            max_rounds=1,
            terminal_tools=_TERMINAL_TOOLS,
        )

    if result.terminal_call is None:
        _log.warning(
            "evaluator tool-call cap reached with no terminal tool called — "
            "delivering response as a pass"
        )
        return (
            EvaluatorResult(
                verdict="pass",
                decision_log="(tool-call cap reached — response delivered unverified)",
                needs_regeneration=_regeneration_needed[0],
            ),
            working_blob,
        )

    name = result.terminal_call["name"]
    if name == "report_contradiction":
        description = str(result.terminal_call["arguments"].get("description", ""))
        return (
            EvaluatorResult(
                verdict="contradiction",
                violations=[Violation(type="contradiction", description=description)],
                decision_log=description,
                needs_regeneration=_regeneration_needed[0],
            ),
            working_blob,
        )

    return (
        EvaluatorResult(
            verdict="pass",
            decision_log="Response is consistent with established facts.",
            needs_regeneration=_regeneration_needed[0],
        ),
        working_blob,
    )
