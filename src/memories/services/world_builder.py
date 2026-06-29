"""World Builder service — Step 4.

Runs before the Character LLM on every turn. Extracts facts the user's message
states or implies and writes them directly to the character_facts blob via a
single author_set_facts tool call. The user is the author of the story: any
value written here becomes ground truth immediately, regardless of the
schema path's mutability tier or any value already stored.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import aiosqlite

from memories.database import set_facts as db_set_facts
from memories.database import store_decision
from memories.models import Character, Inference
from memories.schema_loader import (
    _collect_leaves,
    load_schema,
    render_current_fact_values,
    render_schema_for_prompt,
)
from memories.services.ollama_client import MAX_TOOL_CALL_ROUNDS, OllamaClient
from memories.services.sse_events import EventCallback, SSEEvent

_log = logging.getLogger(__name__)

_AUTHOR_SET_FACTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "author_set_facts",
        "description": (
            "Record one or more facts the user has stated or clearly implied "
            "about the world. You are the author's hand: values written here "
            "become ground truth immediately, regardless of any existing "
            "value or the path's mutability tier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "description": "The facts to record, one entry per fact.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": (
                                    "Dot-notation schema path, e.g. "
                                    "'Character.State-Of-Mind.Mood'."
                                ),
                            },
                            "value": {
                                "type": "string",
                                "description": "The value to write at this path.",
                            },
                        },
                        "required": ["path", "value"],
                    },
                }
            },
            "required": ["facts"],
        },
    },
}


def build_world_builder_prompt(
    user_message: str,
    character: Character,
    facts_blob: dict[str, Any],
    inferences: list[Inference] | None = None,
) -> str:
    """Build the user-facing content for the World Builder Ollama call."""
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

    parts.append(f'\n## User Message\n"{user_message}"')

    parts.append(
        """
## Your Task
You are the World Builder. The user is the author of this story: anything they
state or clearly imply about the world is ground truth, regardless of any
mutability tier or value already on record. Call author_set_facts with one
entry per fact implied by their message — both explicit statements ("My name
is Jon") and strong implications ("I crossed the room and kissed her" implies
Setting.Location.Space is Interior) qualify.

Only write to paths listed in the Fact Schema above. You may NOT invent new
paths. Batch every fact you find into a SINGLE call to author_set_facts — do
not call it more than once.

If nothing in the message implies any fact, call no tools and reply with a
brief acknowledgement."""
    )

    return "\n".join(parts)


def _set_leaf(blob: dict[str, Any], path: str, value: str | int | float | bool | None) -> None:
    parts = path.split(".")
    node = blob
    for key in parts[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[parts[-1]] = {"Value": value}


async def run_world_builder(
    db: aiosqlite.Connection,
    character: Character,
    session_id: int,
    turn_id: int,
    user_message: str,
    facts_blob: dict[str, Any],
    inferences: list[Inference],
    ollama: OllamaClient,
    on_event: EventCallback = None,
    max_rounds: int = MAX_TOOL_CALL_ROUNDS,
) -> dict[str, Any]:
    """Run the World Builder pass. Returns the updated facts blob.

    Writes and decision-logs each successful author_set_facts call as it
    happens (not deferred to a result object the caller processes later), and
    emits one `world_builder_applied` sidechannel event per call that writes
    at least one fact. Returns the accumulated blob so the caller can use it
    immediately for the system prompt — no extra get_facts() round trip.
    """
    schema = load_schema()
    leaves_by_path = dict(_collect_leaves(schema))
    working_blob: dict[str, Any] = copy.deepcopy(facts_blob)

    prompt = build_world_builder_prompt(user_message, character, facts_blob, inferences)
    model = character.current_model_name or character.modelfile_base
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are the World Builder for a character roleplay system. "
                "Extract facts from the user's message using the tool provided."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    async def _handle_author_set_facts(args: dict[str, Any]) -> str:
        entries: list[dict[str, Any]] = args.get("facts", []) or []
        written: list[dict[str, Any]] = []
        errors: list[str] = []

        for entry in entries:
            path = entry.get("path", "")
            value: Any = entry.get("value")
            leaf = leaves_by_path.get(path)
            if leaf is None:
                errors.append(f"{path}: unknown schema path")
                continue
            if leaf["Type"] == "Enum":
                match = next(
                    (c for c in leaf["Constraint"] if c.lower() == str(value).lower()),
                    None,
                )
                if match is None:
                    errors.append(
                        f"{path}: {value!r} is not a valid value. "
                        f"Valid values: {', '.join(leaf['Constraint'])}"
                    )
                    continue
                value = match
            elif leaf["Type"] == "Integer":
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    errors.append(f"{path}: {value!r} is not a valid integer")
                    continue
            _set_leaf(working_blob, path, value)
            written.append({"path": path, "value": value})

        if written:
            await db_set_facts(db, character.id, working_blob)
            await store_decision(
                db,
                character_id=character.id,
                session_id=session_id,
                turn_id=turn_id,
                pass_name="world_builder",  # nosec B106
                tool_name="author_set_facts",
                tool_args={"facts": entries},
                user_input=None,
            )
            if on_event is not None:
                await on_event(
                    SSEEvent(
                        event="sidechannel",
                        data={
                            "type": "world_builder_applied",
                            "turn_id": turn_id,
                            "facts": written,
                        },
                    )
                )

        result_parts: list[str] = []
        if written:
            written_desc = ", ".join(f"{w['path']}={w['value']!r}" for w in written)
            result_parts.append(f"Wrote {len(written)} fact(s): {written_desc}")
        if errors:
            result_parts.append(f"{len(errors)} error(s): {'; '.join(errors)}")
        return " ".join(result_parts) if result_parts else "No facts written."

    result = await ollama.chat_with_tools(
        model,
        messages,
        [_AUTHOR_SET_FACTS_TOOL],
        {"author_set_facts": _handle_author_set_facts},
        max_rounds=max_rounds,
    )
    if result.cap_reached:
        _log.warning(
            "world builder tool-call cap reached after %d rounds — "
            "proceeding with facts written so far",
            max_rounds,
        )

    return working_blob
