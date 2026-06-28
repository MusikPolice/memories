# Step 4 — World Builder Service

## Overview

Step 4 replaces the Phase 6 fact extractor (`extraction_service.py`) with the World
Builder — the first of the three per-turn passes described in `docs/plan-v2.md`. Where
the old extractor classified user prose into four tiers and produced a structured JSON
verdict that `chat_service.py` then wrote into the legacy `facts` table, the World
Builder expresses every fact it finds as a single tool call, `author_set_facts`, against
the schema-constrained `character_facts` blob that Steps 1–3 built. There is no tier
system, no JSON verdict, and no per-fact approval step: the user is the author of the
story, so anything their message states or clearly implies about the world becomes
ground truth immediately, regardless of the path's mutability or any value already
stored.

This step's blast radius is larger than Steps 1–3 because the World Builder's Ollama
call uses `chat_with_tools()` (non-streaming, tool-calling) instead of `chat()`
(streaming, JSON-verdict) — a different request/response shape than every existing
test in `test_chat_service.py` and `test_api_chat.py` was written against. Most of those
tests mock three sequential `/api/chat` calls (extractor, character, evaluator) via a
shared `_mock_turn()` helper; this step changes what calls[0] looks like everywhere that
helper is used, and deletes the block of tests that specifically exercised the old
four-tier extraction behaviour (Tier 1–4 facts, `extraction_applied`,
`implicit_fact_proposed`).

When Step 4 is complete, `run_turn()` invokes the World Builder before the Character
LLM on every turn. Facts the user's message implies are written directly to
`character_facts` and are visible in the character's system prompt on the same turn. The
Character LLM and Character Evaluator are otherwise unchanged from Step 3 — the
Evaluator still returns a single JSON verdict (`pass` / `contradiction` /
`new_inference_logical` / `new_inference_probabilistic`); its tool-call rewrite is
Step 5. The legacy `facts` table, the `facts.py` router, and the `Fact` model are
untouched — they back an unrelated manual fact-editing UI that has nothing to do with
extraction.

**Success criterion:** all tests in `tests/unit/test_world_builder.py`,
`tests/unit/test_schema_loader.py`, `tests/unit/test_chat_service.py`, and
`tests/integration/test_api_chat.py` pass, along with every other existing test
(`test_api_implication.py`, `test_api_decisions.py`, etc.) after their mechanical mock
updates. The dev server starts cleanly and a message containing an implied fact (e.g.
"I'm freezing, it must be snowing out") results in `Setting.Location.Weather` being set
in `character_facts` and visible in the next system prompt.

---

## What Steps 1–3 Delivered

- `src/memories/fact_schema.json` + `schema_loader.py` — `load_schema()`,
  `apply_mask()`, `check_write_permitted(path, schema) -> mutability str` (raises
  `ValueError` for unknown/grouping paths), `render_schema_for_prompt()`,
  `_collect_leaves(schema, prefix="") -> list[tuple[path, leaf]]` (private but already
  imported cross-module by `evaluator.py`).
- `database.py` — `character_facts` table; `get_facts(db, character_id) -> dict`
  (schema-masked, returns `{}` if no row); `set_facts(db, character_id, blob) -> None`
  (full-blob replace); `patch_fact(db, character_id, path_tuple, value) -> None`
  (single-leaf read-modify-write, used by the *old* per-leaf call sites — Step 4 does
  not use it, see below). `decisions` table has `pass_name`, `tool_name`, `tool_args`
  (dict, JSON column), `user_input` (dict or `None`); `store_decision()` matches.
  `get_inferences()`, `create_inference()` (with `source_fact_paths: list[str]`).
- `services/ollama_client.py` — `chat_with_tools(model, messages, tools,
  tool_handlers, max_rounds=MAX_TOOL_CALL_ROUNDS) -> ToolCallResult`. Drives a
  non-streaming tool-call loop: re-invokes the model after each round of tool calls
  until it returns plain content or `max_rounds` is exhausted. `ToolHandler =
  Callable[[dict[str, Any]], Awaitable[str]]` — async, receives the parsed `arguments`
  dict, returns the string to send back as the tool result; any exception raised is
  caught internally and turned into an `"Error: {exc}"` tool result (the model sees the
  error and can retry — handlers do not need their own top-level try/except for this).
  `ToolCallResult.cap_reached: bool` signals the loop was cut short.
  `MAX_TOOL_CALL_ROUNDS` (module constant, env-overridable, default 10).
- `services/prompt_builder.py` — `build_system_prompt(character, facts_blob,
  inferences=None, experiences=None)` renders the full schema tree; reads `facts_blob`
  fresh on every call, so any blob written before this call is invoked is visible to the
  character.
- `services/evaluator.py` — `build_evaluator_prompt(character, facts_blob, ...)`,
  `run_evaluator(...)`. Still single-shot JSON-verdict based (`ollama.chat(...,
  format="json")`), not yet tool-call based — that conversion is Step 5. Inline in
  `build_evaluator_prompt` (lines 75–102) is a "Current Fact Values" rendering block
  that walks every schema leaf via `_collect_leaves()` and looks up its value in the
  blob — this step extracts that block into a shared `schema_loader` helper (see Part
  B below) because the World Builder needs the identical rendering.
- `services/chat_service.py` — `run_turn()` loads `character, facts_blob, inferences,
  history, turn_id` via one `asyncio.gather`, then runs experience retrieval and fact
  extraction concurrently via a second `asyncio.gather`, then builds the system prompt,
  then runs the Character LLM + Evaluator pair via `run_contradiction_loop()`.
  `SSEEvent` (dataclass: `event: str`, `data: dict`, `.to_sse()`) and `EventCallback =
  Callable[[SSEEvent], Awaitable[None]] | None` are defined directly in this module and
  re-exported; `routers/chat.py` and `routers/test_poc.py` both import `SSEEvent` from
  `chat_service`, not from a dedicated module — Step 4 must preserve that import path
  (see Part A below, this step relocates the *definition* but keeps the *re-export*).

---

## What This Step Does NOT Change

- **The legacy `facts` table, `Fact` model, and `facts.py` router.** These back the
  manual fact CRUD UI (`GET/POST/PUT/DELETE /api/characters/{id}/facts`), which is
  independent of extraction. `create_fact`, `update_fact`, `get_fact_rows`,
  `get_fact`, `delete_fact`, `patch_fact_row`, `get_fact_by_category_key` in
  `database.py` are untouched.
- **The Character Evaluator.** `evaluator.py`'s JSON-verdict loop, `_VALID_VERDICTS`,
  and `EvaluatorResult` are unchanged except for the one extraction noted in Part B
  (moving the "Current Fact Values" rendering into `schema_loader.py` — behaviourally
  identical output, just relocated). The Evaluator's tool-call rewrite is Step 5.
- **The Character LLM call.** `run_contradiction_loop()` still calls plain
  `ollama.chat(model, messages, think=think)` with no `tools` argument — the Character
  LLM has no tool list at all yet. `require_fact` is added in Step 6.
- **The frontend.** `chat.js`'s `buildNotificationFromSidechannel()` already returns
  `null` for any `payload.type` it does not recognise, so introducing a new
  `world_builder_applied` sidechannel type (see Part C) is safe to ship without a
  frontend change — the event is simply not rendered yet. The existing
  `extraction_applied` / `implicit_fact_proposed` cases in `chat.js` (lines 102–120),
  their `index.html` cards, and their handlers in `chat-component.js`
  (`deleteExtractedFact`, `acceptImplicitFact`, `ignoreImplicitFact`) become dead code
  once `chat_service.py` stops producing those payloads — they are **not removed in
  this step**. Cleanup is Step 8's job, consistent with how Step 3 left the
  `implication` dead-code branch in place.
- **`docs/streaming-plan.md`, `tool_gate.py`, `require_fact.py`, `test_poc.py`.** No
  Step 6 prerequisite work happens here.
- **`CLAUDE.md` / `README.md`.** Documentation update is Step 10, done once
  implementation is complete and verified.
- **Mutability enforcement of any kind.** The World Builder is explicitly the one pass
  with unrestricted write authority. `check_write_permitted()`'s mutability return
  value is irrelevant to this step; the World Builder does not call it.

---

## Detailed Design

### Part A — `services/sse_events.py` (new file) — resolving a circular import

The World Builder's `author_set_facts` handler must emit a sidechannel notification via
the same `on_event: EventCallback` mechanism `run_contradiction_loop()` already uses.
`SSEEvent`/`EventCallback` currently live in `chat_service.py`, but `chat_service.py`
will need to import `run_world_builder` from the new `world_builder.py` — so
`world_builder.py` cannot import `SSEEvent` from `chat_service.py` without creating a
cycle. Move the two definitions into a new leaf module:

```python
# src/memories/services/sse_events.py
"""Shared SSE event type for tool handlers that need to emit live notifications."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    event: str
    data: dict[str, object] = field(default_factory=dict)

    def to_sse(self) -> str:
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"


EventCallback = Callable[[SSEEvent], Awaitable[None]] | None
```

In `chat_service.py`, delete the inline `SSEEvent` class and `EventCallback` type alias
(currently lines 56–68) and replace with:

```python
from memories.services.sse_events import EventCallback, SSEEvent
```

Because `chat_service.py` still has `SSEEvent` and `EventCallback` as names in its own
namespace (imported, not defined), `routers/chat.py`'s `from memories.services.chat_service
import SSEEvent, run_turn` and `routers/test_poc.py`'s `from memories.services.chat_service
import SSEEvent` continue to work unchanged — **no edit to either router file is
needed.**

`world_builder.py` imports `from memories.services.sse_events import EventCallback,
SSEEvent` directly.

---

### Part B — `schema_loader.py` — extract `render_current_fact_values()`

**File:** `src/memories/schema_loader.py`

The World Builder's prompt needs the exact "Current Fact Values" block that
`evaluator.py` already builds inline (lines 75–102 today). Rather than duplicate ~25
lines of leaf-walking logic, extract it as a new public function:

```python
def render_current_fact_values(
    facts_blob: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> str:
    """Render every populated schema leaf as `path: "value"  [Mutability]`.

    Unpopulated leaves are summarised by a single trailing note rather than
    enumerated, keeping the block compact when most paths are unset.
    """
    if schema is None:
        schema = load_schema()
    leaves = _collect_leaves(schema)

    lines = ["## Current Fact Values"]
    populated: list[str] = []
    for path, leaf in leaves:
        parts = path.split(".")
        node: Any = facts_blob
        found = True
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                found = False
                break
            node = node[part]
        if not found or not isinstance(node, dict):
            continue
        value = node.get("Value")
        if value is None:
            continue
        populated.append(f'{path}: "{value}"  [{leaf["Mutability"]}]')

    lines.extend(populated)
    lines.append("")
    lines.append("(all other schema paths are unset)")
    return "\n".join(lines)
```

This is a byte-identical refactor of the existing inline block (the only formatting
difference is the function now always appends the trailing blank line + unset note,
matching the existing behaviour for both the populated and empty cases — re-check
against the current `evaluator.py` lines 96–102, which do the same thing for both
branches).

**Update `evaluator.py`:**

```python
# Before (lines 75-102, inline)
parts.append("\n## Current Fact Values")
schema = load_schema()
all_leaves = _collect_leaves(schema)
... # ~25 lines
```

```python
# After
from memories.schema_loader import render_current_fact_values, render_schema_for_prompt

parts.append("")
parts.append(render_current_fact_values(facts_blob))
```

Drop the now-unused `_collect_leaves, load_schema` import from `evaluator.py`'s
`from memories.schema_loader import ...` line, keeping only `render_schema_for_prompt`
and `render_current_fact_values`. No test in `test_evaluator_service.py` should need to
change — the rendered prompt text is identical to before.

---

### Part C — `services/world_builder.py` (new file, replaces `extraction_service.py`)

**Delete** `src/memories/services/extraction_service.py` in full (all of
`ExtractedFact`, `FactUpdate`, `ImplicitProposal`, `ExtractionResult`,
`ExtractionParseError`, `build_extractor_prompt`, `parse_extraction_result`,
`run_fact_extractor`). **Create** `src/memories/services/world_builder.py`:

```python
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

from memories.database import store_decision
from memories.database import set_facts as db_set_facts
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


def _set_leaf(blob: dict[str, Any], path: str, value: Any) -> None:
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
            value = entry.get("value")
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
                pass_name="world_builder",
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
```

Note the import alias `db_set_facts` for `database.set_facts` — needed because the
parameter `facts_blob` and the imported `set_facts` name would otherwise be fine, but
`db_set_facts` makes the call site (`await db_set_facts(db, character.id,
working_blob)`) unambiguous against the module's own conceptual "set facts" operation.
Either naming is acceptable; pick one and use it consistently.

**Why decision logging and persistence happen inside the handler, not after the loop
returns:** per `docs/plan-v2.md`'s Decision Logging section, tools that do not require
user input — `author_set_facts` is explicitly listed — are "logged immediately when the
tool is called, before the result is returned to the LLM." Doing the `set_facts` +
`store_decision` + notification inside the handler, before it returns its result string,
satisfies that ordering and also means a second `author_set_facts` call later in the same
pass (the model is *asked* to batch into one call, but nothing stops it from calling
twice) persists incrementally rather than silently overwriting the first call's writes.

**Why this is safe to run concurrently with `retrieve_experiences()`:** `run_turn()`
(Part D below) keeps the existing pattern of running the World Builder pass and
experience retrieval concurrently via `asyncio.gather`. Both touch the same
`aiosqlite.Connection`, but aiosqlite serialises all operations on a connection through
a single background thread/queue regardless of which coroutine issued them — concurrent
`await db.execute(...)` calls from two coroutines queue safely. The two passes also
touch disjoint tables (`character_facts` + `decisions` vs. `experiences`), so there is no
logical read/write conflict either.

---

### Part D — `services/chat_service.py` — wire in the World Builder

**Imports.** Remove:

```python
from memories.database import create_fact, ..., update_fact
from memories.services.extraction_service import (
    ExtractionParseError,
    ExtractionResult,
    run_fact_extractor,
)
```

`create_fact` and `update_fact` become unused once the Tier 1/2 write loop (below) is
deleted — confirm no other use in the file before removing. Add:

```python
from memories.services.ollama_client import OllamaClient, OllamaConnectionError, OllamaResponseError
from memories.services.sse_events import EventCallback, SSEEvent
from memories.services.world_builder import run_world_builder
```

(`SSEEvent`/`EventCallback` move here from the deleted inline definitions, per Part A.)

**Delete** the inline `SSEEvent` dataclass and `EventCallback` type alias (current lines
56–68).

**Replace** the extraction block inside `run_turn()`:

```python
# Before
async def _run_extraction_safe() -> ExtractionResult:
    try:
        return await run_fact_extractor(user_content, character, [], inferences, ollama)
    except (ExtractionParseError, OllamaConnectionError) as exc:
        _log.warning("fact extraction failed: %s", exc)
        return ExtractionResult()

(active, experience_scores), extraction_result = await asyncio.gather(
    retrieve_experiences(db, session.character_id, user_content, ollama, top_k=TOP_K_EXPERIENCES),
    _run_extraction_safe(),
)

# Process extraction results ... (Tier 1/2 create_fact/update_fact loop, ~20 lines)
```

```python
# After
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
        return facts_blob

(active, experience_scores), facts_blob = await asyncio.gather(
    retrieve_experiences(db, session.character_id, user_content, ollama, top_k=TOP_K_EXPERIENCES),
    _run_world_builder_safe(),
)
```

The `facts_blob =` reassignment is safe even though `_run_world_builder_safe` reads the
*original* `facts_blob` from the enclosing scope: Python evaluates both arguments to
`asyncio.gather` (and awaits them) before the tuple-unpacking assignment happens, so the
closure sees the pre-World-Builder value at call time, and the name is rebound to the
post-World-Builder value only once gather completes. No Tier 1/2 write loop remains:
the World Builder writes directly to `character_facts` inside its own handler, so there
is nothing left for `run_turn()` to do with the result except use it.

Everything downstream that already reads `facts_blob` — `build_system_prompt(character,
facts_blob, inferences, active or None)` and `run_contradiction_loop(model,
base_messages, character, facts_blob, ...)` — now automatically sees the
World-Builder-updated blob without any further changes, because both call sites already
came *after* this gather in the existing code.

**Return signature.** `run_turn()`'s return tuple drops `extraction_result` (no longer
produced):

```python
# Before
) -> tuple[str, str, int, EvaluatorResult, dict[int, float], ExtractionResult]:

# After
) -> tuple[str, str, int, EvaluatorResult, dict[int, float]]:
```

Update the final `return` statement to drop `extraction_result`, and update the
docstring's tuple description accordingly.

---

### Part E — `routers/chat.py` — drop the dead extraction sidechannels

**File:** `src/memories/routers/chat.py`

Update the result-tuple unpacking (currently 6 elements) to 5:

```python
# Before
(
    content,
    thinking,
    turn_id,
    eval_result,
    experience_scores,
    extraction_result,
) = _task.result()

# After
(
    content,
    thinking,
    turn_id,
    eval_result,
    experience_scores,
) = _task.result()
```

**Delete** the two sidechannel blocks that read `extraction_result` (current lines
122–170: the `extraction_applied` block and the `implicit_fact_proposed` block in full).
The World Builder now emits its own `world_builder_applied` sidechannel directly via
`on_event` from inside `run_world_builder()` (Part C) — `chat.py` does not need to build
or emit anything for it; the event is already queued and yielded by the existing
`_on_event`/`_q` plumbing in `_stream()` before `_task` resolves.

No other part of `chat.py` changes. The `status: extracting` event emitted at the top of
`_stream()` (`yield 'event: status\ndata: {"state": "extracting"}\n\n'`) is left exactly
as-is — it still accurately describes what the World Builder pass does, and
`frontend/chat.js`'s `_STATE_LABELS.extracting = 'Extracting facts from
conversation…'` needs no change.

---

## Transitional State After Step 4

After Step 4 and before Step 5:

- Facts the user states or implies are written to `character_facts` by the World
  Builder before the Character LLM runs, and are visible in the system prompt on the
  same turn. This closes the Step 3 gap where extraction wrote to a table the prompt
  never read.
- The World Builder has unrestricted write authority — it never checks mutability and
  cannot be blocked by an `Immutable` fact already being set. This is correct, final
  behaviour per `docs/plan-v2.md`, not a stopgap.
- The Character Evaluator is unchanged from Step 3: a single JSON-verdict call, no tool
  list, four valid verdicts. It cannot yet call `set_fact` or `propose_inference`, and
  `Mutable`/`Fluid` character-implied changes are still only noted in `decision_log`
  text, not acted on. That conversion is Step 5.
- The Character LLM still has no tool list at all (`require_fact` lands in Step 6).
- The `decisions` table now contains rows from two distinct passes
  (`pass_name="world_builder"` and `pass_name="character_evaluator"`) instead of one,
  whenever the World Builder writes at least one fact. A turn with nothing to extract
  produces only the `character_evaluator` row, same as before this step.
- A new `world_builder_applied` sidechannel event type is emitted but has no frontend
  handler — it is silently ignored by `chat.js`. The pre-existing `extraction_applied`
  and `implicit_fact_proposed` sidechannel handling in the frontend is now unreachable
  dead code (chat_service.py never emits those payloads again). Both are accepted gaps,
  cleaned up in Step 8 alongside the other deferred UI work.
- The legacy `facts` table stops receiving any writes from chat turns (the extractor
  that wrote to it is deleted). It is still fully readable/writable via the
  `facts.py` router's manual CRUD UI, which is unrelated to this step.

---

## Test Plan

### New: `tests/unit/test_world_builder.py` (replaces `tests/unit/test_extraction_service.py`)

Delete `tests/unit/test_extraction_service.py` in full. New file, using the `ollama`
fixture and `make_tool_call_response` / `make_multi_tool_call_response` /
`make_plain_tool_response` from `tests/unit/conftest.py`, plus a `db`/`character`
fixture for decision/blob assertions.

**`build_world_builder_prompt` — prompt content**

- `test_world_builder_prompt_includes_user_message`
- `test_world_builder_prompt_includes_fact_schema_section` — `"## Fact Schema"` present.
- `test_world_builder_prompt_includes_current_fact_values_section` —
  `"## Current Fact Values"` present.
- `test_world_builder_prompt_includes_populated_fact_value` — a blob with
  `Character.Identity.Occupation = "surgeon"` produces `"surgeon"` in the prompt.
- `test_world_builder_prompt_includes_inferences`
- `test_world_builder_prompt_omits_inferences_section_when_none` — empty list still
  renders `"(no inferences established yet)"`, not a blank section.
- `test_world_builder_prompt_instructs_single_batched_call` — prompt text mentions
  calling `author_set_facts` only once.
- `test_world_builder_prompt_instructs_no_new_paths`

**`run_world_builder` — tool-call mechanics**

- `test_run_world_builder_no_tool_call_returns_blob_unchanged` — model returns plain
  content immediately (`make_plain_tool_response`); returned blob `==` input
  `facts_blob`; no row in `get_decisions`; `on_event` not called.
- `test_run_world_builder_single_fact_written_to_blob`
- `test_run_world_builder_multiple_facts_in_one_call_all_written` — one
  `author_set_facts` call with 3 entries; all 3 present in the returned blob.
- `test_run_world_builder_preserves_existing_unrelated_facts` — pre-seed `facts_blob`
  with an unrelated path; assert it survives in the returned blob.
- `test_run_world_builder_overwrites_existing_value_regardless_of_mutability` —
  pre-seed an `Immutable` leaf (e.g. `Character.Identity.Name`); World Builder writes a
  different value to the same path with no error.
- `test_run_world_builder_unknown_path_returns_per_entry_error_other_entries_still_applied`
  — a batch with one bad path and one good path; the good path is present in the
  returned blob, the bad path is absent.
- `test_run_world_builder_enum_value_coerced_case_insensitively` — value `"calm"` for
  `Character.State-Of-Mind.Mood` is stored as `"Calm"`.
- `test_run_world_builder_enum_invalid_value_not_written` — value `"Apprehensive"` is
  rejected; path absent from the returned blob.
- `test_run_world_builder_integer_value_coerced_from_string` — `"33"` for
  `Character.Identity.Age` stored as the int `33`.
- `test_run_world_builder_integer_invalid_value_not_written`
- `test_run_world_builder_multiple_sequential_calls_accumulate` — two separate
  `author_set_facts` rounds (`side_effect` with two tool-call responses before the final
  plain response); facts from both calls present in the final blob.

**Decision logging**

- `test_run_world_builder_logs_decision_when_facts_written` — one row with
  `pass_name == "world_builder"`, `tool_name == "author_set_facts"`.
- `test_run_world_builder_decision_tool_args_contains_facts_list`
- `test_run_world_builder_no_decision_logged_when_no_tool_call`

**Sidechannel notification**

- `test_run_world_builder_emits_sidechannel_event_per_call` — `on_event` called once
  with `data["type"] == "world_builder_applied"` and `data["facts"]` matching what was
  written.
- `test_run_world_builder_no_event_emitted_when_no_facts_written`
- `test_run_world_builder_on_event_none_does_not_raise` — `on_event=None` with a
  fact-writing call completes without error.

**Cap behaviour**

- `test_run_world_builder_cap_reached_keeps_partial_writes` — pass `max_rounds=2` with
  a `side_effect` that keeps returning tool calls; assert facts written in the rounds
  that did execute are present in the returned blob, and a warning is logged (use
  `caplog`).

---

### `tests/unit/test_schema_loader.py` — additions

- `test_render_current_fact_values_header_present` — `"## Current Fact Values"`.
- `test_render_current_fact_values_empty_blob_shows_unset_note` —
  `"(all other schema paths are unset)"` present, no populated lines.
- `test_render_current_fact_values_populated_leaf_shown_with_mutability_tag` — a blob
  value renders as `path: "value"  [Mutability]`.
- `test_render_current_fact_values_only_lists_populated_leaves` — an unpopulated leaf
  alongside a populated one does not get its own line.

---

### `tests/unit/test_evaluator_service.py` — no changes required

The refactor in Part B is behaviour-preserving — `build_evaluator_prompt()`'s output is
identical before and after. All existing tests should pass unmodified; run them to
confirm rather than editing anything.

---

### `tests/unit/test_chat_service.py` — updates

**Module docstring (lines 1–22):** update "calls[0] — extractor LLM (Phase 6 fact
extraction)" to "calls[0] — World Builder LLM (Step 4)". Note that calls[0] may be more
than one HTTP request if the World Builder's mock simulates multiple tool-call rounds —
state that the *default* `_mock_world_builder()` helper used by most tests is a single
no-op round, preserving the "three calls total" assumption for every test that does not
specifically exercise World Builder fact-writing.

**Imports:** remove `from memories.services.extraction_service import ExtractionResult`
and `make_extractor_ndjson` from the `tests.unit.conftest` import line; add
`make_plain_tool_response`, `make_tool_call_response`, `make_multi_tool_call_response`,
and `get_facts` from `memories.database`.

**Helpers to update:**

- `_mock_extractor()` → rename `_mock_world_builder()`; body becomes
  `httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))`.
- `_mock_turn()` — update its docstring comment ("extractor" → "World Builder"); body
  unchanged (`_mock_world_builder()` replaces `_mock_extractor()` as the first list
  entry).

**Tests to delete** (the Phase 6 block, lines 876–1153 — all tested behaviour that no
longer exists):

- `test_run_turn_calls_extractor_before_character_llm`
- `test_run_turn_auto_adds_tier1_facts`
- `test_run_turn_auto_updates_tier2_facts`
- `test_run_turn_does_not_write_implicit_proposals_to_db`
- `test_run_turn_extraction_does_not_affect_system_prompt` — its premise (legacy-table
  writes are invisible to the blob-based prompt) is exactly what this step fixes; see
  the replacement test below, which asserts the opposite.
- `test_run_turn_character_prompt_does_not_include_implicit_proposals`
- `test_run_turn_returns_extraction_result`
- `test_run_turn_on_extraction_failure_continues_with_empty_result`
- `test_run_turn_on_ollama_connection_error_during_extraction_continues`
- `test_run_turn_deduplicates_tier1_facts_that_already_exist`
- `test_run_turn_empty_extraction_result_does_not_change_facts`

**Tests to update (rename + adjust assertion, behaviour otherwise intact):**

- `test_run_turn_embed_and_extractor_both_called_when_experiences_present` → rename
  `test_run_turn_embed_and_world_builder_both_called_when_experiences_present`; update
  its docstring/comment; assertion (`len(chat_route.calls) == 3`) is unchanged.
- `test_run_turn_returns_five_tuple_with_scores_dict` — change `assert len(result) ==
  6` to `assert len(result) == 5`, and `*_, scores, _ = result` to `*_, scores =
  result`. (The test's name already says "five_tuple"; this step is what finally makes
  that true.)

**Tests to add** (new World Builder integration block, replacing the deleted Phase 6
block):

- `test_run_turn_world_builder_runs_before_character_llm` — calls[0]'s request body
  contains `"tools"` with `body["tools"][0]["function"]["name"] ==
  "author_set_facts"`; calls[1]'s request body has no `"tools"` key.
- `test_run_turn_world_builder_writes_fact_to_character_facts_blob` — mock calls[0] with
  a tool call writing `Character.State-Of-Mind.Mood = "Anxious"`; after `run_turn`,
  `get_facts(db, character.id)` contains it.
- `test_run_turn_world_builder_fact_appears_in_character_system_prompt` — same setup;
  assert calls[1]'s system message contains the new value.
- `test_run_turn_world_builder_preserves_facts_not_touched_by_this_turn` — pre-seed an
  unrelated path via `set_facts` before calling `run_turn`; mock a no-op World Builder
  pass; assert the pre-seeded value survives in `get_facts` afterward.
- `test_run_turn_world_builder_failure_continues_with_unchanged_facts` — calls[0] is an
  `httpx.ConnectError`; turn still completes; calls[1]'s system message matches the
  pre-existing (unchanged) blob.
- `test_run_turn_world_builder_decision_logged_when_facts_written` — after a
  fact-writing turn, `get_decisions` returns two rows; one has `pass_name ==
  "world_builder"`.
- `test_run_turn_no_world_builder_decision_when_no_facts_written` — no-op pass produces
  exactly one decision row (`character_evaluator` only), matching pre-Step-4 row count.
- `test_run_turn_character_llm_request_has_no_tools_key` — explicit regression guard
  for "Character LLM tool list does not include `author_set_facts`": calls[1]'s request
  body has no `"tools"` key at all.
- `test_run_turn_world_builder_sidechannel_emitted_via_on_event` — pass an `on_event`
  collector into `run_turn`; assert one `SSEEvent(event="sidechannel", data={"type":
  "world_builder_applied", ...})` was received when a fact was written.
- `test_run_turn_world_builder_no_sidechannel_when_no_facts_written` — collector
  receives no `world_builder_applied` event for a no-op pass.

---

### `tests/integration/test_api_chat.py` — updates

**Imports:** drop `make_extractor_ndjson`; add `make_plain_tool_response` (and
`make_tool_call_response` for the new tests) from `tests.unit.conftest`.

**Helpers to update:**

- `_mock_extractor()` → rename `_mock_world_builder()`; body becomes
  `httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))`.
- `_mock_turn()` — comment update only.

**Tests to delete** (lines 1163–1626 — the Tier 1–4 / `extraction_applied` /
`implicit_fact_proposed` block):

- `test_turn_tier1_fact_added_to_db`
- `test_turn_tier1_fact_in_character_prompt`
- `test_turn_tier2_update_overwrites_fact_in_db`
- `test_turn_tier2_character_prompt_uses_new_value`
- `test_turn_tier3_proposal_not_written_to_db`
- `test_turn_tier4_proposal_does_not_overwrite_existing_fact`
- `test_turn_emits_extraction_applied_when_tier1_or_tier2_present`
- `test_turn_extraction_applied_added_list_has_key_and_fact_id`
- `test_turn_extraction_applied_updated_list_has_old_and_new_values`
- `test_turn_emits_implicit_fact_proposed_when_implicit_proposals_present`
- `test_turn_implicit_fact_proposed_new_proposals_list_populated`
- `test_turn_implicit_fact_proposed_update_proposals_has_old_value`
- `test_turn_both_sidechannel_events_emitted_in_same_turn`
- `test_turn_no_sidechannel_when_extraction_empty`

**Tests to update (kept — not extraction-system-specific, just need the helper rename
to keep working, plus one mock-shape fix):**

- `test_turn_emits_status_extracting_before_status_generating` — no change beyond the
  `_mock_extraction_turn()` helper it calls being updated for the new response shape
  (see below); the assertion about SSE state ordering is untouched.
- `test_turn_on_extractor_failure_still_delivers_response` → rename
  `test_turn_on_world_builder_failure_still_delivers_response`. The old mock injected
  malformed NDJSON content to trigger `ExtractionParseError`; that exception type no
  longer exists. Replace the first `side_effect` entry with `httpx.ConnectError("refused")`
  to exercise the real failure mode `_run_world_builder_safe()` now catches
  (`OllamaConnectionError`). Update the docstring accordingly.
- `_mock_extraction_turn(...)` helper (local to the Phase 6 block, used only by the two
  tests above after the rest of the block is deleted) — rename
  `_mock_world_builder_turn(...)`; its `new_facts=`/`fact_updates=`/
  `implicit_proposals=` parameters are removed since they no longer have a meaning;
  it becomes a thin wrapper equivalent to `_mock_turn()` and may simply be deleted in
  favour of reusing `_mock_turn()` directly in both retained tests.

**Tests to add** (replacing the deleted block — integration-level checks; the
exhaustive per-type validation matrix already lives in `test_world_builder.py`):

- `test_turn_world_builder_fact_written_to_character_facts_db` — mock calls[0] with a
  tool call; after the SSE response completes, `get_facts(db, character.id)` contains
  the written value.
- `test_turn_world_builder_fact_appears_in_character_prompt` — calls[1]'s request body
  system message contains the new value.
- `test_turn_emits_world_builder_applied_sidechannel` — SSE stream contains a
  `sidechannel` event with `type == "world_builder_applied"`.
- `test_turn_world_builder_applied_payload_has_path_and_value` — the event's `facts`
  list contains `{"path": ..., "value": ...}` entries matching what was written.
- `test_turn_no_world_builder_applied_sidechannel_when_no_facts_written`
- `test_turn_world_builder_unknown_path_does_not_write_but_turn_completes` — mock a
  tool call with a bogus path; `get_facts` does not contain it; the `message` event is
  still delivered normally.
- `test_turn_decisions_includes_world_builder_row_when_facts_written` —
  `GET /api/sessions/{id}/decisions` includes a row with `pass_name ==
  "world_builder"` after a fact-writing turn.

---

### `tests/integration/test_api_implication.py` — mechanical update only

- Update the import line to drop `make_extractor_ndjson`, add `make_plain_tool_response`.
- `_implication_turn()` — replace `httpx.Response(200, content=make_extractor_ndjson())`
  with `httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))`.
- `_pass_turn()` is unaffected — it already has no extractor call (regeneration after
  an accepted implication skips the pre-character pass entirely, consistent with
  `docs/plan-v2.md`'s description of edit/reject regeneration skipping the World
  Builder).
- No test additions or deletions; `_VIOLATION`/`_INFERENCE_VIOLATION` and the
  `implication` verdict flow they exercise are untouched by this step.

---

### `tests/integration/test_api_decisions.py` — mechanical update only

- Update the import line to drop `make_extractor_ndjson`, add `make_plain_tool_response`.
- `_mock_turn()` — replace `httpx.Response(200, content=make_extractor_ndjson())` with
  `httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))`.
- No test additions, deletions, or assertion changes — every existing test in this file
  exercises generic decision-row presence/ordering, which is unaffected as long as the
  World Builder mock stays a no-op (no second decision row appears).

---

## Files changed by this step

| Action | File | Notes |
|---|---|---|
| Add | `src/memories/services/sse_events.py` | `SSEEvent` + `EventCallback`, relocated to break a circular import |
| Add | `src/memories/services/world_builder.py` | Replaces `extraction_service.py` |
| Delete | `src/memories/services/extraction_service.py` | Fully superseded |
| Modify | `src/memories/schema_loader.py` | Add `render_current_fact_values()` |
| Modify | `src/memories/services/evaluator.py` | Use the new shared helper; drop now-unused imports |
| Modify | `src/memories/services/chat_service.py` | Import `SSEEvent`/`EventCallback` from `sse_events`; replace extraction call with World Builder call; drop Tier 1/2 write loop; return tuple shrinks to 5 elements |
| Modify | `src/memories/routers/chat.py` | Update tuple unpacking to 5 elements; delete `extraction_applied`/`implicit_fact_proposed` sidechannel blocks |
| Add | `tests/unit/test_world_builder.py` | Full new test suite |
| Delete | `tests/unit/test_extraction_service.py` | Superseded |
| Modify | `tests/unit/test_schema_loader.py` | Add `render_current_fact_values` tests |
| Modify | `tests/unit/test_chat_service.py` | Delete Phase 6 block; add World Builder block; rename/update two tests; update helpers and docstring |
| Modify | `tests/integration/test_api_chat.py` | Delete Tier 1–4/extraction sidechannel block; add World Builder integration tests; mechanical helper updates |
| Modify | `tests/integration/test_api_implication.py` | Mechanical helper update only |
| Modify | `tests/integration/test_api_decisions.py` | Mechanical helper update only |

No changes to `database.py`, `models/__init__.py`, `routers/facts.py`,
`routers/decisions.py`, `routers/test_poc.py`, `routers/require_fact.py`,
`services/tool_gate.py`, `services/prompt_builder.py`, `services/inference_service.py`,
`services/experience_service.py`, or the frontend.

---

## Edge Cases

- **Enum coercion is case-insensitive; no match is a hard per-entry error.** Per
  `docs/plan-v2.md`'s "Enum validation policy," `"calm"` → `"Calm"` succeeds silently;
  `"Apprehensive"` (no match) is rejected for that entry only — other entries in the
  same batch still apply. Per the project's existing enum-validation-loop pitfall: the
  error string returned to the model lists the path and the value that failed, **not**
  a copy of the full `Constraint` list reproduced separately for each failed entry in a
  multi-error batch — keep the message terse (one `Valid values: A, B, C` clause per
  failed entry is fine; the risk this guidance addresses is repeating the *entire* list
  redundantly across many retries, not stating it once per genuine error).
- **Integer coercion accepts numeric strings.** The model may emit `"value": "33"` (a
  JSON string) or `"value": 33` (a JSON number) for an `Integer` leaf depending on how
  it chooses to call the tool; `int(value)` handles both. A non-numeric string is a
  per-entry error.
- **World Builder authority overrides mutability unconditionally.** Writing to an
  already-set `Immutable` path produces no error and no contradiction — this is
  correct, not a bug. `leaves_by_path` lookups never consult `Mutability` for
  permission; it is read only for `Type`/`Constraint`.
- **Multiple `author_set_facts` calls in one pass.** The prompt asks for a single
  batched call, but the model is not prevented from calling it more than once (e.g.
  across tool-call rounds). `working_blob` accumulates across calls; each call persists
  and decision-logs independently, so no writes are lost regardless of how many rounds
  occur.
- **Tool-call cap reached mid-extraction.** Unlike the Character Evaluator (Step 5),
  the World Builder has no terminal tool to invoke — there is nothing analogous to
  `report_pass`/`report_contradiction` for it to call. Reaching `max_rounds` simply
  means extraction stops wherever it is; whatever was written in completed rounds is
  kept (already persisted incrementally), a warning is logged, and `run_turn()`
  proceeds normally with the partial `working_blob`. No fallback system message is
  injected — there is no terminal tool to instruct the model toward.
  `OllamaConnectionError`/`OllamaResponseError` is the only case `chat_service.py`'s
  `_run_world_builder_safe()` catches; the cap-reached case is not an exception at all
  (`chat_with_tools` returns a normal `ToolCallResult` with `cap_reached=True`) and
  needs no special handling at the `run_turn()` call site.
- **World Builder Ollama failure does not sink the turn.** Mirrors the old extractor's
  resilience posture: `_run_world_builder_safe()` catches
  `OllamaConnectionError`/`OllamaResponseError` and returns the unchanged `facts_blob`,
  so a Character LLM call still happens using the pre-turn fact state.
- **Concurrent DB access during the `asyncio.gather`.** The World Builder pass (which
  writes to `character_facts` and `decisions`) and `retrieve_experiences()` (which reads
  `experiences`) run concurrently inside the same `asyncio.gather`. This is safe:
  aiosqlite serialises all operations on one connection through a single worker
  regardless of caller, and the two passes touch disjoint tables.
- **No facts extracted is the common case and must stay cheap.** A message with no
  factual content results in exactly one World Builder HTTP call (model returns plain
  content with no tool calls) — no DB write, no decision row, no sidechannel event. This
  must remain true for the vast majority of conversational turns; it is the baseline
  every "no-op" test in this plan exercises.
- **`facts_blob` variable shadowing in `run_turn()`.** The closure inside
  `_run_world_builder_safe()` must read the pre-gather `facts_blob` (the value loaded
  by the first `asyncio.gather` at the top of `run_turn()`), not a value that has
  already been reassigned — this is automatically correct because the reassignment
  (`(active, experience_scores), facts_blob = await asyncio.gather(...)`) only happens
  after both gathered coroutines have been awaited to completion, but it is worth
  flagging explicitly during implementation since it is easy to introduce a bug here by
  reordering statements.

---

## Post-Implementation Cleanup Tasks

(Populated by `/review-step` after implementation.)
