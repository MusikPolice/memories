# Step 6 — `require_fact` and asyncio.Queue Coordination

## Overview

Step 6 wires the `require_fact(path, reason, suggested_value?)` tool into the live
Character LLM call path, replacing the synthetic Step 0b PoC with the real production
flow. Today the Character LLM is invoked via `ollama.chat(model, messages,
think=think)` — a plain streaming call with no tool list at all. After this step it is
invoked via `ollama.chat_with_tools(model, messages, [_REQUIRE_FACT_TOOL], {...},
think=think)`, mirroring how the World Builder (Step 4) and Character Evaluator
(Step 5) already converted from plain/JSON-mode calls to tool-calling loops. The
Character LLM's tool list contains exactly one tool — `require_fact` — and, unlike the
World Builder and Evaluator, it has no `terminal_tools`: the loop ends naturally the
moment the model returns plain content, exactly as `chat_with_tools()` already behaves
by default. Calling `require_fact` is mutually exclusive with generating prose in the
same round; the model either produces a final response or asks for a missing fact, not
both.

The suspension mechanism itself is not new. `tool_gate.py` (per-turn `asyncio.Queue`
keyed by `(session_id, turn_id)`) and the `POST
/{session_id}/turns/{turn_id}/require-fact/respond` endpoint
(`routers/require_fact.py`) were both built in Step 0b and already proved correct
against a synthetic stand-in (`routers/test_poc.py`) and against the real
`chat_with_tools()` loop (`tests/unit/test_tool_gate.py::test_gate_resolves_through_
callback_chain`). Step 6's job is narrower than it sounds: create the gate once per
turn in `run_turn()`, implement the `require_fact` tool handler that suspends on it,
and delete the now-redundant PoC scaffold.

Because the Character LLM now drives a tool-calling loop, `ollama.chat_with_tools()`
gains two things it never needed for the World Builder or Evaluator: a `think`
parameter (the Character LLM is the only pass that ever sets `think=True`) and
`thinking` token capture on the returned `ToolCallResult` (previously only `chat()`
captured this). Both are additive — every existing `chat_with_tools()` caller
(World Builder, Evaluator) is unaffected because `think` defaults to `False` exactly as
it always implicitly was.

**Success criterion:** all tests in `tests/unit/test_tool_calling.py`,
`tests/unit/test_chat_service.py`, `tests/unit/test_tool_gate.py`, and the
mechanically-updated integration files (`test_api_chat.py`, `test_api_decisions.py`,
`test_api_implication.py`) pass, along with every other existing test. The new
`tests/integration/test_require_fact_live.py` passes, exercising the real
`POST /api/sessions/{id}/messages` SSE endpoint end-to-end through a require_fact
suspend/resume cycle. The dev server starts cleanly with no PoC routes mounted.

---

## What Steps 0b–5 Delivered

- `src/memories/services/tool_gate.py` — `create_gate(session_id, turn_id)` (raises
  `ValueError` on duplicate registration), `await_gate(session_id, turn_id) -> str |
  None` (raises `KeyError` if no gate exists for the key — this is load-bearing for
  Step 6, see Edge Cases), `resolve_gate(session_id, turn_id, value)` (raises
  `asyncio.QueueFull` on double-resolve), `cleanup_gate(session_id, turn_id)`
  (no-op if absent). Module-level `_pending: dict[tuple[int, int], asyncio.Queue[str |
  None]]`, `maxsize=1` per gate — a single gate supports any number of sequential
  suspend/resume cycles across a turn (the queue is empty again after each `get()`),
  it just cannot have two values in flight at once.
- `src/memories/routers/require_fact.py` — `POST
  /{session_id}/turns/{turn_id}/require-fact/respond` with body `{"value": str |
  None}` (default `None`). Calls `resolve_gate()`; maps `KeyError` → 404, `asyncio.
  QueueFull` → 409. Already mounted in `main.py` at prefix `/api/sessions`. No changes
  in this step beyond a docstring edit (Part C).
- `src/memories/routers/test_poc.py` + `POST /{session_id}/test-require-fact-poc` — the
  Step 0b synthetic SSE endpoint that simulated the suspend/resume cycle without a real
  LLM call. Its own docstring already flags it as scaffolding to be removed once
  `require_fact` is wired into real `run_turn()` orchestration — that is this step
  (Part D).
- `tests/unit/test_tool_gate.py` — direct gate unit tests, plus
  `test_gate_resolves_through_callback_chain`, which already proves a `require_fact`
  handler awaiting a gate resolves correctly inside a real `ollama.chat_with_tools()`
  loop running concurrently with a resolver coroutine. This is the exact pattern Step 6
  productionises; no changes to this file.
- `tests/integration/test_require_fact_poc.py` — uvicorn-backed (not
  `ASGITransport`-backed) integration tests proving the SSE stream survives indefinite
  suspension, keep-alive pings flow, and the gate is cleaned up on every exit path
  (normal, error, dismiss, double-accept race). Deleted this step (Part D); its
  fixture pattern (`_bound_socket`, `poc_client`, `_SseCollector`) is the basis for the
  new `tests/integration/test_require_fact_live.py`.
- `src/memories/services/ollama_client.py` — `chat_with_tools(model, messages, tools,
  tool_handlers, max_rounds=MAX_TOOL_CALL_ROUNDS, terminal_tools=None) ->
  ToolCallResult`. `ToolCallResult` fields: `content: str`, `history: list[dict]`,
  `rounds: int`, `cap_reached: bool`, `terminal_call: dict | None`. No `think` support;
  no `thinking` field — `chat_with_tools()` strips `thinking` from each round's message
  before appending to history but never surfaces it to the caller. `chat()` (used by
  the eager pass, revalidation, and the session-end evaluator — none of which this step
  touches) already supports `think` and returns `thinking` in its metadata dict.
- `src/memories/services/chat_service.py` — `run_contradiction_loop(db, session_id,
  turn_id, model, base_messages, character, facts_blob, user_content, ollama,
  think=False, max_retries=MAX_CONTRADICTION_RETRIES, inferences=None,
  experiences=None, on_event=None) -> tuple[str, str, EvaluatorResult]`. Inside its
  retry loop it currently calls `raw_content, metadata = await ollama.chat(model,
  messages, think=think)` for the Character LLM, then `ev, facts_blob = await
  run_evaluator(db, character, session_id, turn_id, facts_blob, user_content, content,
  ollama, ...)` — the evaluator's returned blob is already threaded back into the local
  `facts_blob` so Fluid writes from one contradiction-retry attempt survive into the
  next. `run_turn(db, session_id, user_content, ollama, think=False, on_event=None)`
  resolves `character, facts_blob, inferences, history, turn_id` via one
  `asyncio.gather`, runs the World Builder and experience retrieval in a second
  `asyncio.gather`, builds `system_prompt` via `build_system_prompt()`, stores the user
  message, builds `base_messages`, calls `run_contradiction_loop()`, stores the
  assistant message, and returns `(char_content, char_thinking, turn_id, eval_result,
  experience_scores)`. No gate creation anywhere in this path today.
- `src/memories/services/evaluator.py` / `src/memories/services/world_builder.py` —
  both establish the project's tool-handler convention this step follows: a
  module-level tool-schema dict constant, private `_set_leaf()` /
  `_lookup_leaf_value()` helpers duplicated locally rather than imported across module
  boundaries (Step 5's stated rationale: "the two are simple enough that copying is
  clearer than reaching across a module boundary for a 7-line function"), and a
  `working_blob = copy.deepcopy(facts_blob)` pattern mutated inside the handler before
  being written back via `db_set_facts()`.
- `src/memories/schema_loader.py` — `load_schema()`, `check_write_permitted(path,
  schema) -> str` (returns `"Immutable"` / `"Mutable"` / `"Fluid"`; raises `ValueError`
  for unknown paths/groupings with a model-readable message), `_collect_leaves(node,
  prefix="") -> list[tuple[str, dict]]`.
- `src/memories/services/prompt_builder.py` — `_WORLD_STATE_PREAMBLE` already contains
  the line *"If a value is missing and you need it, call require_fact() rather than
  making one up."* (written ahead of this step). **No change needed in this step.**
- `src/memories/fact_schema.json` — confirmed to contain `Immutable` leaves of all
  three `Type`s used in this step's tests: `String` (`Character.Identity.Name`),
  `Integer` (`Character.Identity.Age`), and `Enum`
  (`Character.Appearance.Body.Build`, constraint `["Slim", "Athletic", "Average",
  "Curvy", "Stocky", "Heavyset"]`).
- `tests/unit/conftest.py` — `make_tool_call_response()`, `make_multi_tool_call_
  response()`, `make_plain_tool_response(content) -> bytes` (no `thinking` parameter
  today), `make_tool_call_response_with_thinking()`.

---

## What This Step Does NOT Change

- **The World Builder and its tool list.** `world_builder.py`, `author_set_facts`, and
  its handler are untouched.
- **The Character Evaluator.** `evaluator.py`'s tool list, handlers, and nudge/cap
  fallback are untouched. `run_evaluator()`'s signature and return type are unchanged.
- **`propose_inference` on the Character LLM (Option B from `docs/plan-v2.md`).** The
  Implementation Sketch for Step 6 lists only `require_fact` for the Character LLM's
  tool list. Option B is real design-doc content but is not scheduled to any numbered
  step in the sketch; it stays out of scope here. The Character LLM's tool list after
  this step contains exactly `require_fact` — nothing else.
- **Mutable and unset-Immutable `set_fact` approval flows in the Evaluator.** Still
  stub errors per Step 5. Step 7's job.
- **`routers/chat.py`.** Not edited at all. The existing generic
  `asyncio.Queue`-polling pattern in `_stream()` (poll with a 50 ms timeout, yield
  pings otherwise) already supports a task that suspends indefinitely mid-`run_turn()`
  — that is exactly what happens when the Character LLM calls `require_fact`. No new
  code is needed there for the suspension to work; events placed on `_q` via `on_event`
  (including the `require_fact` sidechannel card) flow through unchanged.
- **The frontend.** `chat.js`'s `buildNotificationFromSidechannel()` returns `null` for
  the `require_fact` sidechannel type, so the card ships inert — exactly like
  `fact_update_fluid` and `inference_proposed` did after Step 5. Wiring it into the UI
  (a blocking card with a text/dropdown input depending on schema `Type`) is Step 8.
- **`implication.py`'s `accept_implication()` regeneration path.** It already calls
  `run_contradiction_loop(db, session_id, turn_id, ...)` (Step 5 added the three
  leading args). If the regenerated Character LLM call invokes `require_fact` here, no
  gate exists for that `(session_id, turn_id)` — the original gate was created and torn
  down inside the original `run_turn()` call that produced the now-being-regenerated
  turn. This is intentional and requires no new code: see Edge Cases for why the
  failure mode is graceful, not a crash.
- **`CLAUDE.md` / `README.md`.** Step 10.

---

## Detailed Design

### Part A — `services/ollama_client.py` — `think` and `thinking` on `chat_with_tools()`

**File:** `src/memories/services/ollama_client.py`

**`ToolCallResult` — add one field**, appended after `terminal_call` so no existing
keyword-argument construction (e.g. in `tests/unit/test_evaluator_service.py`) breaks:

```python
# Before
@dataclass
class ToolCallResult:
    content: str
    history: list[dict[str, Any]]
    rounds: int
    cap_reached: bool
    terminal_call: dict[str, Any] | None = None

# After
@dataclass
class ToolCallResult:
    content: str
    history: list[dict[str, Any]]
    rounds: int
    cap_reached: bool
    terminal_call: dict[str, Any] | None = None
    # Accumulated `thinking` text across every round, in order, joined with no
    # separator — mirrors chat()'s `"".join(thinking_parts)` behaviour. "" if the
    # model never thought, or if `think=False` was passed (the default).
    thinking: str = ""
```

**`chat_with_tools()` — add the `think` parameter** at the end of the signature
(matching how Step 5 appended `terminal_tools` last):

```python
# Before
async def chat_with_tools(
    self,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, ToolHandler],
    max_rounds: int = MAX_TOOL_CALL_ROUNDS,
    terminal_tools: frozenset[str] | None = None,
) -> ToolCallResult:

# After
async def chat_with_tools(
    self,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, ToolHandler],
    max_rounds: int = MAX_TOOL_CALL_ROUNDS,
    terminal_tools: frozenset[str] | None = None,
    think: bool = False,
) -> ToolCallResult:
```

Update the docstring to add: *"``think`` is forwarded to Ollama on every round exactly
as it is for ``chat()``; ``ToolCallResult.thinking`` accumulates every round's thinking
text, stripped from history before the next round is sent, for the same reason
``chat()`` strips it from the conversation it returns."*

**Body changes** — add a `thinking_parts` accumulator, send `think` in every round's
payload, and capture each round's thinking before stripping it from history. All three
return points gain `thinking="".join(thinking_parts)`:

```python
        url = f"{self.base_url}/api/chat"
        history: list[dict[str, Any]] = list(messages)
        content = ""
        rounds = 0
        thinking_parts: list[str] = []

        ...

        while rounds < max_rounds:
            payload: dict[str, Any] = {
                "model": model,
                "messages": history,
                "tools": tools,
                "stream": False,
                "think": think,
            }
            ...
            data: dict[str, Any] = response.json()
            rounds += 1
            msg: dict[str, Any] = data["message"]
            thought: str = msg.get("thinking", "") or ""
            if thought:
                thinking_parts.append(thought)
            history.append({k: v for k, v in msg.items() if k != "thinking"})

            tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []
            if not tool_calls:
                content = msg.get("content", "") or ""
                return ToolCallResult(
                    content=content,
                    history=history,
                    rounds=rounds,
                    cap_reached=False,
                    thinking="".join(thinking_parts),
                )

            ...

            if terminal_call is not None:
                return ToolCallResult(
                    content=content,
                    history=history,
                    rounds=rounds,
                    cap_reached=False,
                    terminal_call=terminal_call,
                    thinking="".join(thinking_parts),
                )

        return ToolCallResult(
            content=content,
            history=history,
            rounds=rounds,
            cap_reached=True,
            thinking="".join(thinking_parts),
        )
```

Nothing else in the function changes — the `for call in tool_calls:` block, error
handling, and `terminal_call` detection from Step 5 are untouched.

**Backward compatibility.** `think` defaults to `False`, so every existing caller
(`world_builder.py`, `evaluator.py`, and every test that doesn't pass `think=`) now
sends an *explicit* `"think": False` in the payload where it previously sent no
`think` key at all. No test in the suite asserts the absence of a `"think"` key (
confirmed by grep across `test_tool_calling.py`, `test_evaluator_service.py`,
`test_world_builder.py`), so this is safe. `thinking` defaults to `""` and is additive
to every existing `ToolCallResult` access pattern in the codebase.

---

### Part B — `services/chat_service.py` — the `require_fact` tool, handler, and gate wiring

**File:** `src/memories/services/chat_service.py`

#### Imports

```python
# Before
from __future__ import annotations

import asyncio
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
from memories.services.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from memories.services.prompt_builder import build_system_prompt
from memories.services.sse_events import EventCallback as EventCallback
from memories.services.sse_events import SSEEvent as SSEEvent
from memories.services.world_builder import run_world_builder

# After
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
```

`store_decision` and `set_facts` were removed from this file's imports in Step 5 (the
evaluator's own decision-logging made the aggregate call here redundant); both are
needed again for `require_fact`'s own decision row and fact write.

#### New module-level constants and helpers (after the existing `_log`/
`MAX_CONTRADICTION_RETRIES` lines)

```python
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
```

These two helpers are byte-for-byte duplicates of the ones already in
`evaluator.py` and `world_builder.py` — intentional, per the project's established
convention of copying small (~7-line) private helpers rather than importing across
module boundaries (see "What Steps 0b–5 Delivered" above).

#### `run_contradiction_loop()` — new pre-loop setup, rewritten character-LLM call

The function signature is **unchanged** (still `db, session_id, turn_id, model,
base_messages, character, facts_blob, user_content, ollama, think=False,
max_retries=..., inferences=None, experiences=None, on_event=None`). Add the schema
lookup and the `_handle_require_fact` closure right after the existing local-variable
declarations, before the `for attempt in range(...)` loop:

```python
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

        value = await await_gate(session_id, turn_id)

        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_llm",  # nosec B106
            tool_name="require_fact",
            tool_args={"path": path, "reason": reason, "suggested_value": suggested_value},
            user_input={"value": value},
        )

        if value is None:
            return (
                f"No value was provided for {path}. Do not invent one — respond "
                "without depending on it, or acknowledge that it is not yet known."
            )

        coerced: Any = value
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
                    "require_fact: user value %r for %s is not a valid integer — "
                    "storing verbatim",
                    value,
                    path,
                )

        working_blob = copy.deepcopy(facts_blob)
        _set_leaf(working_blob, path, coerced)
        await db_set_facts(db, character.id, working_blob)
        facts_blob = working_blob

        return f"{path} = {coerced!r}. Use this value now."
```

`nonlocal facts_blob` is what makes this work without changing
`run_contradiction_loop()`'s return signature: `facts_blob` is a parameter of the
enclosing function, the same name the loop already rebinds via `ev, facts_blob =
await run_evaluator(...)`. The closure captures the *name*, not a snapshot of its
value, so a write inside `_handle_require_fact` is immediately visible to the very
next `run_evaluator()` call in the same attempt, and to any subsequent
`_handle_require_fact` call in a later attempt — exactly the same threading
guarantee Step 5 established for Fluid `set_fact` writes across contradiction
retries.

**Replace** the character-LLM invocation inside the `for attempt in
range(max_retries + 1):` loop:

```python
# Before
        if attempt == 0 and on_event is not None:
            await on_event(SSEEvent(event="status", data={"state": "generating"}))
        raw_content, metadata = await ollama.chat(model, messages, think=think)
        content = raw_content
        thinking = str(metadata.get("thinking", ""))

# After
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
```

No other line in `run_contradiction_loop()` changes — the evaluator call, the
contradiction-hint bookkeeping, and the final return are exactly as Step 5 left them.

#### `run_turn()` — gate creation, one `try`/`finally` wrap

The gate must exist before `run_contradiction_loop()` can run (its
`_handle_require_fact` calls `await_gate()`, which raises `KeyError` if no gate is
registered for the key) and must be removed once the turn is done, on every exit path
including exceptions — exactly the guarantee `cleanup_gate()` is designed for and that
`test_require_fact_poc.py` already proved against the PoC endpoint.

```python
# Before
    character, facts_blob, inferences, history, turn_id = await asyncio.gather(
        get_character(db, session.character_id),
        get_facts(db, session.character_id),
        get_inferences(db, session.character_id),
        get_messages(db, session_id),
        next_turn_id(db, session_id),
    )
    assert character is not None

    # --- Parallel: experience retrieval (embed) + World Builder (LLM + DB writes) ---
    async def _run_world_builder_safe() -> dict[str, Any]:
        ...
    ( ... entire existing body ... )
    return char_content, char_thinking, turn_id, eval_result, experience_scores

# After
    character, facts_blob, inferences, history, turn_id = await asyncio.gather(
        get_character(db, session.character_id),
        get_facts(db, session.character_id),
        get_inferences(db, session.character_id),
        get_messages(db, session_id),
        next_turn_id(db, session_id),
    )
    assert character is not None

    create_gate(session_id, turn_id)
    try:
        # --- Parallel: experience retrieval (embed) + World Builder (LLM + DB writes) ---
        async def _run_world_builder_safe() -> dict[str, Any]:
            ...
        ( ... entire existing body, indented one level deeper ... )
        return char_content, char_thinking, turn_id, eval_result, experience_scores
    finally:
        cleanup_gate(session_id, turn_id)
```

Concretely: indent every line from the `# --- Parallel: experience retrieval ---`
comment through the final `return` statement one level deeper inside a `try:` block;
add `create_gate(session_id, turn_id)` immediately before the `try:`; add `finally:
cleanup_gate(session_id, turn_id)` after it. No line inside that body changes.

`get_session`/`session.ended_at` validation and the initial `asyncio.gather()` that
resolves `turn_id` stay outside the `try` — they raise before any gate exists, so
there is nothing to clean up for those paths.

---

### Part C — `routers/require_fact.py` — docstring update only

**File:** `src/memories/routers/require_fact.py`

```python
# Before
"""Permanent accept/dismiss endpoint for require_fact blocking cards.

Used by both the Step 0b PoC and the production flow (Step 6).  The client
POSTs here when the user fills in (or dismisses) a require_fact card.
"""

# After
"""Accept/dismiss endpoint for require_fact blocking cards.

The client POSTs here when the user fills in (or dismisses) a require_fact card
surfaced by the Character LLM's require_fact tool handler (see
chat_service.run_contradiction_loop._handle_require_fact).
"""
```

No code in this file changes — `_RespondBody(value: str | None = None)`,
`resolve_gate()`, and the 404/409 error mapping are exactly what the production flow
needs already.

`src/memories/services/tool_gate.py` is unchanged entirely — no docstring or code
edits.

---

### Part D — Remove the Step 0b PoC scaffold

**Delete** `src/memories/routers/test_poc.py` in full — its only consumer
(`tests/integration/test_require_fact_poc.py`) is also deleted in this step (Part of
Test Plan below), and its own docstring already states it is "scaffolding that will be
removed in Step 6 when require_fact is integrated into the real run_turn()
orchestration."

**File:** `src/memories/main.py`

```python
# Before
from memories.routers import (
    characters,
    chat,
    decisions,
    experiences,
    facts,
    implication,
    inferences,
    require_fact,
    schema,
    sessions,
    test_poc,
)
...
app.include_router(test_poc.router, prefix="/api/sessions", tags=["test_poc"])
app.include_router(require_fact.router, prefix="/api/sessions", tags=["require_fact"])

# After
from memories.routers import (
    characters,
    chat,
    decisions,
    experiences,
    facts,
    implication,
    inferences,
    require_fact,
    schema,
    sessions,
)
...
app.include_router(require_fact.router, prefix="/api/sessions", tags=["require_fact"])
```

`require_fact.router` stays mounted — it is the permanent endpoint, not PoC scaffold.

---

### Part E — `tests/unit/conftest.py` — `thinking` on `make_plain_tool_response()`

**File:** `tests/unit/conftest.py`

```python
# Before
def make_plain_tool_response(content: str) -> bytes:
    """Build a non-streaming Ollama JSON response with plain content and no tool calls."""
    obj = {
        "message": {"role": "assistant", "content": content, "tool_calls": None},
        "done": True,
    }
    return json.dumps(obj).encode()

# After
def make_plain_tool_response(content: str, thinking: str = "") -> bytes:
    """Build a non-streaming Ollama JSON response with plain content and no tool calls.

    Pass *thinking* to simulate a final round that both thinks and replies in plain
    text — the shape `chat_with_tools()` must handle when `think=True` is requested
    for the Character LLM.
    """
    message: dict[str, Any] = {"role": "assistant", "content": content, "tool_calls": None}
    if thinking:
        message["thinking"] = thinking
    obj = {"message": message, "done": True}
    return json.dumps(obj).encode()
```

Purely additive — every existing single-argument call site is unaffected.

---

## Transitional State After Step 6

After Step 6 and before Step 7:

- The Character LLM communicates via `chat_with_tools()` with a single tool,
  `require_fact`. It still has no access to `set_fact`, `author_set_fact`, or
  `propose_inference` — fact writes from the character's perspective remain the
  Character Evaluator's job, post-generation.
- An unset `Immutable` path the character needs mid-generation is requested via
  `require_fact`, which suspends the SSE stream, surfaces a blocking
  `require_fact` sidechannel card (inert in the UI until Step 8), and resumes the
  Character LLM with the user's confirmed/edited/dismissed value once
  `POST .../require-fact/respond` is called.
- An `Immutable` path that is already set, or any `Mutable`/`Fluid` path, returns a
  tool error if the model mistakenly calls `require_fact` on it — no suspension
  occurs, matching the evaluator's existing pattern of returning a guiding error
  rather than crashing.
- `require_fact` decisions are logged with `pass_name="character_llm"`,
  `tool_name="require_fact"`, `tool_args` carrying the model's call arguments, and
  `user_input={"value": ...}` carrying what the user actually provided (or `None` for
  a dismiss) — this is the first decision row ever logged with `pass_name=
  "character_llm"`; the column comment in `database.py`'s `decisions` DDL already
  anticipated this value.
- The Step 0b PoC endpoint and its dedicated integration test file are gone. The
  suspension mechanism they validated is now exercised by the real
  `/api/sessions/{id}/messages` endpoint instead, via
  `tests/integration/test_require_fact_live.py`.
- The Character Evaluator's `set_fact` still returns stub errors for Mutable and
  unset-Immutable paths — Step 7's job. The Evaluator could in principle now observe
  a freshly-`require_fact`'d Immutable value already set in `facts_blob` and correctly
  treat any matching `set_fact` call from itself as "already set, no-op" rather than
  "unset, stub error" — this naturally falls out of Step 6's blob-threading and needs
  no extra code.
- `routers/implication.py`'s regeneration path can still theoretically trigger
  `require_fact` (its Character LLM call now goes through the same
  `run_contradiction_loop()`), but since no gate exists for an already-completed
  turn being regenerated, any such call resolves as a graceful tool error, not a
  suspension. See Edge Cases.

---

## Test Plan

### `src/memories/services/ollama_client.py` is covered by `tests/unit/test_tool_calling.py` — additions

New section, e.g. `# think / thinking behaviour`, placed after the existing "Terminal
tool behaviour" section:

- `test_think_false_sent_by_default` — `body["think"] is False` in a single-round
  plain-content response with no `think=` argument passed.
- `test_think_true_sent_when_requested` — call with `think=True`; assert
  `body["think"] is True`.
- `test_think_sent_in_every_round` — a tool-call round followed by a plain-content
  round, `think=True`; assert `body["think"] is True` for **both** `route.calls`,
  mirroring `test_tools_and_stream_sent_in_all_rounds`.
- `test_thinking_empty_by_default` — plain-content response with no thinking field;
  `result.thinking == ""`.
- `test_thinking_captured_from_plain_content_response` — use
  `make_plain_tool_response("Hi.", thinking="Considering...")`; assert
  `result.thinking == "Considering..."` and `result.content == "Hi."`.
- `test_thinking_accumulated_across_tool_and_final_rounds` — round 1:
  `make_tool_call_response_with_thinking("set_fact", _DEFAULT_ARGS, thinking="Step
  one.")`; round 2: `make_plain_tool_response("Done.", thinking="Step two.")`; assert
  `result.thinking == "Step one.Step two."` (joined with no separator, matching
  `chat()`'s accumulation behaviour) and `result.content == "Done."`.
- `test_thinking_still_stripped_from_history` — re-run
  `test_thinking_stripped_from_history`'s scenario and additionally assert
  `result.thinking != ""`, proving thinking is captured on the `ToolCallResult` at the
  same time it is stripped from the conversation history fed back to the model.

No changes needed to `tests/unit/test_ollama_client.py` (it tests `chat()`, which is
untouched by this step).

---

### `tests/unit/test_chat_service.py` — updates and additions

**Module docstring** — update the "calls[1] — character LLM" line to note it is now a
tool-calling, non-streaming call (mirroring calls[2]'s existing description), and that
the default `_mock_ok()` helper produces a single no-tool-call round so the "three
calls total" assumption for ordinary tests is preserved.

**Imports** — add `import asyncio`; add `resolve_gate` and the `tool_gate` module
itself to the `tests.unit.conftest`/`memories.services` import lines:

```python
import memories.services.tool_gate as tool_gate_module
from memories.services.tool_gate import resolve_gate
```

**`_mock_ok()` — redefine** to stop using NDJSON streaming format, since the
Character LLM call no longer goes through `ollama.chat()`:

```python
# Before
def _mock_ok(content: str = "I am fine, thank you.") -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(content))

# After
def _mock_ok(content: str = "I am fine, thank you.") -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response(content))
```

Drop `make_ollama_ndjson` from this file's imports if nothing else in it still uses
NDJSON-format mocking (grep `make_ollama_ndjson` in the file first — every call site
identified is this one helper).

Every test that uses `_mock_ok()` or `_mock_turn()` continues to pass unchanged — the
helper produces a structurally different but functionally equivalent mock (plain
content, no tool calls, single HTTP round-trip), and `chat_with_tools()` already
returns that content via the same "no tool_calls → return content directly" path that
`evaluator.py`/`world_builder.py` no-op rounds already exercise via
`make_plain_tool_response`.

**Tests to update:**

- `test_run_turn_character_llm_request_has_no_tools_key` — **delete.** Its premise is
  now false: the Character LLM request *does* carry a `tools` key (`require_fact`).
  Replace with `test_run_turn_character_llm_tools_list_is_require_fact_only`:
  `char_body["tools"][0]["function"]["name"] == "require_fact"` and
  `len(char_body["tools"]) == 1`.

**Tests to add** (new section, e.g. `# Step 6 additions — require_fact`):

- `test_run_turn_character_require_fact_suspends_and_resumes` — full code below; the
  canonical happy-path test other require_fact tests are variations of.
- `test_run_turn_character_require_fact_writes_value_to_blob` — same setup; after
  `run_turn`, `get_facts(db, character.id)["Character"]["Identity"]["Name"]["Value"]
  == "Sarah"`.
- `test_run_turn_character_require_fact_dismiss_leaves_path_unset` —
  `resolve_gate(session.id, 1, None)` instead of a value; round 2 mocks an
  acknowledgement reply; `get_facts()` has no `Name` entry under `Character.Identity`
  afterward.
- `test_run_turn_character_require_fact_logs_decision_with_user_input` — after the
  happy-path flow, `get_decisions(db, session.id)` contains a row with
  `pass_name == "character_llm"`, `tool_name == "require_fact"`, and
  `user_input == {"value": "Sarah"}`.
- `test_run_turn_character_require_fact_emits_sidechannel_before_suspension` —
  `on_event` collector receives an event with `data["type"] == "require_fact"`,
  `data["path"] == "Character.Identity.Name"`, `data["reason"]`, and
  `data["suggested_value"]` present, before the resolver fires.
- `test_run_turn_gate_removed_after_turn_completes` — after a normal (non-require_fact)
  `_mock_turn()` run, `(session.id, 1) not in tool_gate_module._pending`.
- `test_run_turn_gate_removed_after_character_llm_connection_error` — mock the
  character-LLM round (`route.calls[1]`) with `httpx.ConnectError`; assert
  `OllamaConnectionError` propagates from `run_turn` **and**
  `(session.id, 1) not in tool_gate_module._pending` afterward — proves the `finally`
  cleans up even when the turn never reaches the gate.
- `test_run_turn_character_require_fact_immutable_already_set_returns_error` —
  pre-seed `Character.Identity.Name = "Alice"` via `set_facts()`; round 1 mocks
  `require_fact` on that same path; round 2 mocks a plain self-corrected reply; assert
  the final content matches round 2's text, no `sidechannel` event with
  `type == "require_fact"` was ever emitted, and exactly 4 total `_CHAT_URL` calls were
  made (world builder + character round 1 + character round 2 + evaluator) — proving
  the handler short-circuits with a tool error rather than suspending.
- `test_run_turn_character_require_fact_mutable_path_returns_error` — round 1 calls
  `require_fact` on `Character.Identity.Occupation` (Mutable); assert the tool result
  text (captured via a thin wrapper around the handler, or inferred indirectly through
  round 2's mocked self-correction) is reachable without suspension — same
  4-total-calls shape as the immutable-already-set test.
- `test_run_turn_character_require_fact_unknown_path_returns_error` — round 1 calls
  `require_fact` with `path="Nonexistent.Path"`; same no-suspension shape; the model
  self-corrects in round 2.
- `test_run_turn_character_require_fact_enum_path_coerced_case_insensitively` — target
  `Character.Appearance.Body.Build` (Immutable Enum); resolver supplies `"athletic"`;
  assert the stored `Value` is `"Athletic"` (constraint-cased).
- `test_run_turn_character_require_fact_integer_path_coerced_from_string` — target
  `Character.Identity.Age` (Immutable Integer); resolver supplies `"34"`; assert the
  stored `Value` is the int `34`, not the string `"34"`.
- `test_run_turn_character_two_sequential_require_fact_calls_same_turn` — round 1 calls
  `require_fact` for `Character.Identity.Name`; after resolving, round 2 *also* calls
  `require_fact`, this time for `Character.Identity.Pronouns`; after resolving that
  one too, round 3 is plain content. Proves the single per-turn gate is correctly
  reusable across more than one suspend/resume cycle within the same turn (the queue
  is empty again after each `get()`).

**Full code for the canonical test** (others above are straightforward variations):

```python
async def test_run_turn_character_require_fact_suspends_and_resumes(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        await seen.wait()
        resolve_gate(session.id, 1, "Sarah")

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Name",
                            "reason": "I need my name to introduce myself",
                            "suggested_value": "Elena",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Hi, I'm Sarah.")),
                _mock_eval("pass"),
            ]
        )
        (content, *_rest), _ = await asyncio.gather(
            run_turn(db, session.id, "What's your name?", ollama, on_event=_on_event),
            _resolver(),
        )

    assert content == "Hi, I'm Sarah."
```

(`turn_id` is `1` because each test gets a fresh in-memory DB via the `db` fixture —
`next_turn_id()` returns `1` for the first message in a new session, matching the
existing convention used implicitly throughout this file.)

---

### `tests/integration/test_api_chat.py` — updates

**Imports** — drop `make_ollama_ndjson` once both call sites below are converted
(`make_plain_tool_response` and `make_tool_call_response` are already imported).

**`_mock_ok()` — same redefinition as the unit-test file:**

```python
def _mock_ok(content: str = "I am fine.") -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response(content))
```

**Inline literal — `test_thinking_event_emitted_when_model_thinks`** (around line
208): replace

```python
httpx.Response(
    200,
    content=make_ollama_ndjson("My answer.", thinking="Let me consider this carefully."),
),
```

with

```python
httpx.Response(
    200,
    content=make_plain_tool_response(
        "My answer.", thinking="Let me consider this carefully."
    ),
),
```

This is exactly the scenario Part E's `make_plain_tool_response(thinking=...)`
addition exists for — it must keep passing unmodified in its assertions
(`thinking_events[0]` content equals the same string).

**Inline literal — `test_accept_implication_on_high_mutability_fact_preserves_
mutability`** (around line 1019): replace
`make_ollama_ndjson("I feel anxious today.")` with
`make_plain_tool_response("I feel anxious today.")`.

**No other changes needed.** Every other `make_ollama_ndjson` call site identified by
grep in this file goes through `_mock_ok()`, already covered by the helper
redefinition above.

**Tests to add:**

- `test_require_fact_sidechannel_type_present_in_real_send_message_flow` — a smoke
  test using the existing `ASGITransport`-backed `client` fixture, asserting only that
  a request whose Character LLM round calls `require_fact` does **not** crash the
  endpoint when immediately resolved out-of-band before the response is read (since
  `ASGITransport` buffers the full response, this test must pre-resolve the gate via
  a concurrent task exactly like the unit-test pattern, *or* be skipped in favour of
  relying entirely on the new `test_require_fact_live.py` for true suspension
  coverage — prefer the latter; do not fight `ASGITransport`'s buffering here. If
  added, gate it behind `asyncio.gather` like the unit test above, with the resolver
  task started before `await client.post(...)`).

---

### `tests/integration/test_api_decisions.py` — updates

**`_mock_turn()` — single-line change:**

```python
# Before
def _mock_turn(character_content: str = "I am fine.") -> list[httpx.Response]:
    return [
        httpx.Response(200, content=make_plain_tool_response("Nothing to extract.")),
        httpx.Response(200, content=make_ollama_ndjson(character_content)),
        httpx.Response(200, content=make_tool_call_response("report_pass", {})),
    ]

# After
def _mock_turn(character_content: str = "I am fine.") -> list[httpx.Response]:
    return [
        httpx.Response(200, content=make_plain_tool_response("Nothing to extract.")),
        httpx.Response(200, content=make_plain_tool_response(character_content)),
        httpx.Response(200, content=make_tool_call_response("report_pass", {})),
    ]
```

Drop `make_ollama_ndjson` from the imports. No test bodies change — every existing
assertion in this file reads `tool_name`/`pass_name`/`turn_id` from the decisions API
response, none of which are affected by how the character round was mocked.

---

### `tests/integration/test_api_implication.py` — updates

**Imports** — drop `make_ollama_ndjson`.

**`_implication_turn()` and `_pass_turn()`** — replace their `make_ollama_ndjson(
character_content)` line with `make_plain_tool_response(character_content)` in both
helpers.

**Three inline literal call sites** (identified by grep: lines ~189, ~357, ~406) —
each is `httpx.Response(200, content=make_ollama_ndjson("..."))` standing in for one
character-LLM round inside a larger `side_effect` list. Replace each with
`httpx.Response(200, content=make_plain_tool_response("..."))`, same string argument.

**No other changes.** Every occurrence of `make_ollama_ndjson` in this file represents
the character-LLM round (confirmed by grep — there are no eager-pass, revalidation, or
session-end-evaluator mocks in this file); the replacement above is exhaustive.

---

### `tests/integration/test_require_fact_poc.py` — delete in full

The PoC endpoint it tests (`POST /{session_id}/test-require-fact-poc`) is removed in
Part D. Its fixtures and SSE-collector helper are not deleted outright — they are the
starting point for the new file below.

---

### `tests/integration/test_require_fact_live.py` — new file

Real end-to-end coverage of `require_fact` through the production
`POST /api/sessions/{id}/messages` SSE endpoint, replacing the PoC's synthetic
endpoint coverage. Must use a real uvicorn server, not the `ASGITransport`-backed
`client` fixture from `tests/integration/conftest.py` — `ASGITransport` buffers the
full response before yielding anything, which deadlocks a test that must POST to the
respond endpoint while the SSE stream is still open (the exact reason
`test_require_fact_poc.py` built its own `poc_client` fixture; see
`docs/streaming-plan.md`'s established convention and this project's
`project_asgi_transport_streaming` memory note).

**Fixture** — adapt `test_require_fact_poc.py`'s `_bound_socket()`/`poc_client()`
pattern, renamed `live_client`, with one addition: override `get_ollama` (the PoC
never called Ollama, so it never needed this) so the real `/messages` endpoint's
Ollama calls flow through a respx-interceptable `OllamaClient`:

```python
@pytest.fixture
async def live_client(db: aiosqlite.Connection) -> AsyncGenerator[AsyncClient, None]:
    ollama_client = OllamaClient(base_url=OLLAMA_BASE_URL)

    async def _override_db() -> AsyncGenerator[aiosqlite.Connection, None]:
        yield db

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_ollama] = lambda: ollama_client

    sock = _bound_socket()
    port = sock.getsockname()[1]
    config = uvicorn.Config(app, loop="none", lifespan="off", log_level="error")
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await asyncio.wait_for(_wait_until(lambda: server.started), timeout=10.0)
    except TimeoutError as exc:
        server.should_exit = True
        await serve_task
        raise RuntimeError("uvicorn failed to start within 10 seconds") from exc

    async with AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        try:
            yield client
        finally:
            server.should_exit = True
            await serve_task

    app.dependency_overrides.clear()
    await ollama_client.aclose()
```

(`_bound_socket`, `_wait_until`, `OLLAMA_BASE_URL = "http://test-ollama-integration:
11434"` and the `_SseCollector` class are copied over from the deleted PoC file
unchanged.) `lifespan="off"` means `deps.set_ollama()`/`deps.set_db()` (normally set by
`main.py`'s lifespan) never run — the dependency overrides above substitute for both.

**Tests:**

- `test_require_fact_sidechannel_emitted_before_done` — mock `_OLLAMA_CHAT_URL` with
  World-Builder no-op, a `require_fact` tool call, then (after responding) a plain
  reply and `report_pass`; start consuming the SSE stream; assert a `sidechannel` event
  with `type == "require_fact"`, `path`, `reason`, and `suggested_value` all present,
  arrives before any `done` event, and that `done` has not yet appeared at the point
  the sidechannel is observed.
- `test_require_fact_accept_resumes_stream_with_value` — after observing the
  sidechannel, `POST .../require-fact/respond` with `{"value": "Sarah"}`; assert the
  eventual `message` event's `content` reflects the round-2 mocked reply, and the
  stream ends with `done`.
- `test_require_fact_accept_writes_fact_to_db` — same flow; after the stream
  completes, `get_facts(db, character.id)["Character"]["Identity"]["Name"]["Value"]
  == "Sarah"`.
- `test_require_fact_dismiss_resolves_with_none` — `POST .../respond` with `{}`
  (empty body, `value` defaults to `None`); stream still completes with `message` +
  `done`; `get_facts()` has no `Name` value afterward.
- `test_require_fact_keep_alive_during_suspension` — at least one `: ping` raw line
  observed between the sidechannel event and the respond POST, proving the SSE
  connection survives indefinite suspension exactly as the PoC proved for the
  synthetic endpoint.
- `test_require_fact_gate_removed_after_turn_completes` — `import
  memories.services.tool_gate as tool_gate_module`; after a full accept cycle,
  `(session_id, turn_id) not in tool_gate_module._pending` (`turn_id` is `1`, the
  first turn in a fresh session created by the test's own fixture).
- `test_require_fact_respond_unknown_turn_returns_404` — `POST` to
  `/api/sessions/{session_id}/turns/9999/require-fact/respond` with no matching gate
  returns 404 (this endpoint's behaviour is unchanged from Step 0b, but worth a
  regression check now that it serves real traffic).

Each test creates its own `character`/`session` via `create_character()`/
`create_session()` against the shared `db` fixture (same pattern as the deleted PoC
file's `session_id` fixture), since `live_client` does not depend on the integration
suite's `character`/`session` fixtures (those assume the `ASGITransport`-backed
`client` fixture's dependency overrides).

---

### `tests/unit/test_tool_gate.py` — no changes

Already covers `tool_gate.py` directly and the generic `chat_with_tools()` +
gate integration via a synthetic tool. Nothing in Step 6 changes `tool_gate.py`'s
behaviour or `chat_with_tools()`'s gate-agnostic contract.

---

## Edge Cases

- **`await_gate()` raising `KeyError` when no gate exists is the safety net for
  `implication.py`'s regeneration path, not a bug to fix.** `accept_implication()`
  calls `run_contradiction_loop()` for an already-completed turn whose gate was
  created and torn down inside the *original* `run_turn()` call. If the regenerated
  Character LLM calls `require_fact` there, `await_gate()` raises `KeyError`
  immediately (the dict lookup happens before any `await`), and
  `chat_with_tools()`'s existing `except Exception as exc: result = f"Error: {exc}"`
  handler-exception wrapping (unchanged since Step 0) converts it into a tool error
  the model receives and can react to, instead of crashing the HTTP request. No new
  code is needed for this — it falls out of infrastructure that already exists for
  unrelated reasons (handler robustness against arbitrary exceptions).
- **A single per-turn gate supports multiple sequential `require_fact` calls within
  the same turn, but not concurrent ones.** `asyncio.Queue(maxsize=1)` is empty again
  immediately after a `get()`, so a second `require_fact` call in a later round (or
  even the same round, processed sequentially by `chat_with_tools()`'s `for call in
  tool_calls:` loop) can `await_gate()` again and be resolved by a second `/respond`
  POST. If the model somehow batches two `require_fact` calls in the *same* round
  (against its own tool-list cardinality — Ollama tool-call batching is per
  invocation, not per tool, so this is possible even with one tool registered), the
  first call's sidechannel card and gate-wait block the second call's handler
  invocation entirely until the first is resolved, because `chat_with_tools()`
  processes `tool_calls` sequentially, not concurrently. The user would see one
  blocking card, resolve it, and only then see the second. This is correct but
  untested directly in this step beyond
  `test_run_turn_character_two_sequential_require_fact_calls_same_turn`, which covers
  the cross-round case; the same-round case is mechanically identical from the
  gate's perspective and is not separately exercised.
- **`require_fact` coercion failures on user-supplied values are silently
  best-effort, not hard errors.** Unlike `set_fact`/`author_set_facts`, where an
  invalid `Enum`/`Integer` value comes from the *model* and a hard error lets it
  retry, a `require_fact` value comes from the *user* via a one-shot HTTP POST with no
  retry loop available at this layer (Step 8's type-aware UI controls — dropdown for
  `Enum`, numeric input for `Integer` — are the real fix, preventing invalid input at
  the source). Step 6 stores the raw string verbatim with a logged warning rather than
  dropping the user's explicit answer or leaving the path permanently unset.
- **`require_fact` is intentionally restricted to `Immutable` paths.** A model that
  calls it on a `Mutable` or `Fluid` path receives a guiding error rather than a
  silent no-op, on the theory that this is a model mistake (the tool's description
  and the `_WORLD_STATE_PREAMBLE` already only ever direct the model to call it for
  unset Immutable values) worth surfacing rather than swallowing.
- **`run_turn()`'s `create_gate()` call can itself raise `ValueError`** if a gate for
  the same `(session_id, turn_id)` somehow already exists — e.g. two concurrent
  `send_message` requests racing on the same session computing the same `turn_id` via
  `next_turn_id()`'s unsynchronized `SELECT MAX(turn_id)`. This is a pre-existing
  concurrency gap in `next_turn_id()` unrelated to this step (no locking exists today
  for concurrent turns on one session) and is not newly introduced by Step 6; the
  `ValueError` simply propagates out of `run_turn()` the same way any other
  unexpected exception does today, surfacing as a task exception in
  `routers/chat.py`'s `_stream()` generator. Out of scope to fix here.
- **`thinking` accumulation across a `require_fact` suspend/resume cycle includes
  thinking from before *and* after the suspension, concatenated with no separator or
  marker.** If the model thinks once before calling `require_fact` and again after
  resuming, `ToolCallResult.thinking` joins both fragments indistinguishably — exactly
  matching `chat()`'s pre-existing behaviour for ordinary multi-chunk streaming, which
  this step does not change or improve upon.

---

## Post-Implementation Cleanup Tasks
