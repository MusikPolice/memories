# Step 5 — Character Evaluator Tool-Call Loop

## Overview

Step 5 rewrites the Character Evaluator from a single JSON-verdict Ollama call into a
tool-calling loop, mirroring the World Builder's Step 4 conversion but with a narrower,
mutability-aware tool list. Today `run_evaluator()` sends one `ollama.chat(..., format="json")`
request and parses a `{"verdict": ..., "new_inferences": [...], "violations": [...],
"decision_log": ...}` blob. After this step, `run_evaluator()` drives
`ollama.chat_with_tools()` against four tools — `set_fact`, `propose_inference`,
`report_contradiction`, `report_pass` — and constructs the same `EvaluatorResult` shape
from whichever tools were called, rather than from parsed JSON fields.

This collapses the evaluator's verdict vocabulary from four values
(`pass`/`contradiction`/`new_inference_logical`/`new_inference_probabilistic`) to two
(`pass`/`contradiction`). Inferences are no longer a verdict — `propose_inference` is a
tool that writes directly to the `inferences` table the moment it is called, exactly
once, with no approval gate and no distinction between "logical" (auto-promoted) and
"probabilistic" (held for user review). This retires the auto-promote block in
`chat_service.run_turn()` and the entire `new_inference_probabilistic` review flow's
*production* path (the `/accept-inference` and `/ignore-inference` endpoints in
`implication.py` are left in place — Step 5 does not remove API surface — but nothing
in the new evaluator ever produces a verdict that would route a response to them).

`set_fact` is mutability-gated, but only the `Fluid` branch is fully implemented this
step: a fluid call writes immediately, logs a decision, and emits a quiet
`fact_update_fluid` sidechannel notification, exactly mirroring how `author_set_facts`
behaves in the World Builder. A call against an `Immutable` path that is **already
set** returns a tool error instructing the model to call `report_contradiction` instead
— this requires no user-approval plumbing and is implemented now. Calls against an
unset `Immutable` path or any `Mutable` path return a stub tool error
("approval flow not yet implemented") — these are completed in Step 7, once the
`asyncio.Queue` suspension mechanism lands in Step 6.

Because the model is expected to always conclude with a terminal tool
(`report_pass` or `report_contradiction`) and never with plain text, the generic
`chat_with_tools()` loop in `ollama_client.py` needs a way to stop as soon as one of
those tools is called, rather than looping until the model emits plain content (which it
never will). This step adds an optional `terminal_tools` parameter to
`chat_with_tools()` and a `terminal_call` field to `ToolCallResult`, fully backward
compatible with the World Builder's existing (terminal-tool-agnostic) usage.

**Success criterion:** all tests in `tests/unit/test_tool_calling.py`,
`tests/unit/test_evaluator_service.py`, `tests/unit/test_chat_service.py`, and the
mechanically-updated integration files (`test_api_chat.py`, `test_api_decisions.py`,
`test_api_implication.py`) pass, along with every other existing test. The dev server
starts cleanly; a turn where the character implies a `Fluid` fact change (e.g. a mood
shift) results in that fact appearing in `character_facts` and a `fact_update_fluid`
sidechannel event, with no JSON-verdict request ever sent to Ollama for the evaluator
pass.

---

## What Steps 1–4 Delivered

- `src/memories/fact_schema.json` + `schema_loader.py` — `load_schema()`,
  `apply_mask()`, `check_write_permitted(path, schema) -> str` (returns the leaf's
  `Mutability` string, e.g. `"Fluid"`/`"Mutable"`/`"Immutable"`; raises `ValueError` for
  unknown paths or groupings — the message is already model-readable, e.g. *"Unknown
  schema path: ... You may only write to paths listed in the Fact Schema."*),
  `render_schema_for_prompt()`, `render_current_fact_values(facts_blob, schema=None) ->
  str` (Step 4 addition — renders every *populated* leaf as `path: "value"
  [Mutability]`, trailing with `(all other schema paths are unset)`), `_collect_leaves(node,
  prefix="") -> list[tuple[str, dict]]` (private but already cross-module imported by
  `evaluator.py` and `world_builder.py`).
- `database.py` — `character_facts` table; `get_facts(db, character_id) -> dict`
  (schema-masked); `set_facts(db, character_id, blob) -> None` (full-blob replace, used
  via the `set_facts as db_set_facts` import alias in `world_builder.py`); `patch_fact()`
  (unused by the World Builder or Evaluator — both replace the whole blob via
  `set_facts`). `decisions` table has `pass_name`, `tool_name`, `tool_args` (dict, JSON
  column), `user_input` (dict or `None`); `store_decision(db, *, character_id,
  session_id, turn_id, pass_name, tool_name, tool_args, user_input=None) -> Decision`.
  `create_inference(db, *, character_id, statement, derivation, source_fact_ids=None,
  source_inference_ids=None, source_fact_paths=None, depth=1, inference_type="logical")
  -> Inference`. `get_inferences(db, character_id, status="active") -> list[Inference]`.
- `services/ollama_client.py` — `chat_with_tools(model, messages, tools, tool_handlers,
  max_rounds=MAX_TOOL_CALL_ROUNDS) -> ToolCallResult`. Non-streaming (`stream: false`)
  loop: POSTs `/api/chat`, reads `data["message"]`, strips `thinking` before appending
  to history, and either (a) returns immediately if `tool_calls` is empty/`None` (plain
  content) or (b) calls each handler in `msg["tool_calls"]`, appends a `{"role": "tool",
  "content": result, "tool_call_id": call_id}` message per call, and loops. Handler
  exceptions are caught internally and turned into `f"Error: {exc}"` tool results — the
  model sees them and can retry; handlers themselves do not need defensive try/except.
  `ToolCallResult` fields today: `content: str`, `history: list[dict]`, `rounds: int`,
  `cap_reached: bool`. `MAX_TOOL_CALL_ROUNDS` (module constant, env-overridable, default
  10). `ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]`.
- `services/sse_events.py` — `SSEEvent(event: str, data: dict = {})` with `.to_sse() ->
  str` (calls `json.dumps(self.data)` directly — callers must pre-serialize any
  non-JSON-native values, e.g. `model_dump(mode="json")` for Pydantic models with
  `datetime` fields); `EventCallback = Callable[[SSEEvent], Awaitable[None]] | None`.
- `services/world_builder.py` — `run_world_builder(db, character, session_id, turn_id,
  user_message, facts_blob, inferences, ollama, on_event=None,
  max_rounds=MAX_TOOL_CALL_ROUNDS) -> dict[str, Any]`. Establishes the pattern this step
  follows closely: a module-level tool-schema dict constant, a `_set_leaf()` helper that
  mutates a `copy.deepcopy()`'d `working_blob`, an inline per-entry coercion block
  (case-insensitive `Enum` match returning one replacement hint on failure, `int()`
  coercion for `Integer`), and a handler that calls `db_set_facts()` +
  `store_decision()` + `on_event(...)` synchronously, before returning its result
  string — i.e. side effects happen *inside* the handler, not deferred to a value the
  caller post-processes. `author_set_facts` is scoped to the World Builder's own tool
  list; it is never passed to the Character LLM or Character Evaluator.
- `services/evaluator.py` (pre-Step-5 state) — `build_evaluator_prompt(character,
  facts_blob, user_message, character_response, contradiction_hints=None,
  inferences=None, experiences=None) -> str` renders `## Fact Schema` (via
  `render_schema_for_prompt()`) and `## Current Fact Values` (via
  `render_current_fact_values()`), then a JSON-output task description.
  `run_evaluator(character, facts_blob, user_message, character_response, ollama,
  contradiction_hints=None, inferences=None, experiences=None) -> EvaluatorResult` calls
  `ollama.chat(model, messages, think=False, format="json")`, strips markdown fences,
  `json.loads()`s the content, validates `verdict` against `_VALID_VERDICTS =
  frozenset({"pass", "contradiction", "new_inference_logical",
  "new_inference_probabilistic"})`, force-overrides to `"contradiction"` if any
  violation has `type == "contradiction"`, and `EvaluatorResult.model_validate(data)`s
  the rest. `EvaluatorParseError` wraps both JSON-decode failures and
  `ValidationError`/unknown-verdict failures. Pydantic models: `NewInference
  (inference_type, statement, derivation, source_fact_paths: list[str] = [],
  source_inference_ids: list[int] = [])`, `Violation(type, description)`,
  `ContradictionNotification(iteration, description)`, `EvaluatorResult(verdict,
  new_inferences=[], violations=[], decision_log, contradiction_notifications=[],
  max_retries_exceeded=False)`.
- `services/chat_service.py` (pre-Step-5 state) — `run_contradiction_loop(model,
  base_messages, character, facts_blob, user_content, ollama, think=False,
  max_retries=MAX_CONTRADICTION_RETRIES, inferences=None, experiences=None,
  on_event=None) -> tuple[str, str, EvaluatorResult]` retries the character LLM +
  evaluator pair, accumulating `contradiction_hints`/`contradiction_notifications` from
  `ev.violations` where `type == "contradiction"`. `run_turn()` calls it, then if
  `eval_result.verdict == "new_inference_logical"`, loops `eval_result.new_inferences`,
  computes depth via `compute_depth()`, and calls `create_inference()` for each one
  under the depth cap — this is the block Step 5 deletes. `run_turn()` then
  unconditionally calls `store_decision(..., pass_name="character_evaluator",
  tool_name="evaluator_verdict", tool_args={"verdict": eval_result.verdict},
  user_input=None)` — this call is also deleted (superseded by per-tool-call logging
  inside the new `run_evaluator()`).
- `routers/implication.py` — `accept_implication()` calls `run_contradiction_loop(model,
  history_messages, character, facts_blob, user_text, ollama,
  max_retries=MAX_CONTRADICTION_RETRIES, inferences=inferences)` to regenerate after a
  user edits/accepts an implied fact, then calls its own `store_decision(...,
  tool_name="evaluator_verdict", ...)`. Both call sites change in this step (mechanical
  argument addition + one deletion); the rest of the endpoint (`_AcceptImplicationBody`,
  Fact creation, message replacement) is untouched.
- `routers/chat.py` — reads `eval_result.verdict`, `.violations`, `.new_inferences`,
  `.contradiction_notifications`, `.max_retries_exceeded` to build SSE payloads. The
  `if eval_result.verdict in ("implication", "new_inference_probabilistic"):` branches
  (lines 104, 112) were already partially dead since Step 3 (`implication` half) and
  fully reachable via `new_inference_probabilistic` until now. Per the existing
  project convention (Step 3's `implication` branch, Step 4's
  `extraction_applied`/`implicit_fact_proposed` cards), dead branches are **left in
  place** until the plan's explicitly-scheduled cleanup step — here, Step 7 — rather
  than removed opportunistically.

---

## What This Step Does NOT Change

- **The World Builder.** `world_builder.py`, `author_set_facts`, and its handler are
  untouched. The World Builder still has unrestricted write authority and still runs
  before the Character LLM.
- **The Character LLM call.** `run_contradiction_loop()` still invokes
  `ollama.chat(model, messages, think=think)` with no `tools` argument for the character
  itself. `require_fact` is Step 6.
- **Mutable and unset-Immutable `set_fact` approval flows.** Both return a stub tool
  error this step. The `asyncio.Queue`-based suspension mechanism that lets them hold a
  response pending user input does not exist until Step 6; the approval branching logic
  itself is Step 7.
- **`/accept-inference`, `/ignore-inference`, `/accept-implication`,
  `/ignore-implication` endpoints in `implication.py`.** Their bodies, Pydantic models,
  and Fact/Inference-creation logic are unchanged (only the `run_contradiction_loop()`
  call-site arguments and the redundant decision-log call change — see Detailed Design).
  Removing these endpoints is explicitly Step 7's job per `docs/plan-v2.md`.
- **`routers/chat.py`.** Not edited at all. Its dead `("implication",
  "new_inference_probabilistic")` checks become *fully* dead this step (previously only
  half-dead) but removal is Step 7's job, matching how Step 4 left
  `extraction_applied`/`implicit_fact_proposed` cards in place.
- **`MAX_INFERENCE_DEPTH`/`compute_depth()` for the per-turn evaluator path.** The eager
  pass (`run_eager_pass()`, `inference_service.py`) keeps using both unchanged. The
  per-turn `propose_inference` tool does **not** accept `source_inference_ids`
  (see Detailed Design) and therefore never invokes `compute_depth()` — its inferences
  are always written at `depth=1`. This is an intentional simplification, not an
  oversight; see Edge Cases.
- **The frontend.** `chat.js`'s `buildNotificationFromSidechannel()` returns `null` for
  unrecognised `payload.type`, so the two new sidechannel types this step introduces
  (`fact_update_fluid`, `inference_proposed`) ship inert, exactly like
  `world_builder_applied` in Step 4. Wiring them into the UI is Step 8.
- **`CLAUDE.md` / `README.md`.** Step 10.
- **`tool_gate.py`, `require_fact.py`, `test_poc.py`, `docs/streaming-plan.md`.** No
  Step 6 prerequisite work happens here.

---

## Detailed Design

### Part A — `services/ollama_client.py` — `terminal_tools` and `terminal_call`

**File:** `src/memories/services/ollama_client.py`

The evaluator's tools always end in exactly one terminal call
(`report_pass`/`report_contradiction`), but `chat_with_tools()`'s only stopping
condition today is "the model returned plain content with no `tool_calls`" — something
the evaluator's system prompt explicitly tells the model never to do. Without a way to
recognise a terminal tool, the loop would run until `max_rounds`, re-invoking the model
once for every round even after it has already told the evaluator how to conclude.

**`ToolCallResult` — add one field:**

```python
# Before
@dataclass
class ToolCallResult:
    content: str  # final plain-text response from the model
    history: list[dict[str, Any]]  # full message history including all tool turns
    rounds: int  # number of HTTP round-trips made
    cap_reached: bool  # True if the loop was cut short by max_rounds

# After
@dataclass
class ToolCallResult:
    content: str  # final plain-text response from the model
    history: list[dict[str, Any]]  # full message history including all tool turns
    rounds: int  # number of HTTP round-trips made
    cap_reached: bool  # True if the loop was cut short by max_rounds
    # Set when a round contains a call to a tool named in chat_with_tools()'s
    # terminal_tools argument. {"name": str, "arguments": dict, "result": str} for
    # the first such call in that round; None if terminal_tools was not passed or no
    # terminal tool was called before the loop ended (plain content or cap).
    terminal_call: dict[str, Any] | None = None
```

**`chat_with_tools()` — add the parameter and the post-round check:**

```python
# Before
async def chat_with_tools(
    self,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, ToolHandler],
    max_rounds: int = MAX_TOOL_CALL_ROUNDS,
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
) -> ToolCallResult:
```

Update the docstring to mention the new parameter: *"If `terminal_tools` is given, the
loop returns as soon as a round contains a call to one of those tool names —
`ToolCallResult.terminal_call` carries that call's name, arguments, and result. Pass
`None` (the default) to preserve the original behaviour of looping until the model
returns plain content or `max_rounds` is exhausted."*

Inside the `while rounds < max_rounds:` loop, the existing `for call in tool_calls:`
block (lines ~222–244) gains a `terminal_call` local that is populated on the **first**
matching call in the round, and is checked once the `for` loop completes — every call in
the round is still processed and appended to history before the function can return,
so nothing the model called is silently dropped:

```python
            terminal_call: dict[str, Any] | None = None
            for call in tool_calls:
                fn: dict[str, Any] = call["function"]
                name: str = fn["name"]
                args: dict[str, Any] = fn["arguments"]
                call_id: str | None = call.get("id")
                handler = tool_handlers.get(name)
                if handler is None:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        result = await handler(args)
                    except Exception as exc:
                        result = f"Error: {exc}"
                log.debug(
                    "chat_with_tools tool_result id=%s name=%s result=%r", call_id, name, result
                )
                tool_msg: dict[str, Any] = {"role": "tool", "content": result}
                if call_id is not None:
                    tool_msg["tool_call_id"] = call_id
                history.append(tool_msg)

                if terminal_tools is not None and terminal_call is None and name in terminal_tools:
                    terminal_call = {"name": name, "arguments": args, "result": result}

            if terminal_call is not None:
                log.debug(
                    "chat_with_tools terminal_call=%s rounds=%d", terminal_call["name"], rounds
                )
                return ToolCallResult(
                    content=content,
                    history=history,
                    rounds=rounds,
                    cap_reached=False,
                    terminal_call=terminal_call,
                )

        log.debug("chat_with_tools cap_reached rounds=%d", rounds)
        return ToolCallResult(content=content, history=history, rounds=rounds, cap_reached=True)
```

`content` at the point of the early return is `""` (no plain-content branch was taken
that round) — callers that care about the terminal tool's payload read
`result.terminal_call["arguments"]`, not `result.content`.

**Backward compatibility.** `terminal_tools` defaults to `None`; when `None`, the new
`if terminal_tools is not None and ...` guard is always `False`, so `terminal_call`
stays `None` and the function's control flow is byte-for-byte identical to today for
every existing caller (`world_builder.py`, and every existing test in
`test_tool_calling.py` and `test_world_builder.py`). No existing test changes as a
result of this Part.

---

### Part B — `services/evaluator.py` — full tool-call rewrite

**File:** `src/memories/services/evaluator.py`

#### Imports

```python
# Before
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from memories.models import Character, Experience, Inference
from memories.schema_loader import render_current_fact_values, render_schema_for_prompt
from memories.services.ollama_client import OllamaClient

# After
from __future__ import annotations

import copy
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

_log = logging.getLogger(__name__)
```

`json` and `ValidationError` are no longer used — there is no JSON to parse. The
`set_facts as db_set_facts` alias and the `_collect_leaves`/`check_write_permitted`
imports mirror `world_builder.py`'s existing style exactly.

#### Delete `_VALID_VERDICTS`

```python
_VALID_VERDICTS = frozenset(
    {
        "pass",
        "contradiction",
        "new_inference_logical",
        "new_inference_probabilistic",
    }
)
```

Nothing validates a raw verdict string anymore — the verdict is derived from which
terminal tool fired, not parsed from text. Delete the constant in full.

#### Keep `EvaluatorParseError`, `NewInference`, `Violation`, `ContradictionNotification`, `EvaluatorResult` — unchanged

```python
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
```

None of these models change shape. `NewInference`/`EvaluatorResult.new_inferences` are
kept even though `run_evaluator()` never populates `new_inferences` again — deleting the
field would make `routers/chat.py` line 117
(`"new_inferences": [i.model_dump() for i in eval_result.new_inferences]`) and
`routers/implication.py` line 169 (same pattern) fail `mypy --strict` on an attribute
that no longer exists, even though both lines sit inside an `if` branch that is now
permanently unreachable. Leaving the field in place keeps those two files type-checking
without editing them, consistent with deferring their cleanup to Step 7.
`EvaluatorParseError` is likewise kept for `chat_service.py`'s `except
EvaluatorParseError:` clause to remain valid, even though nothing in the new
`run_evaluator()` raises it (there is no JSON-decode or schema-validation failure mode
left to wrap — see Edge Cases).

#### Tool schema constants — new

```python
_TERMINAL_TOOLS = frozenset({"report_pass", "report_contradiction"})

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
```

#### Small private helpers — new

Duplicated from `world_builder.py`'s `_set_leaf` (not imported — it is private to that
module and the two are simple enough that copying is clearer than reaching across a
module boundary for a 7-line function), plus one new read-only counterpart:

```python
def _set_leaf(blob: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node = blob
    for key in parts[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[parts[-1]] = {"Value": value}


def _lookup_leaf_value(blob: dict[str, Any], path: str) -> Any | None:
    node: Any = blob
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node.get("Value") if isinstance(node, dict) else None
```

#### `build_evaluator_prompt()` — task description rewritten, signature unchanged

```python
def build_evaluator_prompt(
    character: Character,
    facts_blob: dict[str, Any],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
```

Signature is unchanged. Everything through the `## Previously Flagged Contradictions`
block (current lines 70–98) is unchanged. Replace the `## Your Task` block (current
lines 100–146) — delete the JSON-output instructions and the four-verdict priority
list, replace with tool-calling instructions:

```python
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
```

#### `run_evaluator()` — full rewrite

```python
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
            coerced: Any = value
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
                    coerced = int(value)
                except (TypeError, ValueError):
                    return f"Error: {path}: {value!r} is not a valid integer"
            _set_leaf(working_blob, path, coerced)
            await db_set_facts(db, character.id, working_blob)
            await store_decision(
                db,
                character_id=character.id,
                session_id=session_id,
                turn_id=turn_id,
                pass_name="character_evaluator",
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
            return f"Wrote {path} = {coerced!r}."

        if mutability == "Immutable":
            current = _lookup_leaf_value(working_blob, path)
            if current is not None:
                return (
                    f"Error: {path} is Immutable and already set to {current!r}. "
                    "You may not change it. If the character's response conflicts with "
                    "this, call report_contradiction instead."
                )
            return (
                f"Error: {path} is Immutable and unset. The approval flow for unset "
                "Immutable facts is not implemented yet — leave this path alone."
            )

        # Mutable
        return (
            f"Error: {path} is Mutable. The approval flow for Mutable facts is not "
            "implemented yet — leave this path alone."
        )

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
            pass_name="character_evaluator",
            tool_name="propose_inference",
            tool_args={
                "statement": statement,
                "derivation": derivation,
                "source_paths": source_paths,
            },
            user_input=None,
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
            pass_name="character_evaluator",
            tool_name="report_contradiction",
            tool_args={"description": description},
            user_input=None,
        )
        return "Contradiction recorded."

    async def _handle_report_pass(_args: dict[str, Any]) -> str:
        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_evaluator",
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
        model, messages, _EVALUATOR_TOOLS, handlers, max_rounds=max_rounds,
        terminal_tools=_TERMINAL_TOOLS,
    )

    if result.terminal_call is None:
        # The model exhausted max_rounds (or returned plain content) without ever
        # calling a terminal tool. Give it exactly one more round with an explicit
        # instruction, outside the original round budget, before giving up.
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
            model, nudged_history, _EVALUATOR_TOOLS, handlers, max_rounds=1,
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
            ),
            working_blob,
        )

    return (
        EvaluatorResult(
            verdict="pass", decision_log="Response is consistent with established facts."
        ),
        working_blob,
    )
```

Note the **return type change**: `run_evaluator()` now returns `tuple[EvaluatorResult,
dict[str, Any]]`, not bare `EvaluatorResult`. This is required so that
`run_contradiction_loop()` can thread Fluid writes made in one contradiction-retry
attempt into the next attempt's `working_blob` (see Part C) — without it, a second
`run_evaluator()` call would `copy.deepcopy()` the *original* pre-turn `facts_blob` and
a full-blob `db_set_facts()` write from that attempt would silently discard any Fluid
fact written during the first attempt.

---

### Part C — `services/chat_service.py`

**File:** `src/memories/services/chat_service.py`

#### Imports — remove three, no additions

```python
# Remove from the `from memories.database import (...)` block:
    create_inference,
    store_decision,

# Remove the whole line:
from memories.services.inference_service import MAX_INFERENCE_DEPTH, compute_depth
```

Each of `create_inference`, `store_decision`, `MAX_INFERENCE_DEPTH`, and
`compute_depth` is used only in the two blocks deleted below; confirm via `grep -n
"create_inference\|store_decision\|MAX_INFERENCE_DEPTH\|compute_depth"
src/memories/services/chat_service.py` that no other use survives before deleting the
imports.

#### `run_contradiction_loop()` — signature gains `db`, `session_id`, `turn_id`

```python
# Before
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

# After
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
```

`db`/`session_id`/`turn_id` are needed because the evaluator's tool handlers now write
to the DB and log decisions directly — they were previously parameter-free at this
layer because the old evaluator only ever returned data for the caller to act on.

Inside the loop, replace the `run_evaluator()` call and reassign `facts_blob` from its
second return value so later contradiction-retry attempts see Fluid writes from earlier
ones:

```python
# Before
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
            ...

# After
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
            ...
```

`facts_blob` is a parameter of `run_contradiction_loop()`, so reassigning it inside the
`for attempt in range(...)` loop is a normal local-variable rebind — no `nonlocal`
needed, and nothing outside the function observes the rebind (`run_contradiction_loop`'s
own return signature is unchanged: `tuple[str, str, EvaluatorResult]`; no caller reads
the blob back out of it — see Part D and the `implication.py` call site).

#### `run_turn()` — three deletions, one call-site update

**Delete** the auto-promote block in full:

```python
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
```

This is fully superseded by `propose_inference`'s handler writing directly inside
`run_evaluator()` — `eval_result.verdict` can no longer be `"new_inference_logical"`, so
this block's body is unreachable even before deletion.

**Delete** the now-redundant aggregate decision log:

```python
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
```

The evaluator's own `report_pass`/`report_contradiction`/`set_fact`/`propose_inference`
handlers now each log their own decision row, immediately, inside `run_evaluator()`.
Keeping this call would double-log every clean turn (one `report_pass` row from the
handler, one `evaluator_verdict` row from here) and is the reason
`test_get_decisions_after_one_turn`/`test_get_decisions_ordered_by_turn_id_desc` (in
`tests/integration/test_api_decisions.py`) would otherwise start failing on a row-count
mismatch.

**Update** the `run_contradiction_loop()` call site:

```python
# Before
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

# After
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
```

No other part of `run_turn()` changes. The `_log.info("session=%d turn=%d verdict=%s
violations=%d", ...)` call at the end still works unmodified — `eval_result.verdict` and
`eval_result.violations` still exist with the same meaning (just a narrower value
space for `verdict`, and at most one `Violation` in practice).

---

### Part D — `routers/implication.py` — mechanical call-site update

**File:** `src/memories/routers/implication.py`

**Update** the `run_contradiction_loop()` call inside `accept_implication()`:

```python
# Before
    new_content, _, ev = await run_contradiction_loop(
        model,
        history_messages,
        character,
        facts_blob,
        user_text,
        ollama,
        max_retries=MAX_CONTRADICTION_RETRIES,
        inferences=inferences,
    )

# After
    new_content, _, ev = await run_contradiction_loop(
        db,
        session_id,
        turn_id,
        model,
        history_messages,
        character,
        facts_blob,
        user_text,
        ollama,
        max_retries=MAX_CONTRADICTION_RETRIES,
        inferences=inferences,
    )
```

`db`, `session_id`, and `turn_id` are all already in scope as the endpoint's own
parameters — this is a pure argument-list change, no new state to thread in.

**Delete** the redundant decision log immediately after that call, for the same reason
as Part C:

```python
    await store_decision(
        db,
        character_id=session.character_id,
        session_id=session_id,
        turn_id=turn_id,
        pass_name="character_evaluator",  # nosec B106
        tool_name="evaluator_verdict",
        tool_args={"verdict": ev.verdict},
        user_input=None,
    )
```

After this deletion, check whether `store_decision` is still imported/used elsewhere in
`implication.py` (it is not, based on the current file — it has exactly one other
call site, inside `accept_implication()`, which is the block just deleted) and remove
the import if it becomes unused. Run `grep -n "store_decision" src/memories/routers/implication.py`
to confirm before removing the import line.

No other part of `implication.py` changes. `_AcceptImplicationBody`,
`_AcceptInferenceBody`, `accept_inference()`, `ignore_inference()`,
`ignore_implication()`, and the Phase 6 `undo-user-fact`/`accept-implicit-fact`
endpoints are all untouched.

---

## Transitional State After Step 5

After Step 5 and before Step 6:

- The Character Evaluator communicates exclusively via tool calls. No `format="json"`
  request is ever sent for the evaluator pass.
- The evaluator's verdict space is exactly `{"pass", "contradiction"}`. The
  `new_inference_logical`/`new_inference_probabilistic` verdicts can never occur again;
  `chat_service.run_turn()`'s old auto-promote block and the routers' dead
  `("implication", "new_inference_probabilistic")` checks are accordingly unreachable —
  the former is deleted this step, the latter two (`chat.py`, `implication.py`) are
  left in place per the plan's Step 7 cleanup schedule.
- `Fluid` `set_fact` calls apply immediately, write to `character_facts`, log a decision,
  and emit a `fact_update_fluid` sidechannel notification with no frontend handler yet
  (Step 8).
- `set_fact` against an **already-set** `Immutable` path returns a tool error that
  steers the model toward `report_contradiction` — this is final, correct behaviour,
  not a stopgap. `set_fact` against an unset `Immutable` path or any `Mutable` path
  always returns a stub error; no such call can ever succeed until Step 7.
- `propose_inference` writes an `Inference` row immediately on every call, with no
  approval gate, `depth` always `1`, and a quiet `inference_proposed` sidechannel
  notification with no frontend handler yet (Step 8). The eager pass
  (`run_eager_pass()`) is completely unaffected and continues to support multi-hop
  `source_inference_ids` chains and the `MAX_INFERENCE_DEPTH` cap for its own
  inferences.
- The `/accept-inference`, `/ignore-inference`, `/accept-implication`,
  `/ignore-implication` endpoints still exist and still work exactly as before for any
  inference/fact created through other means (the eager pass, or a turn whose
  evaluator pass happened before this step's deploy) — they simply can never be
  triggered by a turn evaluated under the new code, since no verdict they react to can
  occur anymore. Removing them is Step 7.
- The Character LLM still has no tool list at all (`require_fact` lands in Step 6).
- Decision rows for the evaluator pass are now logged per-tool-call rather than once
  per turn. A clean turn still produces exactly one row (`report_pass`); a turn with N
  contradiction retries now produces N `report_contradiction` rows plus one final
  `report_pass`/`report_contradiction` row, plus one row per `set_fact`/
  `propose_inference` call — strictly more granular than before, and more rows per
  retried turn than pre-Step-5 (which always logged exactly one `evaluator_verdict` row
  regardless of retries).

---

## Test Plan

### `tests/unit/test_tool_calling.py` — additions

New tests for the `terminal_tools`/`terminal_call` mechanism, placed in a new
`# Terminal tool behaviour` section. Reuse the file's existing `_SET_FACT_TOOL`,
`_MESSAGES`, `_ok_handler` fixtures; add a tiny terminal tool schema locally:

```python
_REPORT_PASS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "report_pass", "parameters": {"type": "object", "properties": {}}},
}
```

- `test_terminal_tools_none_preserves_existing_behaviour` — a response with no
  `tool_calls` (plain content) and `terminal_tools=None` behaves exactly as today:
  `result.terminal_call is None`, `cap_reached is False`.
- `test_terminal_call_set_when_terminal_tool_invoked` — mock a single
  `make_tool_call_response("report_pass", {})`; call with
  `terminal_tools=frozenset({"report_pass"})`; assert `result.terminal_call ==
  {"name": "report_pass", "arguments": {}, "result": <handler's return value>}` and
  `result.cap_reached is False`.
- `test_terminal_call_stops_loop_without_further_http_calls` — mock only ONE response
  (a terminal tool call) via `respx.post(...).mock(side_effect=[...])` with exactly one
  entry; assert the route's `call_count == 1` after `chat_with_tools()` returns (if the
  loop incorrectly continued, respx would raise `StopIteration`/an assertion error on a
  second, unmocked request).
- `test_terminal_call_processes_other_calls_in_same_round_first` — mock a
  `make_multi_tool_call_response([("set_fact", args), ("report_pass", {})])`; assert
  both the `set_fact` handler ran (via a call-log side effect) and
  `result.terminal_call["name"] == "report_pass"`.
- `test_terminal_call_takes_first_match_when_multiple_in_one_round` — mock a round with
  two terminal-named calls (`report_pass` then a second tool also in `terminal_tools`,
  e.g. a stand-in `report_contradiction`); assert `terminal_call["name"]` is the first
  one in call order.
- `test_non_terminal_tool_calls_continue_looping_as_before` — `terminal_tools` given,
  but the model only ever calls `set_fact` (never a terminal tool) until it returns
  plain content; assert `terminal_call is None` and `result.content` is the plain text,
  matching pre-Step-5 behaviour for a non-evaluator caller that happens to pass
  `terminal_tools` for an unrelated tool name.
- `test_cap_reached_with_terminal_tools_and_no_terminal_call` — `max_rounds=2`, model
  only calls `set_fact` every round; assert `cap_reached is True` and `terminal_call is
  None` (distinguishing "ran out of rounds" from "got a terminal call").

---

### `tests/unit/test_evaluator_service.py` — comprehensive rewrite

#### Module-level fixtures — add

```python
import aiosqlite

from memories.services.sse_events import SSEEvent
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_multi_tool_call_response,
    make_plain_tool_response,
    make_tool_call_response,
)
```

Drop the `make_evaluator_ndjson`, `make_ollama_ndjson` imports (no longer used by this
file) and the `import json` import if nothing else in the file needs it after the
rewrite (check first). `_FACTS_BLOB`, `_CHARACTER`, `_USER_MSG`, `_CHAR_RESPONSE` are
retained unchanged. Add a `db`/`session_id`/`turn_id` fixture trio — this file does not
currently use the `db` fixture from `tests/unit/conftest.py`; add it as a parameter to
every `run_evaluator(...)`-calling test (the shared `db` fixture from
`tests/conftest.py`/`tests/unit/conftest.py` is already available project-wide) and use
literal `session_id=1, turn_id=1` for tests that don't care about their values.

#### `build_evaluator_prompt` tests — mostly retained, two updated

Retained unchanged: `test_evaluator_prompt_includes_all_facts`,
`test_evaluator_prompt_includes_character_response`,
`test_evaluator_prompt_includes_user_message`,
`test_evaluator_prompt_no_facts_uses_fallback_text`,
`test_evaluator_prompt_with_contradiction_hints_lists_them`,
`test_evaluator_prompt_includes_established_inferences`,
`test_evaluator_prompt_no_inferences_uses_fallback`,
`test_evaluator_prompt_includes_inference_ids`,
`test_evaluator_prompt_contains_immutable_contradiction_instruction` (re-check this one
specifically — its assertion text must still match the rewritten `## Your Task` block;
update the literal substring it searches for if needed, e.g. from old wording to *"call
report_contradiction"*), `test_evaluator_prompt_includes_active_experiences`,
`test_evaluator_prompt_no_experiences_uses_fallback`,
`test_evaluator_prompt_includes_experience_ids`,
`test_evaluator_prompt_includes_experience_source_label`,
`test_evaluator_prompt_includes_schema_section`,
`test_evaluator_prompt_includes_immutable_paths`,
`test_evaluator_prompt_includes_fact_values_section`,
`test_evaluator_prompt_renders_populated_path`,
`test_evaluator_prompt_empty_blob_produces_unset_note`.

**Tests to delete** (JSON-output-specific, no longer meaningful):

- `test_evaluator_raises_parse_error_on_implication_verdict` — there is no verdict
  string to parse at all anymore; the model expresses everything via tool calls.
- `test_evaluator_raises_parse_error_on_experience_update_verdict` — same.
- `test_violation_has_no_suggested_fact_field` — unrelated to this step's scope but
  re-verify it still passes; if it constructs `Violation(type=..., description=...,
  suggested_fact=...)` expecting a `ValidationError`/`TypeError`, no change needed since
  `Violation` is untouched.

**Tests to add (prompt content):**

- `test_evaluator_prompt_mentions_set_fact_tool` — `"set_fact"` appears in the rendered
  prompt's task description.
- `test_evaluator_prompt_mentions_propose_inference_tool` — `"propose_inference"`
  appears.
- `test_evaluator_prompt_mentions_report_contradiction_tool` — `"report_contradiction"`
  appears.
- `test_evaluator_prompt_mentions_report_pass_tool` — `"report_pass"` appears.
- `test_evaluator_prompt_instructs_terminal_call` — prompt text instructs concluding
  with exactly one of the two terminal tools (substring match on e.g. `"ALWAYS finish
  by calling exactly one"`).
- `test_evaluator_prompt_no_json_output_instructions` — regression guard: the rendered
  prompt does NOT contain `"Return a JSON object"` or `"\"verdict\":"` (catches an
  incomplete rewrite where old and new task text are accidentally concatenated).

#### `run_evaluator` — request shape

**Tests to delete:**

- `test_evaluator_request_sends_think_false` — `chat_with_tools()`'s payload has no
  top-level `think` key at all; this assertion is no longer meaningful.
- `test_evaluator_request_sends_format_json` — `chat_with_tools()` never sends
  `format`.
- `test_evaluator_raises_parse_error_on_non_json` — there is no JSON response body to
  fail to parse; `chat_with_tools()` reads `data["message"]["tool_calls"]`/`["content"]`
  from a structured response, not raw text.
- `test_evaluator_strips_markdown_code_fence` — no markdown-fence-wrapped JSON exists
  in the new flow.
- `test_evaluator_raises_parse_error_on_unescaped_quote_in_string` — same reasoning as
  the non-JSON test; this exact failure mode (malformed string inside a JSON blob)
  cannot occur when there is no JSON blob.
- `test_evaluator_raises_parse_error_on_missing_verdict` — no `verdict` JSON field
  exists to be missing.
- `test_evaluator_raises_parse_error_on_missing_decision_log` — same.
- `test_evaluator_raises_parse_error_on_unknown_verdict` — same; an unrecognised tool
  name from the model is handled by `chat_with_tools()`'s existing "Unknown tool: ..."
  fallback, not by raising `EvaluatorParseError`.
- `test_evaluator_contradiction_priority_overrides_other_verdict` — there is no
  multi-violation JSON list to prioritise across; `report_contradiction` is itself the
  single terminal call, so there is nothing to "override."
- `test_new_inference_source_fact_paths_are_strings` — `NewInference` is no longer
  populated by `run_evaluator()`; this test asserted JSON-parse coercion behaviour that
  no longer exists. (Coverage for the new `source_paths` plumbing moves to the
  `propose_inference` handler tests below.)

**Tests to add (request shape):**

- `test_evaluator_request_sends_tools_list` — `body["tools"]` contains all four tool
  names (`set_fact`, `propose_inference`, `report_contradiction`, `report_pass`).
- `test_evaluator_request_sends_stream_false` — `body["stream"] is False`.
- `test_evaluator_request_has_no_format_key` — `"format" not in body`.

#### `run_evaluator` — terminal tool mapping

- `test_run_evaluator_report_pass_returns_pass_verdict` — mock
  `make_tool_call_response("report_pass", {})`; `result, _ = await
  run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)`;
  assert `result.verdict == "pass"`.
- `test_run_evaluator_report_contradiction_returns_contradiction_verdict` — mock
  `make_tool_call_response("report_contradiction", {"description": "wrong city"})`;
  assert `result.verdict == "contradiction"` and
  `result.violations[0].description == "wrong city"`.
- `test_run_evaluator_report_contradiction_violation_type_is_contradiction` —
  `result.violations[0].type == "contradiction"`.
- `test_run_evaluator_batched_set_fact_then_report_pass_in_one_round` — mock
  `make_multi_tool_call_response([("set_fact", {...fluid path...}), ("report_pass",
  {})])`; assert `result.verdict == "pass"` and exactly one HTTP call was made
  (`route.call_count == 1`).
- `test_run_evaluator_returns_evaluator_result_type` — retained/renamed from the
  existing test of the same name; assert `isinstance(result, EvaluatorResult)` on the
  first element of the returned tuple.
- `test_run_evaluator_returns_updated_facts_blob` — second element of the returned
  tuple is a `dict`.

#### `_handle_set_fact` — fluid path

- `test_set_fact_fluid_path_writes_to_db` — mock
  `make_multi_tool_call_response([("set_fact", {"path":
  "Character.State-Of-Mind.Mood", "value": "Anxious"}), ("report_pass", {})])`; after
  `run_evaluator`, `get_facts(db, character.id)` contains `Mood == "Anxious"`.
- `test_set_fact_fluid_path_enum_coerced_case_insensitively` — value `"calm"` stored as
  `"Calm"`.
- `test_set_fact_fluid_path_enum_invalid_value_returns_error_and_path_unwritten` — value
  `"Apprehensive"`; the tool result string starts with `"Error:"`; the path is absent
  from the blob returned by `run_evaluator`.
- `test_set_fact_fluid_path_integer_coerced_from_string` — analogous to the World
  Builder's equivalent test, using an `Integer` leaf if one exists at `Fluid`
  mutability; if the current schema has no `Fluid` `Integer` leaf, this test may instead
  target `Character.State-Of-Mind.Energy` (Enum) and a second test added against
  whichever `Mutable`/`Immutable` `Integer` leaf exists purely to exercise the
  type-coercion branch with `mutability` forced via a monkeypatched `check_write_permitted`
  — prefer adding a `Fluid` `Integer` leaf to a **test-local** schema fixture instead of
  monkeypatching; see note below.
- `test_set_fact_fluid_path_logs_decision` — one decision row, `tool_name == "set_fact"`,
  `tool_args == {"path": ..., "value": ...}`.
- `test_set_fact_fluid_path_emits_sidechannel_event` — `on_event` collector receives
  `SSEEvent(event="sidechannel", data={"type": "fact_update_fluid", ...})`.
- `test_set_fact_unknown_path_returns_error` — `check_write_permitted` raises
  `ValueError`; tool result starts with `"Error:"`; no DB write, no decision row.

**Note on schema-dependent fixtures.** Read `src/memories/fact_schema.json` before
writing these tests to confirm which leaves are `Fluid`/`Mutable`/`Immutable` of each
`Type` — do not assume `Character.State-Of-Mind.Mood`/`Energy` are the only `Fluid`
leaves; use whatever the live schema defines, and prefer leaves already used by
`test_world_builder.py`'s equivalent coercion tests for consistency.

#### `_handle_set_fact` — immutable path

- `test_set_fact_immutable_unset_returns_stub_error` — pick an `Immutable` leaf with no
  pre-seeded value (default empty blob); tool result mentions `"not implemented"`; no
  DB write.
- `test_set_fact_immutable_already_set_returns_contradiction_hint` — pre-seed the leaf's
  value via `set_facts()`; call `set_fact` with a *different* value; tool result starts
  with `"Error:"` and contains `"report_contradiction"`.
- `test_set_fact_immutable_already_set_value_unchanged_in_returned_blob` — the blob
  `run_evaluator` returns still has the original value, not the rejected one.

#### `_handle_set_fact` — mutable path

- `test_set_fact_mutable_path_returns_stub_error_regardless_of_current_value` — both an
  unset and a pre-seeded `Mutable` leaf return a `"not implemented"` error; no DB write
  in either case.

#### `_handle_propose_inference`

- `test_propose_inference_writes_inference_row` — mock
  `make_multi_tool_call_response([("propose_inference", {"statement": "...",
  "derivation": "..."}), ("report_pass", {})])`; `get_inferences(db, character.id)`
  contains the statement.
- `test_propose_inference_stored_with_depth_one` — `depth == 1` regardless of how many
  prior inferences exist for the character.
- `test_propose_inference_stores_source_paths` — `source_fact_paths == [...]` matches
  the `source_paths` argument.
- `test_propose_inference_missing_source_paths_defaults_to_empty_list` — call without a
  `source_paths` key at all; stored `source_fact_paths == []`.
- `test_propose_inference_logs_decision` — `tool_name == "propose_inference"`.
- `test_propose_inference_emits_sidechannel_event` — `data["type"] ==
  "inference_proposed"`; `data["inference"]["statement"]` matches.
- `test_propose_inference_multiple_calls_in_one_round_all_written` — a
  `make_multi_tool_call_response` with two `propose_inference` entries plus
  `report_pass`; both inferences present in `get_inferences`.

#### Cap and nudge behaviour

- `test_run_evaluator_nudge_round_fires_when_cap_reached_without_terminal_call` — `
  max_rounds=1` and the model calls only `set_fact` (never terminal) on its first
  round; mock a SECOND response for the nudge round that DOES call `report_pass`;
  assert `result.verdict == "pass"` and the route received exactly 2 calls.
- `test_run_evaluator_nudge_message_instructs_terminal_tool` — inspect
  `route.calls[1].request.content` (the nudge round's request body); the appended
  system message text is present in `body["messages"]`.
- `test_run_evaluator_falls_back_to_pass_when_nudge_also_fails` — `max_rounds=1`; both
  the original round AND the nudge round call only `set_fact`; assert `result.verdict
  == "pass"` and `"cap reached"` in `result.decision_log`.
- `test_run_evaluator_logs_warning_on_cap_fallback` — same setup, asserted via
  `caplog` for a `WARNING`-level record mentioning "tool-call cap reached".
- `test_run_evaluator_no_nudge_round_when_terminal_call_made_within_budget` — normal
  `max_rounds=10`, model calls `report_pass` on round 1; assert exactly one HTTP call
  total (no nudge round attempted).

---

### `tests/unit/test_chat_service.py` — updates

**Module docstring (lines 1–26):** update "calls[2] — evaluator LLM" to clarify it is
now a tool-calling, non-streaming call, and that — like calls[0] — it may be more than
one HTTP request if a test specifically exercises multi-round evaluator behaviour
(`set_fact`/`propose_inference` followed by a terminal call in a *separate* round
rather than batched). State that the default `_mock_eval()` helper used by most tests
remains a single no-op `report_pass()` round, preserving the "three calls total"
assumption.

**Imports:** drop `make_evaluator_ndjson` from the `tests.unit.conftest` import line;
add `make_multi_tool_call_response`.

**Helpers to update:**

- `_mock_eval(verdict="pass", new_inferences=None, violations=None)` — **delete the
  `new_inferences` parameter** (no caller can make it do anything meaningful anymore —
  `propose_inference` is a separate tool call, not embedded in a verdict). Redefine the
  body:

  ```python
  def _mock_eval(
      verdict: str = "pass",
      violations: list[dict] | None = None,
  ) -> httpx.Response:
      if verdict == "contradiction":
          description = (violations or [{}])[0].get("description", "contradiction")
          return httpx.Response(
              200,
              content=make_tool_call_response("report_contradiction", {"description": description}),
          )
      return httpx.Response(200, content=make_tool_call_response("report_pass", {}))
  ```

  Every existing call site of the form `_mock_eval()`, `_mock_eval("pass")`, or
  `_mock_eval("contradiction", violations=contradiction_violation)` (where
  `contradiction_violation` is `[{"type": "contradiction", "description": "...",
  "suggested_fact": None}]`) continues to work unchanged — `violations[0]["description"]`
  is still the field being read; the now-meaningless `"type"`/`"suggested_fact"` keys in
  those literals are simply ignored rather than round-tripped through Pydantic. Grep for
  `_mock_eval(` and confirm no call site passes `new_inferences=` before relying on this
  (the two that do are deleted below).
- `_mock_turn(character_content="...", evaluator_verdict="pass", new_inferences=None,
  violations=None)` — **delete the `new_inferences` parameter**; its body's call to
  `_mock_eval(evaluator_verdict, new_inferences, violations)` becomes
  `_mock_eval(evaluator_verdict, violations)`.

**Tests to delete** (depend on the removed auto-promote/depth-cap block or the removed
`new_inference_*` verdicts):

- `test_new_inference_logical_creates_inference_row`
- `test_new_inference_probabilistic_does_not_create_db_row`
- `test_lazy_inference_depth_computed_before_storing`
- `test_lazy_inference_at_max_depth_is_stored`
- `test_lazy_inference_exceeding_depth_cap_not_stored`
- `test_run_turn_falls_back_to_pass_on_evaluator_parse_error` — its premise (malformed
  NDJSON text causing `json.JSONDecodeError`) cannot occur in the tool-calling model;
  there is no equivalent "evaluator returned garbage" failure mode worth testing at
  this layer (the cap/nudge-fallback tests in `test_evaluator_service.py` already cover
  "model never cooperates").

**Tests to update:**

- `test_pass_verdict_decision_stored` — change the assertion from
  `decisions[0].tool_args["verdict"] == "pass"` to
  `decisions[0].tool_name == "report_pass"` (and optionally
  `decisions[0].tool_args == {}`). `len(decisions) == 1` is unchanged.
- All contradiction-retry tests using a literal
  `contradiction_violation = [{"type": "contradiction", "description": "...",
  "suggested_fact": None}]` dict (`test_contradiction_response_not_stored_on_first_attempt`,
  `test_contradiction_triggers_second_character_call`,
  `test_contradiction_second_call_messages_include_system_note`,
  `test_contradiction_loop_exits_on_pass`,
  `test_contradiction_loop_final_response_is_stored`,
  `test_contradiction_max_retries_exceeded_delivers_anyway`,
  `test_contradiction_notifications_collected_per_iteration`) — **no change needed**
  beyond the `_mock_eval()` helper redefinition above; their `_mock_eval("contradiction",
  violations=contradiction_violation)` calls keep working because `_mock_eval` still
  reads `violations[0]["description"]`. Verify each still passes after the helper
  change rather than editing them.
- `test_evaluator_called_with_inferences` — no change; `route.calls[2]` is still the
  evaluator's first (and, for a no-op pass, only) HTTP request, and its
  `body["messages"][1]["content"]` is still the rendered evaluator prompt containing
  established-inference text.

**Tests to add:**

- `test_run_turn_evaluator_fluid_set_fact_writes_to_blob` — mock
  `_CHAT_URL` side effects: `_mock_world_builder()`, `_mock_ok(...)`, then
  `httpx.Response(200, content=make_multi_tool_call_response([("set_fact",
  {"path": "<a Fluid leaf>", "value": "<value>"}), ("report_pass", {})]))`; after
  `run_turn`, `get_facts(db, character.id)` contains the written value.
- `test_run_turn_evaluator_fluid_set_fact_emits_sidechannel` — pass an `on_event`
  collector into `run_turn`; assert a `fact_update_fluid` sidechannel event was
  received.
- `test_run_turn_evaluator_propose_inference_writes_row` — same pattern with
  `("propose_inference", {"statement": "...", "derivation": "..."})`; assert
  `get_inferences` contains it after `run_turn`.
- `test_run_turn_decisions_no_longer_include_evaluator_verdict_tool_name` — regression
  guard: after any completed turn, no decision row has
  `tool_name == "evaluator_verdict"` (catches a reintroduction of the deleted
  aggregate-logging call).
- `test_run_turn_character_llm_request_has_no_tools_key` — retained from Step 4's
  suite if not already present; otherwise add it: `route.calls[1]`'s (character LLM)
  request body has no `"tools"` key, confirming the Character LLM still cannot see
  `set_fact`/`propose_inference`/etc.

---

### `tests/integration/test_api_chat.py` — updates

**Imports:** drop `make_evaluator_ndjson`; add `make_tool_call_response` (and confirm
`make_multi_tool_call_response` is imported if any new test needs it).

**Helpers to update:**

- `_mock_eval(verdict="pass", new_inferences=None, violations=None)` (local to this
  file, lines ~45–50) — same redefinition as `test_chat_service.py`'s helper: drop
  `new_inferences`, body produces `make_tool_call_response("report_contradiction",
  {...})` or `make_tool_call_response("report_pass", {})`.
- `_mock_turn(...)` (lines ~53–65) — drop the `new_inferences` passthrough parameter.

**Tests to delete:**

- `test_send_message_new_inference_probabilistic_emits_sidechannel` (lines ~442–465) —
  the verdict it mocks can no longer occur.

**Tests to update (mechanical only — no assertion change, just confirm they still pass
after the helper redefinition):** every test listed in the earlier grep that calls
`_mock_turn(...)`/`_mock_eval(...)` without `new_inferences=` needs no edits. Run the
full file after the helper change and fix any genuine failures individually rather than
assuming the list above is exhaustive — this file has the largest call-site count in
the suite and a line-by-line enumeration here would drift from the real file quickly.

**`test_accept_implication_on_high_mutability_fact_preserves_mutability`** (line ~1107)
and the other direct `make_evaluator_ndjson("implication", violations=[_VIOLATION])`
calls inside accept-implication tests (search for `make_evaluator_ndjson("implication"`)
— replace each with `make_tool_call_response("report_pass", {})`. The `"implication"`
argument was already inert before this step (see "What Steps 1–4 Delivered" — the old
evaluator's `EvaluatorParseError` fallback already silently converted it to a `"pass"`
verdict); these calls exist only to produce *some* assistant message at the target
`turn_id` for the endpoint under test to act on, and that purpose is unaffected by the
literal mock content.

**Tests to add:**

- `test_send_message_fact_update_fluid_emits_sidechannel` — mock calls[2] (evaluator)
  with a `set_fact` + `report_pass` batch via `make_multi_tool_call_response`; assert
  the SSE stream contains a `sidechannel` event with `type == "fact_update_fluid"`.
- `test_send_message_inference_proposed_emits_sidechannel` — same pattern with
  `propose_inference`; assert `type == "inference_proposed"`.
- `test_send_message_evaluator_contradiction_still_triggers_regeneration` — confirms the
  end-to-end SSE contradiction flow (existing `contradiction` sidechannel +
  `regenerating`/`reviewing` status pair) still fires correctly when calls[2]'s first
  response is `make_tool_call_response("report_contradiction", {"description": ...})`.

---

### `tests/integration/test_api_decisions.py` — updates

**Imports:** drop `make_evaluator_ndjson`; add `make_tool_call_response`.

**Helper to update:**

```python
# Before
def _mock_turn(character_content: str = "I am fine.") -> list[httpx.Response]:
    return [
        httpx.Response(200, content=make_plain_tool_response("Nothing to extract.")),
        httpx.Response(200, content=make_ollama_ndjson(character_content)),
        httpx.Response(200, content=make_evaluator_ndjson()),
    ]

# After
def _mock_turn(character_content: str = "I am fine.") -> list[httpx.Response]:
    return [
        httpx.Response(200, content=make_plain_tool_response("Nothing to extract.")),
        httpx.Response(200, content=make_ollama_ndjson(character_content)),
        httpx.Response(200, content=make_tool_call_response("report_pass", {})),
    ]
```

**Tests to update:**

- `test_get_decisions_contains_verdict_field` — rename
  `test_get_decisions_contains_tool_name_field`; replace its body's assertions:

  ```python
  # Before
  assert "tool_args" in data[0]
  assert "verdict" in data[0]["tool_args"]

  # After
  assert "tool_name" in data[0]
  assert data[0]["tool_name"] == "report_pass"
  ```

**No other changes.** `test_get_decisions_initially_empty`,
`test_get_decisions_after_one_turn` (`len == 1` — still true: World Builder no-op = 0
rows, `report_pass` handler = 1 row), `test_get_decisions_contains_reasoning_field`,
`test_get_decisions_ordered_by_turn_id_desc` (`len == 2` across two clean turns — still
true for the same reason), and `test_get_decisions_unknown_session_returns_404` all
pass unmodified once the helper above is updated.

---

### `tests/integration/test_api_implication.py` — updates

**Imports:** drop `make_evaluator_ndjson`; add `make_tool_call_response`.

**Helpers to update:**

- `_implication_turn()` (lines ~37–48) and `_pass_turn()` (lines ~51–56) — replace every
  `make_evaluator_ndjson(...)` call (with or without an `"implication"`
  verdict/`violations` argument) with `make_tool_call_response("report_pass", {})`. As
  with `test_api_chat.py`, the specific verdict string passed to the old helper was
  already inert (see above) — these calls exist solely to populate an assistant message
  for the endpoint under test, and `report_pass` is sufficient for that in every case.

**Direct `make_evaluator_ndjson("implication", ...)` calls inline in test bodies**
(search the file for `make_evaluator_ndjson(` — at least 4 occurrences beyond the two
helpers, per the earlier grep) — same replacement: `make_tool_call_response("report_pass",
{})`. Where a test constructs `_VIOLATION`/`v1`/`v2`/`changed_violation` dict literals
purely to pass into `make_evaluator_ndjson(violations=[...])`, the dict itself can stay
(it may be referenced elsewhere, e.g. in an assertion against the *new* fact's expected
shape) but its use as evaluator-mock input is removed.

**After the mechanical replacement, run the full file** and fix any test whose
assertions read `ev.violations`/`ev.new_inferences` expecting the OLD JSON-verdict
shape (e.g. expecting `violations` to carry a `"description"` matching the originally
mocked `_VIOLATION`) — since the replacement responses are always a no-op
`report_pass`, any such assertion was already exercising dead-since-Step-3 behaviour
(see "What Steps 1–4 Delivered") and should be deleted rather than chased into the new
model; if a test genuinely needs to verify the post-regeneration response's
`ev.verdict`, it should construct its OWN dedicated mock for the regeneration call
(e.g. `make_tool_call_response("report_contradiction", {"description": ...})`) rather
than relying on `_pass_turn()`'s shared default.

**No structural changes** to `_AcceptImplicationBody`/`_AcceptInferenceBody` usage,
endpoint paths, or the Fact/Inference assertions themselves — only the evaluator-mock
construction changes.

---

## Files changed by this step

| Action | File | Notes |
|---|---|---|
| Modify | `src/memories/services/ollama_client.py` | Add `terminal_tools` param to `chat_with_tools()`; add `terminal_call` field to `ToolCallResult` |
| Modify | `src/memories/services/evaluator.py` | Full tool-call rewrite: new tool schemas, `_set_leaf`/`_lookup_leaf_value` helpers, rewritten `build_evaluator_prompt()` task text, rewritten `run_evaluator()` (new params, new return type, nudge/cap fallback) |
| Modify | `src/memories/services/chat_service.py` | `run_contradiction_loop()` gains `db`/`session_id`/`turn_id` params and reassigns `facts_blob` from `run_evaluator()`'s second return value; `run_turn()` drops the auto-promote block and the aggregate decision-log call; import cleanup |
| Modify | `src/memories/routers/implication.py` | `run_contradiction_loop()` call site gains three leading args; redundant decision-log call removed; import cleanup if `store_decision` becomes unused |
| Modify | `tests/unit/test_tool_calling.py` | New `terminal_tools`/`terminal_call` test section |
| Modify | `tests/unit/test_evaluator_service.py` | Heavy rewrite — JSON-parsing tests deleted, tool-call tests added |
| Modify | `tests/unit/test_chat_service.py` | Helper redefinition; deletions (auto-promote/depth-cap/parse-error tests); additions (fluid set_fact, propose_inference, decision-shape regression) |
| Modify | `tests/integration/test_api_chat.py` | Helper redefinition; one deletion (`new_inference_probabilistic` sidechannel test); additions (`fact_update_fluid`/`inference_proposed` sidechannel tests) |
| Modify | `tests/integration/test_api_decisions.py` | Helper update; one test renamed/updated |
| Modify | `tests/integration/test_api_implication.py` | Helper + inline mock replacements; case-by-case review of assertions reading the old JSON-verdict shape |

No changes to `src/memories/database.py`, `src/memories/models/__init__.py`,
`src/memories/schema_loader.py`, `src/memories/services/world_builder.py`,
`src/memories/services/sse_events.py`, `src/memories/services/inference_service.py`,
`src/memories/services/experience_service.py`, `src/memories/routers/chat.py`,
`src/memories/routers/facts.py`, `src/memories/routers/decisions.py`, or the frontend.

---

## Edge Cases

- **`EvaluatorParseError` can no longer be raised.** There is no JSON-decode or
  schema-validation step left in `run_evaluator()` for it to wrap. The class definition
  and `chat_service.py`'s `except EvaluatorParseError:` clause are both kept (the
  latter becomes unreachable dead code) purely so neither file needs an unrelated edit
  in this step; flagging it here rather than silently leaving it for `/review-step` to
  discover independently.
- **`propose_inference` never computes a multi-hop depth.** The tool's signature
  (`statement, derivation, source_paths`) has no `source_inference_ids` parameter, so
  every per-turn inference is stored at `depth=1` — `compute_depth()` is never invoked
  for this path. This is a deliberate simplification matching `docs/plan-v2.md`'s
  literal tool signature, not an oversight: multi-hop inference chains remain fully
  supported by the eager pass (`run_eager_pass()`), which still accepts and depth-caps
  `source_inference_ids`. If a later step decides this needs revisiting, it is additive
  (a new optional parameter), not a breaking change to anything built here.
- **Cross-retry blob propagation only matters for `Fluid` writes within Step 5's
  scope.** `run_evaluator()` now returns its `working_blob` so
  `run_contradiction_loop()` can carry Fluid writes forward across contradiction
  retries within one turn. Since contradictions are inherently about `Immutable`
  conflicts (which never mutate the blob — a rejected write never reaches
  `_set_leaf()`), no contradiction-triggered retry in this step's actual behaviour ever
  depends on this propagation; it exists to avoid a latent data-loss bug (an attempt-2
  `set_facts()` full-blob-replace silently erasing attempt-1's Fluid write) and to be
  ready for Step 7, where `Mutable`/`Immutable` writes will also mutate the blob across
  a regenerate-after-edit cycle that this same code path serves.
- **Multiple terminal tool calls in one round.** If the model calls both
  `report_pass()` and `report_contradiction(...)` in the same batched response (against
  its instructions), `chat_with_tools()`'s `terminal_call` is set to whichever appears
  **first** in the model's `tool_calls` list; the second is still executed and logged
  (its decision row exists) but its result is not what `run_evaluator()` acts on. This
  is an accepted, deterministic tie-break — not a validation error — since the model
  was already told never to do this and the project's existing-failure-mode handling
  doesn't special-case it elsewhere either (e.g. World Builder doesn't validate
  "exactly one batched call").
- **The nudge round is a separate `chat_with_tools()` invocation, not an extra
  iteration of the original loop.** This means the nudge round's own internal
  `rounds` counter starts at 0/1 independently of the original call's `rounds` (which is
  already finalized at `max_rounds` by the time the nudge fires) — `docs/plan-v2.md`'s
  "do not count this as another round-trip" is satisfied because the evaluator's
  `MAX_TOOL_CALL_ROUNDS` budget (passed as `max_rounds` to the *first* call) is never
  exceeded by the nudge call, which always passes its own `max_rounds=1`.
- **`report_pass()`/`report_contradiction(...)` always succeed.** Neither handler can
  return an error string — they only log a decision and return a fixed acknowledgement.
  This means once `chat_with_tools()` detects either as a `terminal_call`, the result
  is never re-validated; there is no failure mode analogous to `set_fact`'s coercion
  errors for these two tools, by design (per `docs/plan-v2.md`'s tool table, they take
  no arguments requiring validation other than a free-text `description`).
- **`Enum` coercion error messages stay single-hint, per the project's established
  pitfall.** Mirrors `world_builder.py`'s policy exactly: a failed `set_fact` call
  returns one `Valid values: A, B, C` clause for the failing entry, never a list
  repeated across multiple retries — `_handle_set_fact` returns immediately on the
  first coercion failure rather than collecting a batch of per-entry errors the way
  `author_set_facts` does (the evaluator's `set_fact` is single-entry, unlike the World
  Builder's batched `author_set_facts`).
- **`db_set_facts` writes the entire `working_blob`, not a single-path patch.** Exactly
  like `world_builder.py`, a `Fluid` `set_fact` call writes the full accumulated blob
  via `set_facts()`, not `patch_fact()` — consistent with the project's existing
  "always read/write the complete blob" pattern (see `docs/plan-v2.md`'s "Why a blob is
  sufficient").

---

## Post-Implementation Cleanup Tasks

### CT-1: Integer coercion branch in `_handle_set_fact` is dead code with no test

**Decided:** Fix in follow-up

The `elif leaf["Type"] == "Integer": coerced = int(value)` branch inside the
`if mutability == "Fluid":` block of `_handle_set_fact` in
`src/memories/services/evaluator.py` (lines ~315–319) can never be reached with
the current schema — every Integer leaf (`Character.Identity.Age`,
`User.Identity.Age`, `Setting.Temporal.Current-Year`) is Immutable or Mutable,
not Fluid. The spec listed `test_set_fact_fluid_path_integer_coerced_from_string`
explicitly in the Test Plan and acknowledged the schema gap, directing the
implementer to either add a Fluid Integer leaf to a test-local schema fixture or
monkeypatch `check_write_permitted`. Neither was done. If a Fluid Integer leaf is
added to the schema later, a model could silently write the string `"42"` instead
of the int `42`, corrupting the blob without any test to catch it.

**What to do:**
1. In `tests/unit/test_evaluator_service.py`, add
   `test_set_fact_fluid_path_integer_coerced_from_string` that exercises the
   Integer coercion branch. The cleanest approach is to patch
   `memories.services.evaluator.leaves_by_path` (or monkeypatch
   `check_write_permitted`) to make one path resolve to
   `{"Type": "Integer", "Mutability": "Fluid"}`, then confirm that after calling
   `run_evaluator` with a string `"42"`, the returned blob contains the integer
   `42`, not the string `"42"`.
2. Also add a companion test that confirms an invalid integer string (e.g.
   `"not-a-number"`) returns an error starting with `"Error:"` and leaves the
   blob unchanged.
