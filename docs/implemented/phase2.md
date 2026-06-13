# Phase 2 Implementation Plan — Evaluator Pipeline

## Goals (from plan.md)

- Second LLM call (evaluator) runs after each character response is buffered, before delivery
- Structured JSON verdict from the evaluator; six verdict types handled correctly
- Contradiction loop: response withheld and automatically regenerated until no contradictions remain
- Implication flow: response delivered with ungrounded badge; sidechannel prompts user to establish new Fact; accept/edit triggers server-side regeneration with new Fact in context
- New inference flows: logical inferences auto-promoted to DB; probabilistic inferences surfaced to user with sidechannel prompt
- Decision logged to SQLite for every completed turn
- Reviewing and regenerating indicators shown in the UI while the evaluator runs

**Deliverable:** Every character response is checked against established Facts before delivery. Contradictions never reach the user. Implied Facts and discoverable inferences are surfaced for user review.

---

## Task List

### T1 — Pydantic models

**T1.1 — `Decision` model**
Add to `models/__init__.py`:
```python
class Decision(BaseModel):
    id: int
    character_id: int
    session_id: int
    turn_id: int
    reasoning: str
    verdict: str
    violations: list[dict[str, Any]] | None = None
```

**T1.2 — `Inference` model (minimal)**
Add to `models/__init__.py`. Full use comes in Phase 3; Phase 2 only writes to the table from lazy discovery.
```python
class Inference(BaseModel):
    id: int
    character_id: int
    statement: str
    derivation: str
    source_fact_ids: list[int] = []
    source_inference_ids: list[int] = []
    depth: int
    inference_type: str   # "logical" | "probabilistic"
    status: str           # "active" | "stale" | "invalidated"
```

**T1.3 — `EvaluatorResult` model**
Lives in `services/evaluator.py` (not exported from `models/`; it's an internal service type):
```python
class NewInference(BaseModel):
    inference_type: str          # "logical" | "probabilistic"
    statement: str
    derivation: str
    source_fact_ids: list[int] = []
    source_inference_ids: list[int] = []

class Violation(BaseModel):
    type: str                    # "implication" | "contradiction"
    description: str
    suggested_fact: dict[str, str] | None = None   # {key, value}

class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    decision_log: str
```
`experience_updates` is part of the JSON schema defined in plan.md but is not processed in Phase 2. If the evaluator returns it anyway, treat it as a `pass`.

---

### T2 — Database additions

**T2.1 — Decision repository**
```python
store_decision(
    db, *, character_id, session_id, turn_id, reasoning, verdict,
    violations: list[dict] | None = None
) -> Decision

get_decisions(db, session_id: int) -> list[Decision]
    # ordered by turn_id DESC
```

The `violations` column is stored as a JSON array string. `get_decisions` parses it back to a list.

**T2.2 — Inference repository (minimal)**
```python
create_inference(
    db, *, character_id, statement, derivation,
    source_fact_ids: list[int] = [],
    source_inference_ids: list[int] = [],
    depth: int = 1,
    inference_type: str = "logical"
) -> Inference

get_inferences(db, character_id: int, status: str = "active") -> list[Inference]
```

`source_fact_ids` and `source_inference_ids` are stored as JSON array strings, parsed on read.

**T2.3 — Message update helpers**
Two new functions in `database.py`:

```python
tag_message_ungrounded(
    db, *, session_id: int, turn_id: int, implications: list[dict]
) -> None
    # Sets ungrounded_implications on the assistant message for this turn.
    # implications is a list of {type, description, suggested_fact}.

replace_message_content(
    db, *, session_id: int, turn_id: int, new_content: str
) -> Message
    # Updates content and clears ungrounded_implications on the assistant
    # message for this turn. Used by the accept-implication endpoint after
    # regeneration.
```

Both functions operate on `role='assistant'` for the given `(session_id, turn_id)`.

---

### T3 — Evaluator service

New file: `src/memories/services/evaluator.py`

**T3.1 — Evaluator prompt builder**
```python
def build_evaluator_prompt(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
) -> str
```

Produces a system prompt that:
- Identifies the evaluator's role (fact-checker, not a character)
- Lists the character's Facts as `key: value` pairs
- Quotes the user's message for context
- Quotes the character's response to evaluate
- Specifies the six verdict types with definitions
- Includes the JSON output schema
- If `contradiction_hints` is non-empty (i.e. we are re-evaluating after a previous contradiction attempt), lists each hint so the evaluator knows what was previously flagged

**T3.2 — JSON format schema constant**
A module-level dict in `evaluator.py` describing the expected evaluator JSON output. Passed to `OllamaClient.chat()` via the new `format` parameter:
```python
_EVALUATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "decision_log"],
    "properties": {
        "verdict": {"type": "string"},
        "new_inferences": {"type": "array"},
        "violations": {"type": "array"},
        "decision_log": {"type": "string"},
    },
}
```

**T3.3 — Evaluator call**
```python
class EvaluatorParseError(Exception):
    """Raised when the evaluator returns unparseable or invalid JSON."""

async def run_evaluator(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    ollama: OllamaClient,
    contradiction_hints: list[str] | None = None,
) -> EvaluatorResult
```

Implementation:
1. Build the evaluator system prompt via `build_evaluator_prompt`
2. Call `ollama.chat(model, messages, think=False, format=_EVALUATOR_SCHEMA)`
3. Parse the returned content string as JSON using `json.loads`
4. Validate with `EvaluatorResult.model_validate`; raise `EvaluatorParseError` on failure
5. If `verdict` is `experience_update`, coerce to `pass` (no Experiences in Phase 2)
6. If `verdict` is unknown, raise `EvaluatorParseError`
7. Contradiction priority: if any violation has `type: "contradiction"`, force verdict to `"contradiction"` regardless of what the model returned

The evaluator model is the same as the character model: `character.current_model_name or character.modelfile_base`. Both calls use the same Ollama instance.

---

### T4 — OllamaClient extension

**T4.1 — `format` parameter**
Add optional `format: dict[str, Any] | str | None = None` parameter to `OllamaClient.chat()`. When non-None, include `"format": format` in the request payload. No other changes to the streaming logic — the assembled content will be valid JSON when format is set.

```python
async def chat(
    self,
    model: str,
    messages: list[dict[str, str]],
    think: bool = False,
    format: dict[str, Any] | str | None = None,
) -> tuple[str, dict[str, Any]]:
```

---

### T5 — Chat service update

**T5.1 — New `run_turn` signature**
```python
async def run_turn(
    db: aiosqlite.Connection,
    session_id: int,
    user_content: str,
    ollama: OllamaClient,
    think: bool = False,
) -> tuple[str, str, int, EvaluatorResult]
    # Returns (content, thinking, turn_id, evaluator_result)
```

The router uses `evaluator_result` to decide which SSE events to emit.

**T5.2 — Turn flow**

Updated `run_turn` step sequence:

1. Load session; raise `NotFoundError` if not found; raise `SessionEndedError` if ended
2. Load character, facts, segment
3. Compute `turn_id = next_turn_id(db, session_id)`
4. Store user message
5. Build system prompt and Ollama messages array
6. Call character LLM; buffer response (`character_content`, `thinking`)
7. Call evaluator:
   - Build evaluator prompt; call `run_evaluator`
   - If verdict is `contradiction`: enter contradiction loop (T5.3)
8. Store assistant message (with `ungrounded_implications` set if verdict is `implication` or `new_inference_probabilistic`)
9. Handle non-contradiction verdicts:
   - `new_inference_logical`: call `create_inference` for each item in `new_inferences`
10. Store decision: call `store_decision` with the final verdict and violations
11. Return `(character_content, thinking, turn_id, evaluator_result)`

Step 8 (store message) happens AFTER the contradiction loop resolves — contradicted responses are never written to the DB.

**T5.3 — Contradiction loop**

```python
MAX_CONTRADICTION_RETRIES: int = int(os.getenv("MAX_CONTRADICTION_RETRIES", "3"))
```

When the evaluator returns `contradiction`:
1. Collect violation descriptions as `contradiction_hints`
2. Build a modified character messages array that appends a system-level constraint note: a role `"user"` message with content `"[SYSTEM NOTE: Your previous response contained a contradiction. {descriptions}. Please revise your answer, ensuring you do not contradict any established facts.]"`. This appears as the last message in the array before re-invoking the character LLM.
3. Re-call the character LLM with the modified messages
4. Re-call the evaluator with `contradiction_hints` set
5. If still contradiction: repeat from step 1, accumulating hints
6. After `MAX_CONTRADICTION_RETRIES` consecutive contradictions: break loop; use the most recent response (deliver with warning)
7. When the loop exits cleanly: continue with the clean response

The system note is constructed as a `"user"` role message so it appears naturally in the conversation flow for the model. It is NOT stored in the DB — it is only used for the in-flight regeneration.

The contradiction loop yields a list of `(contradiction_description, iteration_number)` tuples that the router can emit as `sidechannel` SSE events.

**T5.4 — Return value for contradiction events**

`run_turn` collects a list of contradiction notifications during the loop:
```python
class ContradictionNotification(BaseModel):
    iteration: int
    description: str
```

These are included in `EvaluatorResult` (added as an optional field):
```python
class EvaluatorResult(BaseModel):
    ...
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False
```

---

### T6 — API layer additions

**T6.1 — SSE event sequence update**

Full event sequence for a `pass` verdict:
```
event: status
data: {"state": "generating"}

event: status
data: {"state": "reviewing"}

event: message
data: {"role": "assistant", "content": "...", "turn_id": N}

event: done
data: {}
```

Full event sequence for an `implication` verdict:
```
event: status
data: {"state": "generating"}

event: status
data: {"state": "reviewing"}

event: message
data: {"role": "assistant", "content": "...", "turn_id": N, "ungrounded": true}

event: sidechannel
data: {"type": "implication", "turn_id": N, "violations": [...]}

event: done
data: {}
```

Full event sequence for a `contradiction` verdict (one retry loop):
```
event: status
data: {"state": "generating"}

event: status
data: {"state": "reviewing"}

event: sidechannel
data: {"type": "contradiction", "iteration": 1, "description": "..."}

event: status
data: {"state": "regenerating"}

event: status
data: {"state": "reviewing"}

event: message
data: {"role": "assistant", "content": "...", "turn_id": N}

event: done
data: {}
```

The `reviewing` event is emitted before calling `run_turn` (the service handles both LLM calls synchronously, so the router emits `generating`, then awaits `run_turn`, which returns the result including any contradiction events for the router to emit via a generator pattern — see T6.2).

**T6.2 — Generator-based SSE emission**

`run_turn` is synchronous from the router's perspective — it awaits a single coroutine. To emit `sidechannel` and `regenerating` events at the right times during a contradiction loop, the SSE generator needs to interleave with the service.

Design: `run_turn` is refactored to be an async generator (or accepts a callback). Two options:

**Option A (callback):** `run_turn` accepts an optional `on_event: AsyncCallable[[dict], None] | None` callback. During the contradiction loop, it calls `on_event({"type": "contradiction", ...})` before each regeneration. The router provides a callback that appends events to an async queue; the SSE generator reads from the queue.

**Option B (return full result):** `run_turn` remains a plain coroutine and returns all contradiction notifications in `evaluator_result.contradiction_notifications`. The router emits them all at once after `run_turn` returns, before emitting the message event.

**Decision: Option B.** The contradiction loop is fast (server-side, no user interaction required). Emitting all sidechannel events after the loop completes vs. in real time makes no practical difference to the user — the evaluator is the bottleneck, not the SSE delivery. Option B is dramatically simpler to test and avoids async queue complexity.

The router SSE generator becomes:
```python
yield 'event: status\ndata: {"state": "generating"}\n\n'
yield 'event: status\ndata: {"state": "reviewing"}\n\n'
content, thinking, turn_id, eval_result = await run_turn(...)
for notif in eval_result.contradiction_notifications:
    yield f'event: sidechannel\ndata: {json.dumps({"type": "contradiction", ...})}\n\n'
    yield 'event: status\ndata: {"state": "regenerating"}\n\n'
    yield 'event: status\ndata: {"state": "reviewing"}\n\n'
if thinking:
    yield f'event: thinking\ndata: ...\n\n'
msg_data = {"role": "assistant", "content": content, "turn_id": turn_id}
if eval_result.verdict in ("implication", "new_inference_probabilistic"):
    msg_data["ungrounded"] = True
if eval_result.max_retries_exceeded:
    msg_data["contradiction_exhausted"] = True
yield f'event: message\ndata: {json.dumps(msg_data)}\n\n'
if eval_result.violations:
    yield f'event: sidechannel\ndata: ...\n\n'
yield 'event: done\ndata: {}\n\n'
```

**T6.3 — Implication accept/ignore endpoints**

New router: `src/memories/routers/implication.py`
Mounted at `/api/sessions` prefix in `main.py`.

`POST /api/sessions/{session_id}/turns/{turn_id}/accept-implication`
- Body: `{key: str, value: str}` — the fact to create (may differ from suggestion if user edited)
- Validates: session exists, is not ended, session has an assistant message for this turn with `ungrounded_implications` set
- Creates the Fact via `create_fact`
- Reloads facts and rebuilds system prompt
- Re-calls character LLM (no streaming; standard `ollama.chat`) with the updated context; runs evaluator on the response
- If contradiction in regenerated response: contradiction loop (same logic, same max retries)
- Calls `replace_message_content(db, session_id=session_id, turn_id=turn_id, new_content=...)` to overwrite the stored assistant message and clear the `ungrounded_implications` flag
- Updates the decision log for this turn: stores a new decision row with the regenerated verdict
- Returns `{content: str, turn_id: int}`

`POST /api/sessions/{session_id}/turns/{turn_id}/ignore-implication`
- Validates: session exists, turn has ungrounded implications
- No DB change (the message is already tagged; ignoring just dismisses the sidechannel notification)
- Returns 204

**T6.4 — Inference accept/ignore endpoints**

Added to `implication.py`:

`POST /api/sessions/{session_id}/turns/{turn_id}/accept-inference`
- Body: `{statement: str, derivation: str, source_fact_ids: list[int], inference_type: str}`
- Calls `create_inference`; returns 201 with the created Inference object
- Clears the probabilistic inference from the message's `ungrounded_implications` (or leaves it if other violations remain)

`POST /api/sessions/{session_id}/turns/{turn_id}/ignore-inference`
- Returns 204; no DB change

**T6.5 — Decisions endpoint**

New router: `src/memories/routers/decisions.py`
Mounted at `/api/sessions` prefix.

`GET /api/sessions/{session_id}/decisions`
- Validates session exists; returns 404 otherwise
- Returns `list[Decision]` ordered by `turn_id` DESC

---

### T7 — Frontend updates

**T7.1 — Reviewing/regenerating indicator**
Extend the existing `generating...` indicator to handle three states:
- `generating` → "Thinking..."
- `reviewing` → "Reviewing..."
- `regenerating` → "Revising (contradiction found)..."

These are all show/hide states on the same indicator element; the label changes as `status` events arrive.

**T7.2 — Sidechannel notification rendering**
The sidechannel panel gains a Notifications section above the Facts list (visible only when there are active notifications for the current turn). Each notification renders based on `type`:

- `contradiction`: info banner with description (no action needed; auto-resolved server-side)
- `implication`: card with the violation description, the suggested fact (key/value pre-filled), and three buttons: **Accept**, **Edit** (inline), **Ignore**
- `new_inference_probabilistic`: card with the inference statement, derivation, and **Accept** / **Ignore**

Notifications are keyed to `turn_id` and cleared when the next turn begins.

**T7.3 — Ungrounded badge**
Chat messages with `ungrounded: true` in the SSE message event render with a visual badge ("⚠ ungrounded"). The badge is removed when the user accepts the implication and the frontend receives the regenerated response.

**T7.4 — Accept flow (implication)**
On Accept/Edit:
1. Frontend calls `POST /api/sessions/{session_id}/turns/{turn_id}/accept-implication` with `{key, value}`
2. On success, replace the tagged message's `content` with the returned `content`
3. Remove the ungrounded badge
4. Refresh the facts list (the new fact is now present)
5. Dismiss the sidechannel notification

**T7.5 — Accept flow (inference)**
On Accept:
1. Frontend calls `POST /api/sessions/{session_id}/turns/{turn_id}/accept-inference`
2. On success, dismiss the sidechannel notification

**T7.6 — Decisions log panel**
The sidechannel panel gains a collapsible Decisions section at the bottom (collapsed by default). On expand, calls `GET /api/sessions/{session_id}/decisions` and renders each Decision as a row:
```
Turn N  [pass | contradiction | implication | ...]
<reasoning text>
```
Refreshed each time the user expands the panel.

---

## Technology & Infrastructure Decisions

### D1 — Evaluator uses the same model as the character

Both calls use `character.current_model_name or character.modelfile_base`. This is the simplest approach and the right starting point. The evaluator system prompt is entirely different — it instructs the model to act as a critic, not as the character. If evaluator latency proves unacceptable (two full `qwen3:7b` calls per turn), the evaluator can be split to a smaller/faster model by adding a `OLLAMA_EVALUATOR_MODEL` env var without restructuring the code.

### D2 — `format: "json"` (not full JSON schema) for evaluator output

Ollama's structured output with a full JSON schema is appealing but imposes schema-level constraints that can confuse smaller models and produce empty arrays instead of null fields. Using `format: "json"` tells Ollama to return a JSON object without constraining its structure. The evaluator prompt defines the expected schema verbally. Pydantic validates the parsed output with appropriate defaults for missing optional fields.

This is more robust than schema-constrained output for Phase 2, where the model may not yet consistently produce all optional fields. If the model frequently omits required fields, schema constraint can be layered on later.

### D3 — Contradiction loop uses role `"user"` system note, not a modified system prompt

When regenerating after a contradiction, we append a special message to the conversation with role `"user"` and content like `"[SYSTEM NOTE: ...contradiction description...]"`. This is simpler than forking the system prompt for each iteration and still gets the information into the model's context. Prepending `[SYSTEM NOTE:]` signals to the model that this is an instruction, not a user utterance. The message is never stored in the DB.

An alternative is to add a second `role: "system"` message. Ollama supports multiple system messages (they are concatenated), but the behavior is model-dependent. The user-role system note is more reliable across models.

### D4 — `run_turn` stays a plain coroutine (no async generator refactor)

Contradiction notifications are accumulated in memory during the loop and returned in `EvaluatorResult.contradiction_notifications`. The router emits them as SSE events after `run_turn` returns. This avoids async queue complexity and keeps the service layer testable without SSE infrastructure. The practical difference to the user is negligible — the contradiction loop is fast (two LLM calls), not a multi-minute wait.

### D5 — `accept-implication` regenerates via a plain REST endpoint (not SSE)

The accept flow is a discrete user action, not a long-running stream. The regenerated response is typically short and the evaluator runs once. A synchronous REST endpoint returning `{content, turn_id}` is the right tool. The client receives the new content and patches the DOM directly.

If the regeneration itself triggers another implication (the new Fact reveals a second gap), the endpoint returns with `ungrounded: true` again. The client re-renders the notification. This avoids recursive complexity in Phase 2; the user may need to accept multiple implications in sequence, which is acceptable.

### D6 — Decisions stored once per turn (final verdict only)

During a contradiction loop, intermediate evaluations are not stored. Only the final clean verdict is logged. This matches the plan's flowchart where "Log Decision" appears after delivery. Storing all intermediate attempts would fill the decisions log with noise and make it harder to review the audit trail. If debugging the contradiction loop itself becomes necessary, that is a logging concern, not a decisions concern.

### D7 — `experience_updates` verdict coerced to `pass` in Phase 2

The evaluator prompt does not mention Experiences and the character's context contains none. If the model hallucinates an `experience_update` verdict, the service coerces it to `pass` and logs a warning. This prevents Phase 2 from crashing when Phase 4's plumbing doesn't exist yet. The coercion is a one-line guard with a `# TODO Phase 4: handle experience_update` comment.

### D8 — MAX_CONTRADICTION_RETRIES defaults to 3, read from env

Three iterations is enough to detect a prompt-engineering failure vs. a transient model lapse. After three consecutive contradictions, the response is delivered with `contradiction_exhausted: true` in the SSE message event. The frontend renders a warning badge different from the normal ungrounded badge. The env var allows lowering it to 1 during development or raising it for a more persistent character.

---

## Test Plan

Tests are written first. The implementation is complete when all Phase 2 tests pass alongside all existing Phase 1 tests, and overall coverage stays at or above 80% (90% for `services/`).

The existing test infrastructure (in-memory SQLite, `respx` for Ollama mocks) is unchanged. New evaluator mocks return pre-built `EvaluatorResult` objects or raw JSON strings depending on where in the stack the test is operating.

---

### Unit tests — `tests/unit/`

#### `test_evaluator_service.py`

Tests for `build_evaluator_prompt` and `run_evaluator`. Ollama calls mocked with `respx`.

| Test | Asserts |
|------|---------|
| `test_evaluator_prompt_includes_all_facts` | Every `key: value` pair from the fact list appears in the built prompt |
| `test_evaluator_prompt_includes_character_response` | The character response string appears verbatim in the prompt |
| `test_evaluator_prompt_includes_user_message` | The user message appears verbatim in the prompt |
| `test_evaluator_prompt_no_facts_uses_fallback_text` | When facts list is empty, prompt includes a "no facts established" note |
| `test_evaluator_prompt_with_contradiction_hints_lists_them` | `contradiction_hints` non-empty → each hint appears in the prompt |
| `test_evaluator_request_sends_think_false` | Captured Ollama request body has `"think": false` |
| `test_evaluator_request_sends_format_json` | Captured Ollama request body has `"format": "json"` |
| `test_evaluator_parses_pass_verdict` | JSON response `{"verdict": "pass", "decision_log": "..."}` → `EvaluatorResult.verdict == "pass"` |
| `test_evaluator_parses_contradiction_verdict` | `verdict: "contradiction"` → `violations` list has one entry with `type: "contradiction"` |
| `test_evaluator_parses_implication_verdict` | `verdict: "implication"` → `violations[0].suggested_fact` is a dict with `key` and `value` |
| `test_evaluator_parses_new_inference_logical` | `verdict: "new_inference_logical"` → `new_inferences[0].inference_type == "logical"` |
| `test_evaluator_parses_new_inference_probabilistic` | `verdict: "new_inference_probabilistic"` → `new_inferences[0].inference_type == "probabilistic"` |
| `test_evaluator_coerces_experience_update_to_pass` | `verdict: "experience_update"` → returned result has `verdict == "pass"` |
| `test_evaluator_raises_parse_error_on_non_json` | Ollama returns plain text → `EvaluatorParseError` raised |
| `test_evaluator_raises_parse_error_on_missing_verdict` | JSON with no `verdict` key → `EvaluatorParseError` raised |
| `test_evaluator_contradiction_priority_overrides_implication` | JSON has `violations` with both `contradiction` and `implication` types → verdict forced to `"contradiction"` |

#### `test_chat_service.py` (additions to existing file)

| Test | Asserts |
|------|---------|
| `test_evaluator_called_after_character_response` | After character Ollama call, a second Ollama call occurs (captured by respx) |
| `test_run_turn_returns_evaluator_result` | Fourth return value is an `EvaluatorResult` instance |
| `test_pass_verdict_response_stored_and_returned` | Pass verdict → assistant message in DB; content returned |
| `test_pass_verdict_decision_stored` | Pass verdict → decision row in DB with `verdict="pass"` |
| `test_contradiction_response_not_stored_on_first_attempt` | After first contradiction verdict, no assistant message in DB yet |
| `test_contradiction_triggers_second_character_call` | Contradiction verdict → character Ollama called a second time |
| `test_contradiction_second_call_messages_include_system_note` | Second character call's messages array contains a note referencing the contradiction |
| `test_contradiction_loop_exits_on_pass` | Evaluator returns contradiction then pass → response returned; loop exits |
| `test_contradiction_loop_final_response_is_stored` | After clean loop exit, assistant message in DB contains the clean response |
| `test_contradiction_max_retries_exceeded_delivers_anyway` | After `MAX_CONTRADICTION_RETRIES` contradictions, response returned with `max_retries_exceeded=True` in result |
| `test_contradiction_notifications_collected_per_iteration` | Two contradictions → `evaluator_result.contradiction_notifications` has two entries |
| `test_implication_verdict_tags_message_ungrounded` | Implication verdict → stored assistant message has non-null `ungrounded_implications` |
| `test_implication_violations_stored_in_message` | `ungrounded_implications` JSON contains the violation's `suggested_fact` |
| `test_new_inference_logical_creates_inference_row` | `new_inference_logical` verdict → inference row in DB with correct `statement` and `inference_type="logical"` |
| `test_new_inference_probabilistic_tags_message_ungrounded` | Probabilistic verdict → stored assistant message tagged ungrounded |
| `test_new_inference_probabilistic_does_not_create_db_row` | Probabilistic verdict alone → no inference row created (user must accept) |
| `test_decision_stored_for_every_completed_turn` | Every verdict type (pass, implication, contradiction-then-pass) stores exactly one decision row per turn |

---

### Integration tests — `tests/integration/`

#### `test_decisions_repo.py`

| Test | Asserts |
|------|---------|
| `test_store_decision_returns_with_id` | Returned `Decision` has a positive integer `id` |
| `test_store_decision_stores_verdict_and_reasoning` | Fetched decision has matching `verdict` and `reasoning` |
| `test_store_decision_with_violations` | `violations` stored as JSON; retrieved as a Python list |
| `test_store_decision_without_violations` | `violations=None` stored; retrieved as `None` (not an empty list) |
| `test_get_decisions_returns_all_for_session` | All stored decisions for the session are returned |
| `test_get_decisions_ordered_by_turn_id_desc` | Most recent turn's decision appears first |
| `test_get_decisions_isolated_per_session` | Session A decisions not returned when querying session B |

#### `test_inferences_repo.py`

| Test | Asserts |
|------|---------|
| `test_create_inference_stores_statement_and_derivation` | Inference retrievable with matching `statement` and `derivation` |
| `test_create_inference_default_status_is_active` | New inference has `status="active"` |
| `test_create_inference_default_depth_is_one` | New inference has `depth=1` |
| `test_source_fact_ids_stored_and_retrieved_as_list` | `[1, 2]` stored; retrieved as `[1, 2]` |
| `test_source_inference_ids_empty_list_when_not_set` | `source_inference_ids` defaults to `[]` |
| `test_get_inferences_returns_active_only_by_default` | Stale inference not returned by default `get_inferences` |
| `test_inferences_isolated_per_character` | Character A's inferences not returned for character B |
| `test_inference_type_logical_stored_correctly` | `inference_type="logical"` round-trips cleanly |
| `test_inference_type_probabilistic_stored_correctly` | `inference_type="probabilistic"` round-trips cleanly |

#### `test_messages_repo.py` (additions)

| Test | Asserts |
|------|---------|
| `test_tag_message_ungrounded_sets_field` | `tag_message_ungrounded` → message's `ungrounded_implications` is non-null |
| `test_tag_message_ungrounded_stores_violations_as_json` | `ungrounded_implications` is a list of violation dicts when retrieved |
| `test_replace_message_content_updates_text` | After `replace_message_content`, message has the new content |
| `test_replace_message_content_clears_ungrounded` | `ungrounded_implications` is `None` after replacement |
| `test_replace_message_content_nonexistent_turn_raises` | Raises `NotFoundError` for a turn with no assistant message |

#### `test_api_chat.py` (additions to existing file)

| Test | Asserts |
|------|---------|
| `test_send_message_emits_reviewing_event` | `event: status` with `state: "reviewing"` appears in the stream |
| `test_send_message_reviewing_event_before_message_event` | `status(reviewing)` appears before `event: message` |
| `test_send_message_pass_verdict_no_ungrounded_field` | `message` event payload does not contain `ungrounded: true` for a pass verdict |
| `test_send_message_pass_verdict_decision_stored` | After the turn, a decision row with `verdict="pass"` exists in DB |
| `test_send_message_implication_verdict_emits_ungrounded_message` | Implication evaluator mock → `message` event has `ungrounded: true` |
| `test_send_message_implication_verdict_emits_sidechannel` | Implication evaluator mock → `sidechannel` event with `type: "implication"` emitted after `message` |
| `test_send_message_implication_sidechannel_contains_violations` | Sidechannel event payload includes `violations` list with `suggested_fact` |
| `test_send_message_contradiction_emits_sidechannel_before_message` | Contradiction-then-pass evaluator mock → `sidechannel` event with `type: "contradiction"` appears before final `message` |
| `test_send_message_contradiction_emits_regenerating_status` | Contradiction loop → `status: {state: "regenerating"}` emitted |
| `test_send_message_contradiction_loop_final_message_is_clean` | After contradiction + pass → final `message` event does not have `ungrounded: true` |
| `test_send_message_contradiction_response_not_premature` | During contradiction loop, no `message` event is emitted until the clean response |
| `test_send_message_max_retries_exceeded_flag_in_message` | Three consecutive contradictions → `message` event has `contradiction_exhausted: true` |
| `test_send_message_new_inference_logical_stored` | `new_inference_logical` evaluator mock → inference row in DB after turn |
| `test_send_message_new_inference_probabilistic_emits_sidechannel` | Probabilistic inference mock → `sidechannel` event with `type: "new_inference_probabilistic"` |
| `test_send_message_status_event_order_for_pass` | Events in exact order: `status(generating)` → `status(reviewing)` → `message` → `done` |

#### `test_api_implication.py` (new file)

Fixtures: a session with one completed implication-verdict turn (message has `ungrounded_implications` set).

| Test | Asserts |
|------|---------|
| `test_accept_implication_creates_fact_in_db` | POST accept → fact with given `key` and `value` exists in DB |
| `test_accept_implication_returns_200_with_content` | Response is 200 with JSON body containing `content` and `turn_id` |
| `test_accept_implication_content_differs_from_original` | Returned `content` is the regenerated response (from mocked Ollama), not the original |
| `test_accept_implication_clears_ungrounded_implications_on_message` | After accept, assistant message in DB has `ungrounded_implications=None` |
| `test_accept_implication_stores_new_decision` | A new decision row is stored with the regenerated verdict |
| `test_edit_implication_uses_user_provided_value` | Body `{key: "x", value: "different"}` → fact stored with `value="different"` |
| `test_ignore_implication_returns_204` | POST ignore → 204 |
| `test_ignore_implication_message_ungrounded_remains_set` | After ignore, `ungrounded_implications` still non-null in DB |
| `test_accept_implication_unknown_session_returns_404` | POST to non-existent session_id → 404 |
| `test_accept_implication_unknown_turn_returns_404` | POST to non-existent turn_id → 404 |
| `test_accept_implication_on_clean_turn_returns_422` | POST accept on a turn that has no `ungrounded_implications` → 422 |
| `test_accept_inference_creates_inference_in_db` | POST accept-inference → inference row in DB with provided `statement` |
| `test_accept_inference_returns_201_with_inference` | Response is 201 with the created Inference JSON |
| `test_ignore_inference_returns_204` | POST ignore-inference → 204 |
| `test_ignore_inference_does_not_create_inference_row` | After ignore, no new row in inferences table |

#### `test_api_decisions.py` (new file)

| Test | Asserts |
|------|---------|
| `test_get_decisions_initially_empty` | GET before any turns returns `[]` |
| `test_get_decisions_after_one_turn` | GET after one turn returns a list with one decision |
| `test_get_decisions_contains_verdict_field` | Each decision has a `verdict` field |
| `test_get_decisions_contains_reasoning_field` | Each decision has a non-empty `reasoning` field |
| `test_get_decisions_ordered_by_turn_id_desc` | After two turns, the second turn's decision appears first |
| `test_get_decisions_unknown_session_returns_404` | 404 for a non-existent session id |
| `test_get_decisions_includes_violations_for_implication` | Decision for an implication turn has non-null `violations` |

#### `test_ollama_client.py` (additions)

| Test | Asserts |
|------|---------|
| `test_format_parameter_included_in_request_body` | When `format="json"` is passed, the Ollama request payload includes `"format": "json"` |
| `test_format_parameter_absent_when_not_passed` | Without the `format` argument, `"format"` key is absent from the request payload |
| `test_format_schema_dict_passed_through` | Dict passed as `format` appears verbatim in the request body |

---

## Not in Scope for Phase 2

The following are intentionally deferred:

- **Inference injection into system prompt** — Phase 3. Inferences discovered in Phase 2 are stored but not yet included in the character's context.
- **Eager inference generation on Fact add/edit** — Phase 3.
- **Inference cascade on Fact change** — Phase 3.
- **End-of-session evaluator pass (closing journal + Experience proposals)** — Phase 4.
- **Experiences table usage** — Phase 4. The table exists but is empty.
- **`experience_update` verdict full handling** — Phase 4. Phase 2 coerces it to `pass`.
- **Token counting and context budget** — Phase 5a.
- **Compression** — Phase 5b. All messages remain verbatim in the single `session_start` segment.
- **Segment boundary logic** — Phase 5b.
- **Modelfile export** — Phase 6 stretch.
- **Phone-responsive layout** — Phase 6.
- **E2E tests (Playwright)** — deferred until UI is stable.
