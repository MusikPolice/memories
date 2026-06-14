# Step 0b — `require_fact` Suspension PoC

## Why this step exists

Step 0 proved that `chat_with_tools()` drives the tool-call loop reliably and that the
`tool_gate.py` primitives are correct in isolation. Neither of those tests exercised the
coordination between a live SSE generator and a tool handler that blocks on user input.
That coordination is the load-bearing mechanism for `require_fact` (Step 6), mutable
approval (Step 7), and immutable-unset approval (Step 7). All three flows share the same
architectural pattern: a tool handler awaits a gate, the SSE generator stays alive until
the gate resolves, a second HTTP request from the client resolves the gate, and the
tool handler resumes.

Step 0b proves this pattern works in a running FastAPI process before any of those
production flows are built. The PoC does not call a real LLM and does not consult the
schema. It simulates the suspension cycle with a test-only SSE endpoint, confirms the
SSE connection survives the wait, confirms events emitted before and after resolution
are both delivered, and confirms the gate is cleaned up in all exit paths. If any of
these properties fail, the design needs revision before Step 6 begins.

**Success criterion:** all tests in `tests/integration/test_require_fact_poc.py` pass.

---

## What this step does NOT build

- Real schema path validation on `require_fact` — mocked in the PoC
- Actual LLM invocation of `require_fact` — the PoC fires the tool handler directly
- Integration into `run_turn()` — that is Step 6
- Any UI changes — the PoC endpoint returns raw SSE that tests consume programmatically
- The mutable and immutable-unset approval flows — same mechanism, different steps

---

## The critical challenge: emitting events from inside a tool handler

The current SSE architecture in `chat.py` runs `run_turn()` as a background
`asyncio.Task`. The SSE generator polls a `Queue[str]` every 50 ms, yielding status
events when the task puts a status string into it. When the Task blocks on
`await_gate()`, the SSE generator loop continues spinning — the Task is alive but
parked by the event loop, and the generator's `asyncio.wait_for(..., timeout=0.05)` call
yields control back on every iteration. This means the existing architecture already
keeps the SSE connection alive during suspension at no additional cost.

The challenge is event emission. Before the tool handler suspends, it needs to emit a
`sidechannel(require_fact)` event so the client can surface the blocking card. The tool
handler is executing inside `chat_with_tools()`, deep in the Task's call stack, with no
direct access to the SSE generator's yield. A callback is the right seam.

### The on_event callback

The current `StatusCallback = Callable[[str], Awaitable[None]] | None` is too narrow —
it models only status strings. Step 0b generalises this to a typed `EventCallback` that
can carry any SSE event the tool handler needs to emit:

```python
@dataclass
class SSEEvent:
    event: str       # "status", "sidechannel", "thinking", etc.
    data: dict       # serialised as JSON in the SSE data line
```

```python
EventCallback = Callable[[SSEEvent], Awaitable[None]] | None
```

The SSE generator in `chat.py` creates an `asyncio.Queue[SSEEvent]`, wraps it in an
`on_event` async callback, and passes it to `run_turn()`. While the Task is running, the
generator reads from this queue the same way it currently reads status strings. The tool
handler for `require_fact` calls `on_event` with a `sidechannel` event before awaiting
the gate, and the generator delivers it to the client immediately.

`run_turn()` (and any service it calls) receives `on_event` as a parameter wherever an
event needs to be emitted. This means the `require_fact` tool handler is a closure
defined inside the `run_turn()` call site, capturing `on_event`, `session_id`, and
`turn_id` from the surrounding scope.

`StatusCallback` and its usages are replaced by `EventCallback` in the same commit. The
existing status strings (`"generating"`, `"reviewing"`, `"regenerating"`) become
`SSEEvent(event="status", data={"state": "..."})` objects. This is a mechanical rename
with no behavioural change.

---

## The SSE keep-alive

The existing poll loop runs every 50 ms. During a suspension that might last 30+ seconds
while the user reads the blocking card and provides a value, the loop is running but
producing no output. Browsers and intermediate proxies time out idle HTTP connections
after 30–120 seconds. The fix is for the generator to emit an SSE comment line
(`: ping\n\n`) on every iteration where no event was received from the queue and no
Task completion occurred.

SSE comment lines (lines starting with `:`) are part of the SSE specification. All
compliant clients and proxies treat them as keep-alive signals and do not surface them to
application event handlers. They add one line of data per 50 ms iteration — roughly
20 bytes per iteration, negligible bandwidth.

The keep-alive ping is implemented in the SSE generator loop; no changes are needed to
any service layer code.

---

## The turn ID

`tool_gate.py` keys gates by `(session_id, turn_id)`. The `session_id` is already
available to the SSE generator from the route parameter. The `turn_id` is generated by
`next_turn_id()` at the start of `run_turn()`.

In the full system (Step 6), the gate is created inside `run_turn()` after `turn_id` is
established. The accept endpoint `POST /api/sessions/{id}/turns/{turn_id}/require-fact/respond`
uses `turn_id` from the URL path to look up the correct gate.

For the PoC, `turn_id` is a query parameter that the caller supplies. The PoC endpoint
creates the gate using the supplied `turn_id` and cleans it up when the SSE stream ends.
The integration tests supply a fixed `turn_id` so both the PoC SSE endpoint and the
accept endpoint can be called with a known ID.

---

## PoC architecture: the two endpoints

### `POST /api/sessions/{session_id}/test-require-fact-poc`

A new SSE endpoint in a new router `src/memories/routers/test_poc.py`. Query params:
`turn_id: int` (required) and `delay_ms: int = 0` (optional, for testing keep-alive
timing without real waits).

This endpoint:
1. Validates the session exists (uses the real DB)
2. Calls `create_gate(session_id, turn_id)` — if the gate already exists, returns 409
3. Emits `status(generating)` immediately
4. Emits `sidechannel` with `type: require_fact`, `path: "Character.Identity.Name"`,
   `reason: "Integration test"`, and `suggested_value: "Alice"` — this is the event that
   tells the client a blocking card should appear
5. Awaits `await_gate(session_id, turn_id)` — the SSE stream is now suspended
6. When the gate resolves, emits `message` with the resolved value (or `"(dismissed)"` if
   `None`) as the content
7. Emits `done`
8. In a `finally` block: calls `cleanup_gate(session_id, turn_id)`

The SSE generator wraps the above in the same `asyncio.Task` pattern as `chat.py`, with
the same keep-alive ping loop. The `on_event` callback is the mechanism for steps 4 and 6 above.

These endpoints are explicitly PoC scaffolding and will be removed in Step 6 when
`require_fact` is integrated into the real `run_turn()` orchestration. A comment at the
top of `test_poc.py` says so.

### `POST /api/sessions/{session_id}/turns/{turn_id}/require-fact/respond`

This is a permanent endpoint (not PoC scaffolding) in a new router
`src/memories/routers/require_fact.py`. It is the user-facing HTTP handler for all
blocking `require_fact` cards — PoC and production alike.

Request body: `{"value": "Sarah"}` for accept, or `{}` / `{"value": null}` for dismiss.

Behaviour:
1. Looks up `_pending[(session_id, turn_id)]` via `tool_gate` internals, or calls
   `resolve_gate()` which raises `KeyError` if the gate doesn't exist
2. On `KeyError`: returns HTTP 404 with a message like `"No pending require_fact for
   session {id} turn {turn_id}"`
3. On `asyncio.QueueFull` (gate already resolved): returns HTTP 409 with `"Already
   resolved"`
4. On success: calls `resolve_gate(session_id, turn_id, value_or_None)` and returns 200

The `resolve_gate` call is non-blocking (`put_nowait`). The endpoint returns immediately
without waiting for the SSE stream to resume.

---

## Gate lifecycle and error handling

The gate spans from the beginning of a turn to its end, regardless of how the turn ends.
This is enforced by a `try/finally` wrapping the entire gate lifecycle in the PoC
endpoint (and later in `run_turn()`):

```
create_gate(session_id, turn_id)   ← before entering the SSE generator body
try:
    … everything …
finally:
    cleanup_gate(session_id, turn_id)
```

### Error paths

**Exception in the generator body**: The `finally` block fires, the gate is removed from
`_pending`. If the accept endpoint is called after this, it gets a 404.

**Client disconnect (browser closes the tab)**: FastAPI/Starlette detects the closed
connection when the generator tries to yield. An `asyncio.CancelledError` or
`GeneratorExit` propagates. The `finally` block fires. If the generator is suspended on
`await_gate()`, it needs to be woken up by the `finally` path before it can run. The
correct pattern is to wrap `await_gate()` in a try/except for `asyncio.CancelledError`
and call `cleanup_gate()` explicitly, or ensure `cleanup_gate()` is the outer `finally`.
The exact mechanism depends on how Starlette handles generator cancellation; the
integration tests must verify the gate is absent after a simulated client disconnect.

**Accept never called**: The gate stays in `_pending` until the SSE stream times out or
the client disconnects. There is no server-side timeout in this step; that is deferred
future work. The PoC tests should have a test timeout that catches hangs.

**Double accept**: `resolve_gate()` calls `put_nowait()` on a `maxsize=1` queue.
The second call raises `asyncio.QueueFull`, which the accept endpoint converts to
HTTP 409. This is the correct behaviour — the first resolution already triggered the
resume; a second resolution would corrupt the flow.

---

## The "no partial prose" property

Plan v2 requires that the character's response be generated fresh after `require_fact`
resolves, without any prose from before the tool call leaking into the new response.
This property holds by construction in the tool-call model:

`chat_with_tools()` uses `stream: false`. Each Ollama call returns a complete response
that is either a set of tool calls OR a prose string — never a mix. If the model calls
`require_fact`, it has not generated any prose in that invocation. The tool handler
resolves the gate and returns the value as the tool result. The loop re-invokes the
model with the updated message history (`[..., assistant: tool_call, tool: "Sarah"]`),
and the model generates a complete prose response at that point.

There are no partial prose fragments in the message history and no streamed tokens to
discard. The property holds without any additional code, and the PoC does not need to
test it directly. The plan-v2 note about "cleanly discarding the partial response" refers
to this architectural guarantee, not to an explicit discard mechanism.

---

## Test cases

### Integration tests: `tests/integration/test_require_fact_poc.py`

All tests use `httpx.AsyncClient(app=app, base_url="http://test")` with a real in-memory
DB (the existing `db` fixture). They do not mock Ollama. The PoC endpoint does not call
Ollama; it simulates the suspension directly. The `session_id` and `turn_id` used across
tests are fixed constants defined at the top of the file.

**Concurrency pattern for SSE + accept:** the integration tests use `asyncio.create_task`
to run the SSE consumer concurrently. The consumer reads lines from the stream and
collects parsed events in a list. An `asyncio.Event` (an internal synchronisation
primitive, not an SSE event) is set when the `require_fact` sidechannel is seen, so the
main test coroutine knows when to send the accept request. A short `asyncio.wait_for`
wrapper on the consumer task prevents the test from hanging indefinitely if the PoC
endpoint malfunctions.

---

**`test_sse_stream_starts_before_suspension`**

Sends a request to the PoC endpoint. Collects events until the `require_fact`
sidechannel appears (times out the whole test if it does not appear within 5 seconds).
Does not send an accept request. Asserts:
- A `status(generating)` event was received before the sidechannel
- The sidechannel event has `type: require_fact`, a `path`, a `reason`, and a
  `suggested_value`
- The SSE connection is still open (the stream has not yielded `done` or raised)

---

**`test_accept_with_value_resumes_stream`**

Normal accept flow:
1. Consumer task starts, collects events, sets an `asyncio.Event` when the
   `require_fact` sidechannel is seen
2. Main coroutine waits for the event, then calls the accept endpoint with
   `{"value": "Sarah"}`
3. Accept endpoint returns 200
4. Consumer task sees a `message` event followed by a `done` event; stream closes

Asserts:
- Accept endpoint returns HTTP 200
- The `message` event content contains `"Sarah"` (or the message payload includes the
  confirmed value)
- `done` is the final event in the stream
- The gate is absent from `_pending` after the stream closes (inspected directly via
  `tool_gate._pending`)

---

**`test_dismiss_resolves_gate_with_none`**

Same setup as above but calls the accept endpoint with `{}` (no value, i.e. dismiss).

Asserts:
- Accept endpoint returns HTTP 200
- The `message` event content contains `"(dismissed)"` or equivalent (whatever the PoC
  endpoint emits for a `None` resolution)
- `done` is the final event in the stream
- Gate is absent after stream closes

---

**`test_events_before_suspension_are_delivered`**

Asserts that the `status(generating)` event emitted before the gate blocks is received
by the client. This might seem trivial but is a meaningful correctness check: if the
generator's keep-alive loop swallowed early events or the queue had a read-ordering bug,
this test would catch it.

---

**`test_keep_alive_comments_emitted_during_suspension`**

Uses the `delay_ms=100` query parameter to make the PoC endpoint wait 100 ms before
setting up the gate (simulating a short window where the SSE stream is alive but
producing no semantic events). During that window, the SSE generator should emit at
least one `: ping` comment line.

Asserts:
- At least one raw line in the SSE stream starts with `:`
- These comment lines appear before the `require_fact` sidechannel event

The test does not wait more than 1 second total (the accept is sent immediately after
the sidechannel appears).

---

**`test_double_accept_returns_409`**

After the consumer sees the `require_fact` sidechannel, sends the accept request twice
in rapid succession (no delay between them). The first call should return 200; the
second should return 409.

Asserts:
- First accept: HTTP 200
- Second accept: HTTP 409
- The stream still terminates cleanly (the first resolution took effect)

---

**`test_accept_unknown_turn_returns_404`**

Calls the accept endpoint with a `turn_id` that has no corresponding gate in `_pending`
(a turn_id not used in any active stream, e.g. `999999`).

Asserts: HTTP 404 with an informative message.

---

**`test_accept_before_suspension_resolves_immediately`**

Calls the accept endpoint before starting the PoC SSE stream. Because
`asyncio.Queue(maxsize=1)` buffers one item, `resolve_gate()` succeeds and the value
sits in the queue. When the SSE stream starts and the generator eventually calls
`await_gate()`, it returns immediately without blocking.

Asserts:
- Accept endpoint returns 200 (gate must already exist — create it in the test setup)
- SSE stream completes without hanging
- The `message` event contains the pre-supplied value

This test validates the pre-resolve buffering property that `test_resolve_before_await`
verified at the unit level in Step 0.

---

**`test_gate_absent_after_normal_completion`**

After a complete accept flow, inspects `tool_gate._pending` directly.

Asserts: `(session_id, turn_id)` is not present in `_pending`.

---

**`test_gate_absent_after_error`**

The PoC endpoint is parameterised to raise an exception after emitting the sidechannel
but before the gate is resolved (a `?fail=1` query param triggers this code path in the
test-only endpoint). The SSE stream terminates with an error. The accept endpoint is
never called.

Asserts: `(session_id, turn_id)` is not present in `_pending` after the stream closes.

---

**`test_session_not_found_returns_404`**

Calls the PoC SSE endpoint with a `session_id` that does not exist in the DB.

Asserts: returns HTTP 404 immediately (not an SSE stream at all).

---

**`test_duplicate_gate_for_same_key_returns_409`**

Starts the PoC SSE stream for `(session_id=1, turn_id=42)` (it blocks, waiting for
accept). Without resolving that stream, starts a second PoC SSE stream for the same
`(session_id=1, turn_id=42)`.

Asserts: the second request returns HTTP 409 (gate already exists — `create_gate()`
raises `ValueError`). The first stream is unaffected.

---

### Additions to unit tests

These are added to `tests/unit/test_tool_gate.py` (Step 0's file) since they cover the
same module. They do not require HTTP infrastructure.

**`test_gate_resolves_through_callback_chain`**

A simulated tool handler awaits the gate inside `chat_with_tools()`. The test resolves
the gate from a concurrent coroutine, confirms the handler receives the value, and
confirms `chat_with_tools()` returns the correct `ToolCallResult`. The Ollama HTTP calls
are mocked (using `respx`). This is the closest unit-level analogue of the full
integration test.

**`test_sse_event_dataclass`**

Verifies `SSEEvent` is serialisable to the expected SSE wire format (two lines:
`event: <type>` and `data: <json>`). If `SSEEvent` is a dataclass or Pydantic model,
this test asserts its string representation or a helper function produces the correct
output. This guards the event serialisation path that the generator relies on.

---

## Event taxonomy for the sidechannel card

The `sidechannel` SSE event emitted when the tool handler fires `require_fact` has the
following payload structure:

| Field | Type | Content |
|---|---|---|
| `type` | string | `"require_fact"` |
| `turn_id` | int | The turn ID the client uses to call the accept endpoint |
| `path` | string | Dot-notation schema path that needs a value |
| `reason` | string | The character's stated reason for needing the value |
| `suggested_value` | string \| null | Character's suggestion, or null if none given |

The `turn_id` field is included in the sidechannel payload so the client always knows
the correct URL for the accept endpoint, even if it was not tracking turn IDs elsewhere.
This is important for the UI implementation (Step 8): when the card appears, the client
immediately knows where to POST the response.

---

## Interaction with the existing status queue

The existing `chat.py` SSE generator creates `asyncio.Queue[str]` and passes `_on_status`
to `run_turn()`. This callback is called with `"generating"`, `"reviewing"`, etc.

Step 0b replaces `asyncio.Queue[str]` with `asyncio.Queue[SSEEvent]` and replaces
`_on_status` with `_on_event`. The generator reads `SSEEvent` objects from the queue
and formats them into SSE wire format. Status events become `SSEEvent(event="status",
data={"state": "generating"})`. This is a mechanical rename with no external behaviour
change from the client's perspective.

`run_turn()` signature changes from `on_status: StatusCallback = None` to
`on_event: EventCallback = None`. Callers that pass `on_status` are updated to pass
`on_event`. Callers that pass `None` (e.g. integration tests that call `run_turn()`
directly without an SSE wrapper) continue to pass `None`.

---

## Fallback

If the integration tests demonstrate that the SSE connection cannot survive the
suspension period reliably (browser or proxy cut off the connection despite keep-alive
pings), the fallback is the post-generation design: `require_fact` is not a mid-turn
Character LLM tool call at all. Instead, the Character Evaluator detects the unset
immutable path after the character generates its response, surfaces the blocking card,
and the Character LLM is re-invoked fresh with the confirmed value in context. This
eliminates mid-generation suspension entirely.

The fallback has a UX cost (the character generates a response that is withheld until the
user supplies the value, and then the response is regenerated rather than delivered
as-is), but it avoids the keep-alive reliability concern entirely. The gate mechanism is
still used in the fallback — it is the approval flow that changes, not the coordination
mechanism.

If the fallback is adopted, update plan-v2.md to remove `require_fact` from the
Character LLM tool list and note the switch to post-generation detection.

---

## Files changed by this step

| Action | File | Notes |
|---|---|---|
| Add | `src/memories/routers/test_poc.py` | PoC-only SSE endpoint; removed in Step 6 |
| Add | `src/memories/routers/require_fact.py` | Permanent accept/dismiss endpoint |
| Modify | `src/memories/routers/chat.py` | Replace `Queue[str]` with `Queue[SSEEvent]`; add keep-alive pings |
| Modify | `src/memories/services/chat_service.py` | `on_status` → `on_event`; `StatusCallback` → `EventCallback` |
| Modify | `src/memories/main.py` | Register new routers |
| Add | `tests/integration/test_require_fact_poc.py` | All integration tests above |
| Modify | `tests/unit/test_tool_gate.py` | Two new unit tests listed above |

No changes to `tool_gate.py` — it is correct as-is from Step 0.

No DB schema changes. No frontend changes. No changes to `ollama_client.py`.

---

## Dependency order relative to other steps

Step 0b must be complete before Step 6 (`require_fact` and asyncio.Queue coordination),
which builds the real integration of `require_fact` into `run_turn()`. It does not need
to be complete before Steps 1–5, which do not involve blocking tool calls.

Step 0b can proceed in parallel with Steps 1–3 (schema file, DB, and prompt changes),
since it touches none of those layers. It should be complete before Step 5 (Character
Evaluator tool-call loop) because the `EventCallback` pattern established here is used
by the Evaluator's `set_fact` approval flow in Step 7.
