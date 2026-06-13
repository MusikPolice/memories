# Step 0 — Tool-Calling Proof of Concept

## Why this step exists

Tool calling is the load-bearing mechanism for the entire v2 design. The World Builder
uses `author_set_fact`, the Character Evaluator uses `set_fact` and `propose_inference`,
and the Character LLM uses `require_fact`. Every subsequent step depends on the assumption
that the Ollama client can drive the tool-call loop reliably.

The current `OllamaClient` has no tool-calling support. Step 0 adds it and proves it
works through unit tests before any schema, storage, or prompt work begins.

**Success criterion:** all unit tests in `tests/unit/test_tool_calling.py` and
`tests/unit/test_tool_gate.py` pass with `uv run pytest tests/unit/test_tool_calling.py
tests/unit/test_tool_gate.py`.

---

## Part A — Tool-Calling Loop in `OllamaClient`

### What changes

A new method `chat_with_tools()` is added to `OllamaClient`. The existing `chat()`
method is not modified — it remains the path for character generation and evaluator calls
that use structured JSON verdicts (until those are replaced in later steps). The new
method is a clean addition with no backward-compatibility concerns.

### Streaming decision

`chat()` uses `stream: true` with NDJSON. Tool-calling invocations use `stream: false`
for two reasons:

1. The model stops mid-response when it decides to call a tool — collecting streaming
   tokens is meaningless when what we actually need is the `tool_calls` array.
2. Non-streaming is simpler to implement and test. The full response arrives as a single
   JSON object that can be parsed in one step.

`chat_with_tools()` issues non-streaming POST requests to `/api/chat` with
`"stream": false`.

### Ollama API format

**Request with tools:**
```json
{
  "model": "qwen3:7b",
  "messages": [
    {"role": "user", "content": "Set the mood to Anxious"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "set_fact",
        "description": "Set the value of a fact at the given schema path",
        "parameters": {
          "type": "object",
          "properties": {
            "path": { "type": "string" },
            "value": { "type": "string" }
          },
          "required": ["path", "value"]
        }
      }
    }
  ],
  "stream": false
}
```

**Response when model calls a tool:**
```json
{
  "message": {
    "role": "assistant",
    "content": "",
    "tool_calls": [
      {
        "function": {
          "name": "set_fact",
          "arguments": { "path": "Character.State-Of-Mind.Mood", "value": "Anxious" }
        }
      }
    ]
  },
  "done": true,
  "prompt_eval_count": 100,
  "eval_count": 10
}
```

**Response when model returns plain content (no tool calls):**
```json
{
  "message": {
    "role": "assistant",
    "content": "Done. I've set the mood to Anxious.",
    "tool_calls": null
  },
  "done": true,
  "prompt_eval_count": 50,
  "eval_count": 5
}
```

**Tool result message appended by the client before re-invoking:**
```json
{ "role": "tool", "content": "OK" }
```

### Return type

```python
from dataclasses import dataclass

@dataclass
class ToolCallResult:
    content: str                      # final plain-text response from the model
    history: list[dict]               # full message history (inputs + all tool turns + final)
    rounds: int                       # number of HTTP round-trips made
    cap_reached: bool                 # True if the loop was cut short by max_rounds
```

`history` is the complete list of messages including the original `messages` argument,
every assistant tool-call message, every tool-result message, and the final assistant
message. The caller (World Builder, Character Evaluator) uses this list for decisions
logging and to build the next system prompt.

When `cap_reached` is True, `content` may be an empty string if the model never returned
plain content before the cap was hit. The caller is responsible for deciding what to do
in that case (see cap handling below).

### Method signature

```python
from collections.abc import Callable
from typing import Any

ToolHandler = Callable[[dict[str, Any]], str]
# Called with the arguments dict from the tool_call. Must return a result string.
# May raise any exception — the exception message becomes the error string passed back
# to the model as the tool result.

async def chat_with_tools(
    self,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_handlers: dict[str, ToolHandler],
    max_rounds: int = MAX_TOOL_CALL_ROUNDS,
) -> ToolCallResult:
```

`MAX_TOOL_CALL_ROUNDS` is a module-level constant (default `10`) read from the
`MAX_TOOL_CALL_ROUNDS` environment variable, matching the pattern used for
`MAX_CONTRADICTION_RETRIES`.

### Loop logic (pseudocode)

```
history = list(messages)  # copy so we don't mutate the caller's list
content = ""
rounds = 0

while rounds < max_rounds:
    response = POST /api/chat with history + tools, stream=false
    rounds += 1
    msg = response["message"]
    history.append(msg)

    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        # Model returned plain content — we're done
        content = msg.get("content", "")
        return ToolCallResult(content, history, rounds, cap_reached=False)

    # Execute each tool call and append results
    for call in tool_calls:
        name = call["function"]["name"]
        args = call["function"]["arguments"]
        handler = tool_handlers.get(name)
        if handler is None:
            result = f"Unknown tool: {name}"
        else:
            try:
                result = handler(args)
            except Exception as exc:
                result = f"Error: {exc}"
        history.append({"role": "tool", "content": result})

return ToolCallResult(content, history, rounds, cap_reached=True)
```

Notes:
- The model may include multiple entries in `tool_calls` in a single response. Each one
  is executed in order; all results are appended before re-invoking.
- A handler that raises an exception has its `str(exc)` sent back to the model as an
  error string. The model sees this and can retry with a corrected call. This is the
  mechanism for enum validation errors and schema-path-not-found errors.
- An unknown tool name returns an error string rather than raising — the model gets
  a chance to recover rather than the call crashing.
- `OllamaConnectionError` and `OllamaResponseError` propagate normally as uncaught
  exceptions; these are infrastructure failures, not recoverable tool-call failures.

### Cap handling

The `cap_reached` flag is returned to the caller; `chat_with_tools()` does not apply any
fallback policy itself. The policy (non-terminal: inject terminal message; terminal:
treat as pass + log warning) is the caller's responsibility and belongs in the evaluator
or world-builder service layer, not in the HTTP client. This keeps the client simple and
single-purpose.

---

## Part B — asyncio.Queue Turn Gate

### What this is

`require_fact`, mutable-fact approval, and immutable-unset approval all need to suspend
the tool-call loop mid-turn, surface a blocking card to the user via SSE, and resume once
the user responds. The mechanism is a per-turn `asyncio.Queue` — the tool handler
`await`s the queue, an HTTP endpoint `put`s the user's response, and the handler returns
the resolved value to the LLM as the tool result.

This part of Step 0 proves that the coordination mechanism works correctly in isolation,
without requiring a real SSE stream or HTTP server.

### New module: `src/memories/services/tool_gate.py`

```python
import asyncio

_pending: dict[tuple[int, int], asyncio.Queue[str | None]] = {}

def create_gate(session_id: int, turn_id: int) -> None:
    _pending[(session_id, turn_id)] = asyncio.Queue(maxsize=1)

async def await_gate(session_id: int, turn_id: int) -> str | None:
    return await _pending[(session_id, turn_id)].get()

def resolve_gate(session_id: int, turn_id: int, value: str | None) -> None:
    _pending[(session_id, turn_id)].put_nowait(value)

def cleanup_gate(session_id: int, turn_id: int) -> None:
    _pending.pop((session_id, turn_id), None)
```

`str | None` because the user may dismiss the card, in which case `None` is put into the
queue and the tool handler returns a "no value provided" result to the model.

### Why `maxsize=1`

A blocking approval card should only be resolved once. `maxsize=1` ensures the `put`
raises `asyncio.QueueFull` if something calls `resolve_gate` twice for the same turn,
making the double-resolve bug visible rather than silently queuing a second value.

---

## Test helpers to add to `tests/unit/conftest.py`

```python
def make_tool_call_response(
    tool_name: str,
    arguments: dict,
    prompt_eval_count: int = 10,
    eval_count: int = 5,
) -> bytes:
    """Build a non-streaming Ollama JSON response where the model calls one tool."""
    obj = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": tool_name, "arguments": arguments}}
            ],
        },
        "done": True,
        "prompt_eval_count": prompt_eval_count,
        "eval_count": eval_count,
    }
    return json.dumps(obj).encode()


def make_multi_tool_call_response(
    calls: list[tuple[str, dict]],
) -> bytes:
    """Build a response where the model calls multiple tools in one turn."""
    obj = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": name, "arguments": args}}
                for name, args in calls
            ],
        },
        "done": True,
    }
    return json.dumps(obj).encode()


def make_plain_tool_response(content: str) -> bytes:
    """Build a non-streaming Ollama JSON response with plain content and no tool calls."""
    obj = {
        "message": {"role": "assistant", "content": content, "tool_calls": None},
        "done": True,
    }
    return json.dumps(obj).encode()
```

---

## Test file: `tests/unit/test_tool_calling.py`

### Test cases

All tests use `@respx.mock` and the `ollama` fixture. The `tools` list used across tests
is a single `set_fact` tool definition (exact JSON schema matching the Ollama format
shown above). The `tool_handlers` dict maps `"set_fact"` to a simple lambda that records
its arguments and returns `"OK"`.

---

**`test_tools_list_sent_in_request_body`**

Mocks one tool-call response followed by one plain response. Asserts the first HTTP
request body contains a `"tools"` key whose value matches the tools list passed in.

---

**`test_stream_false_used_for_tool_calls`**

Same setup as above. Asserts the HTTP request body has `"stream": false`.

---

**`test_no_tool_calls_returns_plain_content_directly`**

Mocks a single plain response (no tool calls). Asserts `result.content` equals the
mocked content, `result.rounds == 1`, `result.cap_reached` is False, and no handler
was called.

---

**`test_single_tool_call_then_plain_content`**

Mock sequence: tool-call response → plain response after handler returns `"OK"`.

Asserts:
- Handler called once with the correct arguments dict
- `result.content` equals the plain-response content
- `result.rounds == 2`
- `result.cap_reached` is False

---

**`test_multiple_sequential_tool_calls`**

Mock sequence: three tool-call responses (one tool call each) → one plain response.

Asserts:
- Handler called three times in order
- `result.rounds == 4`
- `result.cap_reached` is False

---

**`test_multiple_tool_calls_in_single_response`**

Mock sequence: one response containing two tool calls → one plain response.

Asserts:
- Handler called twice, once per tool call, in declaration order
- Two tool-result messages appended to history before re-invocation
- `result.rounds == 2`

---

**`test_history_contains_all_messages`**

Mock sequence: tool-call response → plain response.

Asserts `result.history` contains, in order:
1. The original user message
2. The assistant tool-call message
3. The tool-result message (`role: "tool"`, `content: "OK"`)
4. The final assistant message

---

**`test_handler_exception_becomes_error_string`**

`set_fact` handler raises `ValueError("'Apprehensive' is not a valid value. Valid values
are: Calm, Anxious, ...")` on first call, then returns `"OK"` on second call.

Mock sequence: two tool-call responses → plain response.

Asserts:
- The tool-result message after the first call contains the error string (not a Python
  traceback — just `str(exc)`)
- Handler called twice total
- `result.rounds == 3`
- Final content is the plain-response content

---

**`test_unknown_tool_name_returns_error_to_model`**

Mock sequence: tool-call response with `tool_name="nonexistent_tool"` → plain response.

Asserts:
- The tool-result message content starts with `"Unknown tool:"` or similar
- Loop continues (model gets a second chance)
- No Python exception raised from `chat_with_tools()`

---

**`test_cap_reached_returns_cap_reached_true`**

`max_rounds=3`. All mocked responses return a tool-call (never plain content).

Asserts:
- `result.cap_reached` is True
- `result.rounds == 3`
- Exactly 3 HTTP requests made

---

**`test_cap_reached_content_is_empty_string`**

Same setup as above. Asserts `result.content == ""` since the model never returned
plain content.

---

**`test_cap_reached_history_contains_all_rounds`**

`max_rounds=2`. All mocked responses return a tool-call.

Asserts `result.history` contains:
- Original message
- Round 1 assistant tool-call message + tool-result message
- Round 2 assistant tool-call message + tool-result message

(No final assistant message because cap was hit before the model returned content.)

---

**`test_connection_error_propagates`**

`respx` mock raises `httpx.ConnectError`. Asserts `OllamaConnectionError` propagates
from `chat_with_tools()`.

---

**`test_non_200_response_propagates`**

`respx` mock returns HTTP 500. Asserts `OllamaResponseError` propagates.

---

**`test_tool_call_with_none_tool_calls_field`**

Some Ollama responses include `"tool_calls": null` rather than omitting the key entirely.
Mock a plain response with `"tool_calls": null`. Asserts this is treated identically to
no tool calls — `result.rounds == 1`, no handler called.

---

**`test_handler_is_callable_with_arguments_dict`**

Verifies the handler is called with a `dict` (the `arguments` field from the tool call),
not a JSON string. This guards against accidentally double-encoding the arguments.

---

## Test file: `tests/unit/test_tool_gate.py`

### Test cases

All tests import directly from `memories.services.tool_gate`.

---

**`test_create_and_resolve_gate`**

```python
create_gate(session_id=1, turn_id=42)
resolve_gate(session_id=1, turn_id=42, value="Sarah")
result = await await_gate(session_id=1, turn_id=42)
assert result == "Sarah"
```

---

**`test_none_value_resolves_gate`**

Resolves with `None` (user dismissed the card). Asserts `await_gate` returns `None`.

---

**`test_multiple_concurrent_gates`**

Creates gates for `(1, 1)`, `(1, 2)`, and `(2, 1)`. Resolves them with different values.
Asserts each `await_gate` returns the value that was put into that specific gate, not
another gate's value.

---

**`test_cleanup_removes_gate`**

Creates a gate, calls `cleanup_gate`, then asserts that `await_gate` raises `KeyError`
(the queue no longer exists).

---

**`test_resolve_before_await`**

Calls `resolve_gate` before `await_gate`. Asserts `await_gate` returns immediately with
the pre-resolved value (Queue handles this naturally since it buffers one item).

---

**`test_double_resolve_raises`**

Resolves the same gate twice. Asserts the second `resolve_gate` raises
`asyncio.QueueFull`. This guards against accidentally surfacing two blocking cards for
the same turn.

---

**`test_cleanup_idempotent`**

Calls `cleanup_gate` on a gate that does not exist. Asserts no exception is raised
(uses `_pending.pop(..., None)`, so the second call is a no-op).

---

---

## Part C — Live tests against real Ollama

### Why these exist

Unit tests prove the *client* behaves correctly when the mock returns a tool-call
response. They cannot prove that the model actually calls tools rather than:
- Hallucinating the call in prose (`"I would call set_fact with path = ..."`)
- Embedding the call in a JSON code block
- Ignoring the tools list entirely

These behaviours can only be observed on real hardware with a real model. Rather than a
manual smoke test, we write proper tests that are gated behind an environment variable
and can be run any time the model or prompt changes.

### Gating mechanism

A module-level `pytestmark` in each live test file skips all tests in that file unless
`LIVE_OLLAMA=1` is set:

```python
import os
import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("LIVE_OLLAMA"),
    reason="Set LIVE_OLLAMA=1 to run against a real Ollama server"
)
```

Run live tests with:
```bash
LIVE_OLLAMA=1 uv run pytest tests/live/ -v
```

The `live` marker is registered in `pyproject.toml` to avoid unknown-marker warnings.
Live tests are never run by the normal `uv run pytest` invocation (they show up as
`skipped`), so they don't affect CI timing or the 80% coverage threshold.

### New directory: `tests/live/`

```
tests/live/
  conftest.py               # live_ollama and live_model fixtures
  test_tool_calling_live.py # real-model tool-call verification tests
```

### `tests/live/conftest.py`

```python
import os
import pytest
from memories.services.ollama_client import OllamaClient

LIVE_MODEL = os.getenv("LIVE_OLLAMA_MODEL", "jaahas/qwen3.5-uncensored:latest")

@pytest.fixture
async def live_ollama():
    client = OllamaClient()  # reads OLLAMA_BASE_URL from env, defaults to localhost:11434
    yield client
    await client.aclose()

@pytest.fixture
def live_model() -> str:
    return LIVE_MODEL

@pytest.fixture
def set_fact_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "set_fact",
            "description": "Record a fact about the character or setting at the given schema path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dot-notation schema path (e.g. 'Character.State-Of-Mind.Mood')"
                    },
                    "value": {"type": "string"}
                },
                "required": ["path", "value"]
            }
        }
    }
```

### Test file: `tests/live/test_tool_calling_live.py`

All tests use the `live_ollama`, `live_model`, and `set_fact_tool` fixtures. The system
prompt used across tests is a minimal fact-extraction instruction:

```
You are a fact-extraction assistant. When the user's message implies a fact about
the character, call set_fact to record it. Do not describe what you would do —
call the tool. Call it once per distinct fact implied.
```

---

**`test_model_calls_tool_not_prose`**

User message: `"Her mood seems really anxious right now."`

A `call_log` list records every `(path, value)` pair the handler is called with.

Asserts:
- `len(call_log) >= 1` — the model made at least one real tool call
- `result.rounds >= 2` — at least one tool-call round plus the final response
- `result.cap_reached` is False

This is the baseline: if it fails, the model is not calling tools at all.

---

**`test_model_extracts_multiple_facts`**

User message: `"She's wearing a blue surgical scrub top and her mood is calm."`

Asserts:
- `len(call_log) >= 2` — model made at least two tool calls (outfit + mood)
- At least one path contains `"Outfit"` or `"Top"` (case-insensitive)
- At least one path contains `"Mood"` (case-insensitive)

This validates multi-fact extraction within a single turn.

---

**`test_model_retries_after_tool_error`**

User message: `"She's feeling really anxious."`

The handler raises `ValueError("Validation failed on first attempt — please retry")` on
the first call, then returns `"OK"` on all subsequent calls. `call_count` tracks
invocations.

Asserts:
- `call_count >= 2` — model retried after the error
- `result.content` is non-empty — model produced a final response
- `result.cap_reached` is False

This validates that error strings in tool results cause the model to self-correct rather
than giving up or looping indefinitely.

---

**`test_latency_within_acceptable_range`**

User message: `"She seems a bit tired and is wearing a white lab coat."`

Records wall-clock time around the `chat_with_tools()` call.

Asserts total elapsed time is under 60 seconds. This is intentionally generous — the
goal is to catch a complete hang or infinite loop, not enforce a tight performance budget.
If the latency is consistently above ~15 seconds for a 2–3 round exchange, note it in
the step 0 review before proceeding.

---

### Interpreting failures

| Failure | Likely cause | Action |
|---|---|---|
| `call_log` empty, `rounds == 1` | Model not calling tool; embedding in prose or ignoring | Try a different system prompt; check model supports function calling |
| `call_log` empty, cap reached | Model looping on non-tool output | Same as above; may need `stream: false` confirmed |
| Enum value invalid after retry | Model not reading tool error messages | Soften error message wording; check message format |
| Latency > 60s | Model hung or infinite loop | Check `max_rounds` is being enforced; check Ollama connectivity |

If `test_model_calls_tool_not_prose` fails consistently after prompt tuning, the plan's
fallback applies: the World Builder and Character Evaluator must fall back to structured
JSON verdicts (matching the current extractor/evaluator pattern), and plan-v2.md requires
revision before Step 1 begins.

---

## Files changed by this step

| Action | File |
|---|---|
| Add | `src/memories/services/tool_gate.py` |
| Modify | `src/memories/services/ollama_client.py` |
| Modify | `tests/unit/conftest.py` |
| Add | `tests/unit/test_tool_calling.py` |
| Add | `tests/unit/test_tool_gate.py` |
| Add | `tests/live/conftest.py` |
| Add | `tests/live/test_tool_calling_live.py` |
| Modify | `pyproject.toml` (register `live` marker) |

No new routes, no DB changes, no frontend changes.
