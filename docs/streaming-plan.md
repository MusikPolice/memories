# Streaming Plan — Option A: Optimistic Streaming + Async Evaluate

## Goal

Stream character response tokens to the client as they are generated, rather than buffering
the full response until after the evaluator has run. The evaluator still runs, and if it finds
a contradiction the client is notified and the corrected response is streamed immediately after.
Users see the first token within seconds of sending a message instead of waiting 10–20 s for
the full generate+evaluate cycle to complete.

This is a research project; jarring visible corrections are acceptable. Contradiction rate is
expected to fall over time as prompts and models improve.

---

## New SSE Event Sequence

The server-side event contract changes in two ways: `event: token` is new, and the timing of
existing events shifts.

**Current sequence (Option B / buffer-everything):**

```
status(extracting)
status(generating)
status(reviewing)
[if contradiction ×N:]
  sidechannel(contradiction)
  status(regenerating)
  status(reviewing)
[if think:]
  thinking
message
[sidechannels: implication / new_inference_probabilistic / experience_update / extraction_applied / implicit_fact_proposed]
done
```

**New sequence (Option A / optimistic streaming):**

```
status(extracting)
status(generating)
token(t₁) … token(tₙ)          ← character tokens streamed as they arrive; first attempt
status(reviewing)
[if contradiction ×N:]
  sidechannel(contradiction)    ← signals client to mark current partial as rejected
  status(regenerating)
  status(generating)
  token(t₁) … token(tₙ)        ← retry tokens; client starts a new streaming bubble
  status(reviewing)
[if think:]
  thinking                      ← still buffered; emitted only for the final clean attempt
message                         ← completion event; carries turn_id, metadata; content is
                                   already rendered — client merges metadata into the bubble
[sidechannels: same as before]
done
```

All tokens for ALL attempts are streamed. The frontend visually distinguishes rejected
attempts (see §Frontend below). Thinking tokens are kept buffered for now; streaming them
is future work.

---

## Server Changes

### 1. `ollama_client.py` — add `on_token` callback to `chat()`

**What changes:**

Add an optional `on_token: Callable[[str], Awaitable[None]] | None = None` parameter to
`chat()`. Inside the streaming loop, call `await on_token(token)` for each non-empty content
token, before appending it to `parts`. Thinking tokens do not trigger `on_token`.

```python
async def chat(
    self,
    model: str,
    messages: list[dict[str, str]],
    think: bool = False,
    format: dict[str, Any] | str | None = None,
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, dict[str, Any]]:
    ...
    async with self._http.stream("POST", url, json=payload) as response:
        ...
        async for line in response.aiter_lines():
            if line.strip():
                chunk = json.loads(line)
                msg = chunk.get("message", {})
                if isinstance(msg, dict):
                    thought = msg.get("thinking", "") or ""
                    if thought:
                        thinking_parts.append(thought)
                    token = msg.get("content", "") or ""
                    if token:
                        parts.append(token)
                        if on_token is not None:
                            await on_token(token)   # ← new
                last_chunk = chunk
```

No other changes to `ollama_client.py`.

**Tests to add in `test_ollama_client.py`:**
- `on_token` callback is called once per content chunk, in order
- `on_token` is not called for thinking chunks
- `on_token` is not called when not provided (no regression)
- `on_token` receiving an exception does not swallow it

---

### 2. `chat_service.py` — thread `on_token` through the contradiction loop

**What changes:**

`run_contradiction_loop()` receives `on_token` and passes it to `ollama.chat()` on every
attempt (not just the first). The caller (the SSE generator in `chat.py`) decides the
callback; it can emit different SSE events or include attempt metadata if needed.

```python
async def run_contradiction_loop(
    ...
    on_token: Callable[[str], Awaitable[None]] | None = None,
) -> tuple[str, str, EvaluatorResult]:
    ...
    for attempt in range(max_retries + 1):
        ...
        raw_content, metadata = await ollama.chat(
            model, messages, think=think, on_token=on_token
        )
```

`run_turn()` receives `on_token` and passes it through to `run_contradiction_loop()`:

```python
async def run_turn(
    db, session_id, user_content, ollama,
    think=False, on_status=None, on_token=None,   # ← new
) -> ...:
    ...
    char_content, char_thinking, eval_result = await run_contradiction_loop(
        ...,
        on_token=on_token,   # ← new
    )
```

**Tests to add in `test_chat_service.py`:**
- `on_token` callback is called for character tokens during `run_contradiction_loop`
- `on_token` is called on retry attempts as well as the first attempt
- `on_token` not provided → no regression (existing tests unchanged)

---

### 3. `chat.py` — unified event queue, token forwarding, contradiction interlock

This is the most significant server change.

**Unified event queue**

Replace the current single `asyncio.Queue[str]` (status only) with a unified queue of tagged
tuples. All callbacks — status and token — enqueue into it:

```python
_q: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

async def _on_status(state: str) -> None:
    await _q.put(('status', state))

async def _on_token(token: str) -> None:
    await _q.put(('token', token))
```

**Drain loop**

```python
while not _task.done():
    try:
        kind, data = await asyncio.wait_for(_q.get(), timeout=0.05)
        if kind == 'status':
            yield f'event: status\ndata: {{"state": "{data}"}}\n\n'
        elif kind == 'token':
            yield f'event: token\ndata: {json.dumps({"content": data})}\n\n'
    except TimeoutError:
        pass
```

**Contradiction interlock**

The contradiction notification sidechannel is currently emitted *after* `run_turn()` returns
(from `eval_result.contradiction_notifications`). With Option A the frontend needs to know a
contradiction occurred *before* the retry tokens start streaming, so it can close the rejected
bubble and open a new one.

There are two ways to solve this:

*Option A1 — inject into the event queue from the service layer*: `run_contradiction_loop()`
calls a new `on_contradiction` callback (same queue, different tag) each time it finds a
contradiction, before the retry's `on_token` calls begin. The existing
`contradiction_notifications` accumulator can stay for the post-run sidechannel.

*Option A2 — insert a sentinel token into the token stream*: Simpler. When a contradiction
is found, enqueue a special `('contradiction', description)` item into the unified queue.
The drain loop emits `event: sidechannel` for it in-line, same as post-run contradictions.
Remove that entry from `eval_result.contradiction_notifications` so it is not emitted again
after `run_turn()` completes.

**Recommendation: Option A2** — one queue, one drain loop, minimal interface changes.

The updated `run_contradiction_loop()` signature:

```python
async def run_contradiction_loop(
    ...
    on_token: Callable[[str], Awaitable[None]] | None = None,
    on_contradiction: Callable[[int, str], Awaitable[None]] | None = None,
) -> tuple[str, str, EvaluatorResult]:
    ...
    for attempt in range(max_retries + 1):
        ...
        if ev.verdict == "contradiction":
            for v in ev.violations:
                if v.type == "contradiction":
                    if on_contradiction:
                        await on_contradiction(attempt + 1, v.description)
                    ...
```

In `chat.py`:

```python
async def _on_contradiction(iteration: int, description: str) -> None:
    sc = json.dumps({"type": "contradiction", "iteration": iteration, "description": description})
    await _q.put(('sidechannel', sc))
    await _q.put(('status', 'regenerating'))
    await _q.put(('status', 'generating'))
```

After `run_turn()` returns, the existing loop that emits `contradiction_notifications` is
removed (those notifications were already emitted live). All other post-run events (thinking,
message, other sidechannels, done) remain unchanged.

**Tests to add in `tests/integration/test_api_chat.py` (or equivalent):**
- Token events appear in the SSE stream before the `message` event
- A contradiction mid-stream produces: in-line `sidechannel(contradiction)` + `status(regenerating)` + `status(generating)` + retry token events, then `message`
- No duplicate contradiction sidechannels (live + post-run)
- Clean response (no contradiction): `token` events, then `message`, then `done`; no contradiction sidechannel

---

## Frontend Changes

### 4. `chat.js` — new SSE state label

Add `streaming` to `_STATE_LABELS` (optional — only if a new visible status state is desired
during token streaming; the existing `generating` label may suffice).

No changes to `buildNotificationFromSidechannel` — the `contradiction` case already exists
and will now fire during streaming rather than post-run.

**Tests:** update `chat.test.js` if `_STATE_LABELS` is extended.

---

### 5. `chat-component.js` — streaming bubble state machine

This is the most significant frontend change.

**New reactive state:**

```js
const streamingMessage = ref(null);   // the in-progress assistant message object, or null
```

`streamingMessage` holds a reference to the message object currently being accumulated.
It is also present in `messages` (pushed by reference), so Vue reactivity propagates
content updates directly.

**Token event handler** (inside the SSE `for` block):

```js
} else if (eventName === 'token' && dataStr) {
  const { content } = JSON.parse(dataStr);
  if (!streamingMessage.value) {
    // First token of a new attempt — push a fresh bubble.
    streamingMessage.value = { role: 'assistant', content: '', streaming: true };
    messages.value.push(streamingMessage.value);
  }
  streamingMessage.value.content += content;
  await scrollToBottom();
```

**Contradiction sidechannel during streaming**

When `sidechannel(contradiction)` arrives and `streamingMessage.value` is non-null, the
current bubble is a rejected attempt. Mark it visually and clear the streaming ref so the
next `token` event opens a new bubble:

```js
} else if (eventName === 'sidechannel' && dataStr) {
  const payload = JSON.parse(dataStr);
  if (payload.type === 'contradiction' && streamingMessage.value) {
    // Seal the rejected bubble before the notification card is inserted.
    streamingMessage.value.streaming = false;
    streamingMessage.value.contradicted = true;
    streamingMessage.value = null;
  }
  const notif = buildNotificationFromSidechannel(payload);
  ...
```

**Message event — merge metadata into the streaming bubble**

The `message` event now carries metadata for an already-rendered bubble. Instead of pushing
a new message, update the existing one:

```js
} else if (eventName === 'message' && dataStr) {
  const payload = JSON.parse(dataStr);
  generating.value = false;
  statusText.value = '';
  if (streamingMessage.value) {
    // Seal and annotate the streaming bubble.
    Object.assign(streamingMessage.value, {
      streaming: false,
      turn_id: payload.turn_id,
      contradictionExhausted: payload.contradiction_exhausted || false,
    });
    streamingMessage.value = null;
  } else {
    // Fallback: no streaming bubble (e.g. evaluator returned before any tokens).
    messages.value.push({
      role: 'assistant',
      content: payload.content,
      turn_id: payload.turn_id,
      contradictionExhausted: payload.contradiction_exhausted || false,
    });
  }
  if (payload.active_experience_ids)
    activeExperienceIds.value = new Set(payload.active_experience_ids);
  if (payload.experience_scores)
    experienceScoreMap.value = buildScoreMap(payload.experience_scores);
  await scrollToBottom();
```

**Cleanup** — reset `streamingMessage` in the `finally` block:

```js
} finally {
  streamingMessage.value = null;
  generating.value = false;
  ...
```

**Tests to add in `chat-component.test.js`:**
- First `token` event pushes a new `{ role: 'assistant', streaming: true }` message
- Subsequent `token` events append to the same message (not push new ones)
- `message` event seals the bubble (sets `streaming: false`, merges `turn_id`)
- `sidechannel(contradiction)` during streaming sets `contradicted: true` on the current bubble and clears `streamingMessage`
- Next `token` after contradiction opens a new bubble
- `finally` block clears `streamingMessage` on error/abort

---

### 6. `index.html` — streaming bubble visual treatment

Two new visual states on assistant message bubbles:

**Streaming cursor** (`streaming: true`): show a blinking `▌` at the end of the content.
Simple CSS animation, no JS required.

```html
<span v-if="msg.streaming" class="streaming-cursor">▌</span>
```

**Contradicted bubble** (`contradicted: true`): visually de-emphasise the rejected response.
Use reduced opacity and a small label:

```html
<div v-if="msg.contradicted" class="contradiction-label">↻ contradicted — see correction below</div>
```

Whether the contradicted bubble is collapsed by default or fully visible is a UX call; either
works. Collapsing is more transparent about what happened; leaving it visible is more
informative for debugging/research purposes.

---

## Implementation Order

Each step can be reviewed and tested independently before the next begins:

1. **`ollama_client.py`** — add `on_token` callback; add unit tests. No user-visible change.
2. **`chat_service.py`** — thread `on_token` and `on_contradiction` through the loop; add
   unit tests. No user-visible change (callbacks not wired yet).
3. **`chat.py`** — unified event queue; wire `_on_token` and `_on_contradiction`; remove the
   post-run contradiction loop; emit `event: token`. Add integration tests. Server now
   streams tokens; frontend ignores them until step 4.
4. **`chat-component.js`** — streaming bubble state machine; add component tests.
5. **`index.html`** — streaming cursor and contradicted bubble styles.
6. **`chat.js` / `chat.test.js`** — add any new label or helper coverage.

---

## Open Questions

1. **Stream thinking tokens?** Currently thinking text is buffered and emitted as a single
   `event: thinking` before `event: message`. Streaming thinking tokens would require a second
   on-screen bubble type (`role: 'thinking', streaming: true`). Deferred — the UX value is
   low since thinking text is already collapsed by default.

2. **Collapsed vs. visible contradicted bubbles?** For a research tool, leaving the rejected
   response visible (but dimmed) is more informative. The plan above assumes visible.

3. **`event: token` data shape.** The plan uses `{"content": "..."}` to mirror the `message`
   event's field name. An alternative is bare text (`data: the token text\n\n`) which is
   slightly smaller but inconsistent with every other event in the stream. Stick with JSON.

4. **Retry token tagging.** The plan streams retry tokens into new bubbles (via the
   `contradiction` sidechannel clearing `streamingMessage`). An alternative is to tag each
   `token` event with `{"content": "...", "attempt": 2}` so the frontend can style retries
   differently without relying on the contradiction event as a separator. Not required for
   the initial implementation.
