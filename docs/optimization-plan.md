# Performance Optimization Plan

Every turn in this app makes **at minimum four sequential Ollama calls** — embed, extract, character, evaluate — plus unbounded database reads and growing context windows. The slowness is architectural, not just hardware. The proposals below are ranked by impact on perceived latency and are independent of each other unless noted.

---

## 1. Parallelize embed and extraction LLM calls — HIGH impact

**What's happening**

`run_turn()` calls these two operations sequentially:
- [`retrieve_experiences()` → `ollama.embed()`](../src/memories/services/chat_service.py) (line 177)
- [`run_fact_extractor()` → `ollama.chat()`](../src/memories/services/chat_service.py) (line 192)

Neither depends on the other's result. The extraction needs `facts` and `inferences`, which are already loaded by line 169. The embed call needs only `user_content`.

**Fix**

```python
active_exps_task = asyncio.create_task(
    retrieve_experiences(db, session.character_id, user_content, ollama, ...)
)
extraction_task = asyncio.create_task(
    run_fact_extractor(user_content, character, facts, inferences, ollama)
)
(active, experience_scores), extraction_result = await asyncio.gather(
    active_exps_task, extraction_task
)
```

**Expected saving:** on a 3B model, extraction typically takes 2–5 s. Embed on `nomic-embed-text` takes ~0.3–1 s. Running them in parallel eliminates whichever is shorter from the critical path.

---

## 2. Parallelize DB queries at turn start — MEDIUM/HIGH impact

**What's happening**

[`run_turn()`](../src/memories/services/chat_service.py) issues eight sequential `await` calls before any LLM work starts (lines 159–223):

```python
session    = await get_session(...)
character  = await get_character(...)
facts      = await get_facts(...)
inferences = await get_inferences(...)
turn_id    = await next_turn_id(...)
# ... then retrieve_experiences/extraction ...
history    = await get_messages(...)
segment    = await get_active_segment(...)
```

`facts`, `inferences`, `history`, and `segment` do not depend on each other and can all run simultaneously. `character` depends on `session`, but nothing else depends on `character` before line 165.

**Fix**

```python
session = await get_session(db, session_id)
character, facts, inferences, history, segment, turn_id = await asyncio.gather(
    get_character(db, session.character_id),
    get_facts(db, session.character_id),
    get_inferences(db, session.character_id),
    get_messages(db, session_id),
    get_active_segment(db, session_id),
    next_turn_id(db, session_id),
)
```

SQLite's WAL mode handles concurrent reads fine. With aiosqlite (thread-per-connection), each `await` yields to the executor — gathering them removes the serial wait.

---

## 3. Stream character tokens to the UI while evaluating in background — HIGH perceived-latency impact

**What's happening**

[`ollama.chat()`](../src/memories/services/ollama_client.py) uses `stream: true` at the HTTP level but **buffers the entire response before returning** (lines 85–88). The user sees "generating…" for the full inference time, then gets the complete message all at once.

**Options**

*Option A — Optimistic streaming + async evaluate (recommended)*: Stream character tokens to the client immediately via a new SSE event type (`event: token`). Concurrently run the evaluator on the assembled response. If the evaluator finds a contradiction, emit a `sidechannel(contradiction)` event and send a corrected message in a second `event: message`. This changes the UX from "wait 20s then read" to "read as it types, rarely see a correction".

*Option B — Two-phase stream*: Deliver the complete first response, then stream a second response if a contradiction is found. Less invasive to the frontend.

**Codebase impact**

The buffering is entirely in [`ollama_client.py:62–88`](../src/memories/services/ollama_client.py). The evaluator prompt in [`evaluator.py`](../src/memories/services/evaluator.py) already receives the full response as a string — it doesn't need to change. The main work is splitting [`run_contradiction_loop()`](../src/memories/services/chat_service.py) into a streaming phase and a parallel evaluator phase.

---

## 4. The extractor is the hidden third LLM call — MEDIUM/HIGH impact

**What's happening**

[`run_fact_extractor()`](../src/memories/services/extraction_service.py) is a full Ollama inference call (line 210) that runs before the character call on every single turn, even when the user message is conversational filler with no factual content ("okay thanks", "hmm tell me more"). The status event emitted is `"extracting"` but the user has no indication that an LLM call is happening here.

**Fixes**

1. **Skip extraction on obviously non-factual turns** with a lightweight heuristic (message length < 20 chars, or a quick keyword check) before firing the LLM.
2. **Parallelize with embed** (covered in item 1 above).
3. **Add a `"skip_extraction"` flag** to the request body so the frontend can skip it for subsequent messages once a session is established.

---

## 5. Limit message history sent to Ollama — HIGH long-session impact

**What's happening**

[`get_messages()`](../src/memories/database.py) returns all messages for the session with no `LIMIT` clause (lines 422–428). [`run_turn()`](../src/memories/services/chat_service.py) appends every message to `base_messages` (lines 236–237). A 50-turn session sends 100 messages (~10K+ tokens) to Ollama on every turn, growing quadratically in cost and linearly in latency.

**Fix**

Apply a sliding window: only include the last N turns (e.g. 20). The segments table already exists for more sophisticated compression later, but a simple window is a one-liner:

```python
# chat_service.py:236
for msg in history[-40:]:   # last 20 turns = 40 messages
    base_messages.append({"role": msg.role, "content": msg.content})
```

---

## 6. Candidate experience embeddings are loaded and deserialized on every turn — MEDIUM impact

**What's happening**

[`retrieve_experiences()`](../src/memories/services/experience_service.py) calls [`get_experiences_with_embeddings(db, character_id)`](../src/memories/database.py) (lines 673–690) which loads every stored experience including full embedding blobs from the DB on every turn, even with `TOP_K_EXPERIENCES=5`. With 500 stored experiences, this deserializes 500 JSON embedding vectors per turn.

**Fixes**

1. **In-memory embedding cache**: Build a module-level `dict[int, list[float]]` keyed by `(character_id, experience_id)` and invalidate on writes. Embed deserialization is the main cost.
2. **FAISS / sqlite-vec**: For larger experience stores, use an approximate nearest-neighbour index that avoids scanning all candidates.
3. **Instrumentation first**: Add a `time.monotonic()` wrapper around `get_experiences_with_embeddings()` to measure actual wall time before investing in the cache.

The dot-product loop ([`experience_service.py:71`](../src/memories/services/experience_service.py)) uses a pure Python `sum(x * y for x, y in zip(...))` — switching to `numpy.dot()` would be 10–50× faster for large vectors, though the deserialization dominates at scale.

---

## 7. Reuse the httpx client across Ollama calls — LOW/MEDIUM impact

**What's happening**

Every [`ollama.chat()`](../src/memories/services/ollama_client.py), [`ollama.embed()`](../src/memories/services/ollama_client.py), and [`ollama.warmup()`](../src/memories/services/ollama_client.py) call creates a new `httpx.AsyncClient` context manager (lines 62–64, 101–103, 125–127). This means a new connection pool — and on HTTP/1.1, a new TCP connection — per call. With 4 calls per turn, that's 4 TCP setup/teardowns on loopback.

**Fix**

Instantiate a single `httpx.AsyncClient` per application lifetime, stored on `OllamaClient` itself:

```python
# ollama_client.py
def __init__(self, base_url: str | None = None) -> None:
    self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    self._http = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
    )

async def aclose(self) -> None:
    await self._http.aclose()
```

Then call `await ollama.aclose()` in the FastAPI lifespan cleanup. With localhost loopback the TCP saving is small (~0.5–2 ms per call), but connection reuse also enables HTTP/2 multiplexing if Ollama ever adds support.

---

## 8. Replace the status-poll busy-wait in the SSE generator — LOW impact

**What's happening**

[`chat.py:54–59`](../src/memories/routers/chat.py) polls the status queue with `get_nowait()`, catches `QueueEmpty` on every iteration, then yields with `await asyncio.sleep(0)`:

```python
while not _task.done():
    try:
        state = _q.get_nowait()
        yield f'event: status\ndata: {{"state": "{state}"}}\n\n'
    except asyncio.QueueEmpty:
        await asyncio.sleep(0)
```

`sleep(0)` is a cooperative yield but it runs on every event loop iteration where no status is queued. Status events may also be delayed by up to one event loop tick after being enqueued.

**Fix**

```python
while not _task.done():
    try:
        state = await asyncio.wait_for(_q.get(), timeout=0.05)
        yield f'event: status\ndata: {{"state": "{state}"}}\n\n'
    except asyncio.TimeoutError:
        pass
```

This eliminates the exception-per-tick pattern and delivers status updates promptly when they arrive.

---

## 9. Add a partial index on experiences — LOW impact (future-proof)

**What's happening**

`experiences` has only a single-column index on `character_id` ([`database.py:87`](../src/memories/database.py)). The query in `get_experiences_with_embeddings` filters `WHERE character_id = ? AND embedding IS NOT NULL`. SQLite can use the `character_id` index but then must scan all rows for that character to filter `embedding IS NOT NULL`.

**Fix**

```sql
CREATE INDEX IF NOT EXISTS idx_experiences_character_embedding
    ON experiences(character_id) WHERE embedding IS NOT NULL;
```

This is a partial index; SQLite will use it for the exact query pattern used in retrieval. Impact is low now but becomes significant at 10K+ experiences.

---

## Summary

| # | Change | Files affected | Impact | Effort |
|---|--------|---------------|--------|--------|
| 1 | Parallelize embed + extraction | [`chat_service.py:177–219`](../src/memories/services/chat_service.py) | High | Low |
| 2 | Parallelize DB reads at turn start | [`chat_service.py:159–223`](../src/memories/services/chat_service.py) | Med–High | Low |
| 3 | Stream character tokens, evaluate async | [`ollama_client.py`](../src/memories/services/ollama_client.py), [`chat_service.py`](../src/memories/services/chat_service.py), [`chat.py`](../src/memories/routers/chat.py) | High | High |
| 4 | Skip extraction on trivial messages | [`chat_service.py:192`](../src/memories/services/chat_service.py), [`extraction_service.py`](../src/memories/services/extraction_service.py) | Med | Low |
| 5 | Window message history (last N turns) | [`chat_service.py:236`](../src/memories/services/chat_service.py) | High (long sessions) | Trivial |
| 6 | Cache experience embeddings in memory | [`experience_service.py:140`](../src/memories/services/experience_service.py), [`database.py:673`](../src/memories/database.py) | Med | Low |
| 7 | Reuse httpx client across calls | [`ollama_client.py:33–133`](../src/memories/services/ollama_client.py) | Low | Low |
| 8 | Replace status poll busy-wait | [`chat.py:54–59`](../src/memories/routers/chat.py) | Low | Low |
| 9 | Partial index on experiences | [`database.py:87`](../src/memories/database.py) | Low now, high later | Trivial |

Items 1, 2, 5, and 8 can all be done in under an hour with no architectural risk. Item 3 (streaming + async eval) is the biggest UX win but requires the most care — it changes the observable contract between the server and frontend.
