# Phase 1 Implementation Plan — Working Skeleton

## Goals (from plan.md)

- FastAPI application serving a single-page Vue 3 UI
- Character LLM call with Facts injected into the system prompt
- Facts stored in SQLite, editable via the sidechannel at any time
- Server-side buffering in place: Ollama response is collected in full before delivery
- Complete response forwarded to the UI over SSE after buffering

**Deliverable:** A working character chatbot where the user manages Facts explicitly and the character speaks from them.

---

## Task List

### T1 — Database layer

**T1.1 Schema initialisation**
Create `init_db()`: runs all `CREATE TABLE IF NOT EXISTS` statements for the complete schema (all tables, not just Phase 1 tables) so later phases never need migrations. Called once at app startup via a FastAPI lifespan event.

**T1.2 Character repository**
`create_character(name, modelfile_base) → Character`
`get_character(id) → Character | None`
`list_characters() → list[Character]`

**T1.3 Facts repository**
`create_fact(character_id, key, value) → Fact` — raises `IntegrityError` on duplicate key per character.
`get_facts(character_id) → list[Fact]`
`update_fact(character_id, key, value) → Fact` — raises `NotFoundError` if key does not exist.
`delete_fact(character_id, key) → None` — raises `NotFoundError` if key does not exist.

**T1.4 Session repository**
`create_session(character_id) → Session` — also creates the initial segment (`boundary_reason="session_start"`, `status="verbatim"`) and stores its id on the session object so that messages can be immediately assigned.
`end_session(session_id) → Session` — sets `ended_at = now()`.
`get_session(session_id) → Session | None`

**T1.5 Message repository**
`store_message(session_id, segment_id, role, content, turn_id) → Message`
`get_messages(session_id) → list[Message]` — ordered by `turn_id` ascending.

**T1.6 Segment repository (minimal)**
`get_or_create_active_segment(session_id) → Segment` — returns the open segment for the session. Phase 5b will add proper boundary logic; for now this always returns the single session_start segment.

---

### T2 — LLM layer

**T2.1 Ollama async client**
Thin async HTTP wrapper around `POST localhost:11434/api/chat`.

- Sends `stream: true` and opens an `httpx` streaming response.
- Reads NDJSON lines; accumulates the `message.content` field from each chunk.
- Returns the fully assembled string and the raw final chunk (which contains `prompt_eval_count` / `eval_count` for Phase 5a).
- The Ollama base URL is read from the `OLLAMA_BASE_URL` environment variable, defaulting to `http://localhost:11434`.
- Raises `OllamaConnectionError` on connection failure; `OllamaResponseError` on non-2xx status.

**T2.2 System prompt builder**
`build_system_prompt(character: Character, facts: list[Fact]) → str`

Produces a prompt of the form:

```
You are {character.name}. Stay in character at all times.

## Your Facts
These are established truths about you. Never contradict them and never invent
details that are not listed here.

{key}: {value}
{key}: {value}
...
```

When `facts` is empty, the Facts section is replaced with a single line:
`No facts have been established yet. Do not invent biographical details.`

---

### T3 — Chat service

**T3.1 Turn orchestration**
`run_turn(session_id, user_content) → str`

1. Load session and character from DB. Raise `NotFoundError` if session unknown or ended.
2. Load all facts for the character.
3. Build the system prompt.
4. Load conversation history (`get_messages(session_id)`).
5. Determine the next `turn_id` (max existing + 1, or 1 for the first turn).
6. Store the user message (`role="user"`, `turn_id=N`).
7. Build the Ollama messages array: `[system_message] + [history] + [current_user_message]`.
8. Call the Ollama client; receive buffered response.
9. Store the assistant message (`role="assistant"`, `turn_id=N`).
10. Return the complete response string.

Facts are loaded from DB on step 2 of every turn — not cached — so mid-conversation fact edits take effect on the next message without requiring a session restart.

---

### T4 — API layer

**T4.1 Characters router** (`/api/characters`)
`POST /api/characters` → 201 `Character`
`GET /api/characters` → 200 `list[Character]`
`GET /api/characters/{id}` → 200 `Character` | 404

**T4.2 Facts router** (`/api/characters/{character_id}/facts`)
`GET /api/characters/{id}/facts` → 200 `list[Fact]`
`POST /api/characters/{id}/facts` → 201 `Fact` | 409 (duplicate key)
`PUT /api/characters/{id}/facts/{key}` → 200 `Fact` | 404
`DELETE /api/characters/{id}/facts/{key}` → 204 | 404

**T4.3 Sessions router** (`/api/sessions`)
`POST /api/sessions` — body: `{character_id}` → 201 `Session` | 404 (unknown character)
`POST /api/sessions/{id}/end` → 200 `Session` | 404
`GET /api/sessions/{id}/messages` → 200 `list[Message]` | 404

**T4.4 Chat router** — SSE endpoint
`POST /api/sessions/{session_id}/messages` — body: `{content: str}`
→ `text/event-stream` | 404 (unknown session) | 409 (session already ended)

SSE event sequence:

```
event: status
data: {"state": "generating"}

event: message
data: {"role": "assistant", "content": "<full response>", "turn_id": <N>}

event: done
data: {}
```

The `status` event is emitted before the Ollama call begins. The `message` event is emitted after the full response has been buffered and stored. Phase 2 will add `{"state": "reviewing"}` between these two events and may withhold the message event on contradiction.

---

### T5 — Frontend

**T5.1 Application shell**
Single `src/memories/frontend/index.html` served as a static file by FastAPI. Vue 3 loaded via CDN. No build step.

**T5.2 Two-panel layout**

```
┌─────────────────────────────────┬──────────────────────┐
│  [Character Name]           ⚙   │                      │
├─────────────────────────────────┤  Facts               │
│                                 │  ─────────────────── │
│  [Chat messages]                │  name: Alice         │
│                                 │  age: 28             │
│  [generating...]                │  [+ Add Fact]        │
│                                 │                      │
│                                 │  [End Session]       │
├─────────────────────────────────┴──────────────────────┤
│  [message input                          ]  [Send]     │
└────────────────────────────────────────────────────────┘
```

**T5.3 Session bootstrap**
On load, the app calls `GET /api/characters` and presents a simple character picker if more than one character exists, or auto-selects the single character. It then creates a session via `POST /api/sessions` and stores the session id in Vue state.

**T5.4 Facts sidechannel**
On session start, fetches and displays all facts. Each fact row has an inline edit input and a delete button. The Add Fact form has key and value fields. All mutations hit the facts API and refresh the list on success.

**T5.5 Chat channel and SSE consumer**
On `Send`, the app opens an `EventSource`-style connection using the `fetch` API with a `ReadableStream` body (since `EventSource` does not support `POST`). It listens for the sequence of events described in T4.4:

- On `status` event with `state: "generating"`: show a "generating..." indicator.
- On `message` event: hide the indicator, append the assistant message to the chat.
- On `done`: close the stream.

Input is disabled and the Send button shows a spinner while the stream is open.

---

## Technology & Infrastructure Decisions

### D1 — Raw `aiosqlite` over SQLAlchemy

The memory model has a well-defined, stable schema; we control all queries; and the schema uses SQLite-specific features (`TEXT` for JSON arrays, `BLOB` for embeddings in later phases). An ORM adds indirection without benefit here. All queries are written as parameterised SQL strings; results are mapped to Pydantic models by the repository layer.

### D2 — SSE over WebSocket for streaming

The chat interaction is strictly server-push after each user message: one request, one response stream. SSE is a better fit than WebSocket for this pattern — it uses plain HTTP, reconnects automatically on drop, and is simpler to test (the response body is a text stream). The frontend uses `fetch` + `ReadableStream` rather than the native `EventSource` API because `EventSource` does not support `POST` bodies.

### D3 — `stream: true` with server-side buffer

The Ollama request uses `stream: true` from Phase 1 forward. The server collects all NDJSON chunks before delivering the result. This is the Phase 2 extension point: the evaluator call will be inserted between "buffer complete" and "deliver to client". If Phase 1 used `stream: false`, switching to streaming in Phase 2 would require restructuring the client.

### D4 — Stateless API; DB is the only session state

No in-memory session cache. Every turn reloads the character, facts, and conversation history from SQLite. This keeps the API horizontally scalable (irrelevant for local use, but avoids building in assumptions) and makes every request independently testable against a known DB state. The cost is a handful of extra queries per turn, which is negligible compared to LLM latency.

### D5 — Full schema initialised at Phase 1

`init_db()` creates all tables (including `inferences`, `experiences`, `decisions`, `segments`) at startup. Tables unused in Phase 1 sit empty. This avoids ALTER TABLE migrations between phases, which are painful with SQLite. The schema in plan.md is the target; this is it.

The one adjustment to the schema for Phase 1: `messages.segment_id` is NOT NULL in the plan but Phase 1 has no segment boundary logic. We will set it NOT NULL and fulfil the constraint by creating a single `session_start` segment when each session is created and assigning all messages to it. Phase 5b will add proper boundary detection on top of this.

### D6 — Fact reloaded per turn, not cached

Facts can be added, edited, or deleted from the sidechannel at any point, including mid-conversation. Loading from DB per turn means the next message always reflects the current fact state without requiring a session restart or a cache invalidation event. Given that fact counts are small (tens, not thousands), this is the right default.

### D7 — Environment configuration via `os.getenv`

Phase 1 needs two runtime settings: `OLLAMA_BASE_URL` and `OLLAMA_MODEL` (the default model used when no `current_model_name` is set on a character). Both are read from environment variables with sane defaults. A `.env` file can be used for local development; python-dotenv is already available as a transitive dependency.

---

## Test Plan

Tests are written first. The implementation is complete when all tests pass and coverage stays at or above 80% overall (90% for `services/`).

Each integration test gets a fresh in-memory SQLite database via a pytest fixture. The fixture:
1. Opens an `aiosqlite.connect(":memory:")` connection.
2. Calls `init_db(conn)` to create all tables.
3. Overrides the FastAPI database dependency so the test client uses this connection.
4. Tears down after each test.

Ollama HTTP calls are mocked with `respx`. No real Ollama instance is required.

---

### Unit tests — `tests/unit/`

#### `test_prompt_builder.py`

| Test | Asserts |
|------|---------|
| `test_character_name_appears_in_prompt` | The character's name is present in the returned string |
| `test_all_facts_injected_as_key_value_pairs` | Every `key: value` pair from the fact list appears verbatim |
| `test_fact_order_preserved` | Facts appear in the order they were passed |
| `test_no_facts_yields_no_invention_instruction` | The empty-facts branch produces the "do not invent" fallback line |
| `test_facts_section_absent_when_empty` | The `## Your Facts` header still appears; only the list body changes |

#### `test_ollama_client.py`

| Test | Asserts |
|------|---------|
| `test_request_sends_model_and_messages` | The mocked Ollama endpoint receives `model` and `messages` in the body |
| `test_request_uses_stream_true` | Request body contains `"stream": true` |
| `test_chunks_are_concatenated` | Multiple NDJSON chunks are assembled into a single string |
| `test_returns_final_chunk_metadata` | The return value includes `prompt_eval_count` and `eval_count` from the last chunk |
| `test_raises_ollama_connection_error_on_network_failure` | `httpx.ConnectError` is wrapped and re-raised as `OllamaConnectionError` |
| `test_raises_ollama_response_error_on_non_200` | A 500 response raises `OllamaResponseError` |

#### `test_chat_service.py`

| Test | Asserts |
|------|---------|
| `test_system_message_is_first_in_ollama_request` | The messages array sent to Ollama has `role="system"` at index 0 |
| `test_history_included_in_ollama_request` | Prior user and assistant messages appear after the system message |
| `test_history_ordered_by_turn_id` | Messages are in ascending turn_id order |
| `test_new_user_message_appended_last` | The current user message is the final element in the messages array |
| `test_facts_reflected_in_system_message` | System message content contains a fact key-value pair from DB |
| `test_user_message_stored_before_llm_call` | If the Ollama mock raises, the user message is still in the DB |
| `test_assistant_message_stored_after_llm_call` | Successful turn writes an assistant message to the DB |
| `test_turn_ids_increment` | Two successive turns produce `turn_id=1` and `turn_id=2` |
| `test_run_turn_raises_on_unknown_session` | `NotFoundError` raised for a session id that does not exist |
| `test_run_turn_raises_on_ended_session` | `NotFoundError` raised when `ended_at` is set |

---

### Integration tests — `tests/integration/`

#### `test_db_init.py`

| Test | Asserts |
|------|---------|
| `test_all_tables_created` | After `init_db()`, all eight table names are present in `sqlite_master` |
| `test_init_is_idempotent` | Calling `init_db()` twice on the same connection does not raise |

#### `test_characters_repo.py`

| Test | Asserts |
|------|---------|
| `test_create_character_returns_with_id` | Returned `Character` has a positive integer `id` |
| `test_get_character_by_id` | Fetched character matches what was inserted |
| `test_get_nonexistent_character_returns_none` | `get_character(9999)` returns `None` |
| `test_list_characters_empty` | Returns empty list on clean DB |
| `test_list_characters_multiple` | Returns all created characters |

#### `test_facts_repo.py`

| Test | Asserts |
|------|---------|
| `test_create_fact_stores_key_and_value` | Inserted fact is retrievable with matching key and value |
| `test_create_fact_duplicate_key_raises` | Second insert with the same `(character_id, key)` raises `IntegrityError` |
| `test_get_facts_returns_only_own_character` | Facts for character A are not returned when querying character B |
| `test_update_fact_changes_value` | After update, `get_facts` returns the new value for that key |
| `test_update_nonexistent_fact_raises` | `NotFoundError` raised for a key that does not exist |
| `test_delete_fact_removes_it` | After delete, the key is absent from `get_facts` |
| `test_delete_nonexistent_fact_raises` | `NotFoundError` raised |

#### `test_sessions_repo.py`

| Test | Asserts |
|------|---------|
| `test_create_session_sets_character_id` | Session's `character_id` matches the one passed in |
| `test_create_session_creates_initial_segment` | A segment with `boundary_reason="session_start"` exists for the new session |
| `test_end_session_sets_ended_at` | `ended_at` is non-null after `end_session` |
| `test_get_session_by_id` | Returns session matching the created one |
| `test_get_nonexistent_session_returns_none` | `get_session(9999)` returns `None` |

#### `test_messages_repo.py`

| Test | Asserts |
|------|---------|
| `test_store_user_message` | Message is retrievable with `role="user"` and correct content |
| `test_store_assistant_message` | Message is retrievable with `role="assistant"` |
| `test_get_messages_ordered_by_turn_id` | Messages returned in ascending `turn_id` order regardless of insert order |
| `test_messages_isolated_per_session` | Messages for session A are not returned when querying session B |
| `test_messages_reference_segment` | Each stored message has a non-null `segment_id` |

#### `test_api_characters.py`

| Test | Asserts |
|------|---------|
| `test_create_character_201` | POST returns 201 and a JSON body with `id`, `name` |
| `test_list_characters_empty_200` | GET returns 200 and `[]` on empty DB |
| `test_list_characters_populated` | GET returns all created characters |
| `test_get_character_200` | GET `/api/characters/{id}` returns the right character |
| `test_get_character_404` | GET for unknown id returns 404 |

#### `test_api_facts.py`

| Test | Asserts |
|------|---------|
| `test_add_fact_201` | POST returns 201 and the created fact |
| `test_add_fact_duplicate_key_409` | Inserting the same key twice returns 409 |
| `test_list_facts_for_character` | GET returns all facts for that character |
| `test_update_fact_200` | PUT returns 200 with updated value |
| `test_update_nonexistent_fact_404` | PUT for unknown key returns 404 |
| `test_delete_fact_204` | DELETE returns 204 |
| `test_delete_nonexistent_fact_404` | DELETE for unknown key returns 404 |
| `test_facts_for_unknown_character_404` | GET facts for a non-existent character returns 404 |

#### `test_api_sessions.py`

| Test | Asserts |
|------|---------|
| `test_start_session_201` | POST returns 201 with session id |
| `test_start_session_unknown_character_404` | POST with bad `character_id` returns 404 |
| `test_end_session_200` | POST `/end` returns 200 |
| `test_end_unknown_session_404` | POST `/end` on bad id returns 404 |
| `test_get_session_messages_initially_empty` | GET messages returns `[]` for a new session |

#### `test_api_chat.py`

| Test | Asserts |
|------|---------|
| `test_send_message_content_type_is_event_stream` | Response header `Content-Type: text/event-stream` |
| `test_send_message_emits_status_event_first` | First SSE event is `event: status` with `state: "generating"` |
| `test_send_message_emits_message_event` | An `event: message` event containing `role` and `content` is present |
| `test_send_message_emits_done_event_last` | Final SSE event is `event: done` |
| `test_send_message_stores_user_message` | After the request, DB contains user message with correct content |
| `test_send_message_stores_assistant_response` | After the request, DB contains assistant message with mocked Ollama content |
| `test_ollama_receives_system_message_with_facts` | Captured Ollama request has a `role: "system"` message containing a known fact's key-value pair |
| `test_ollama_receives_prior_history` | Second message includes the first turn's user and assistant messages in the Ollama request body |
| `test_send_to_unknown_session_404` | POST to non-existent session id returns 404 |
| `test_send_to_ended_session_409` | POST after `end_session` returns 409 |

---

## Not In Scope for Phase 1

The following are intentionally deferred:

- **Evaluator LLM call** — Phase 2. The buffer-then-deliver structure is in place; the evaluator slots in between.
- **Inferences** — Phase 3. The `inferences` table exists but stays empty.
- **Experiences** — Phase 4. The `experiences` table exists but stays empty.
- **Decisions log** — Phase 2. The `decisions` table exists but stays empty.
- **Token counting and compression** — Phase 5a/5b. `prompt_eval_count`/`eval_count` values from Ollama responses are returned by the client (to avoid losing them) but not yet persisted or acted on.
- **Segment boundaries** — Phase 5b. A single `session_start` segment covers all messages in Phase 1.
- **Modelfile export** — Phase 6 stretch.
- **Playwright E2E tests** — deferred until the UI is stable enough to be worth the setup cost. Phase 1 tests cover the API and frontend-facing contracts; visual interaction is verified manually.
- **Phone-responsive layout** — Phase 6. The two-panel layout targets a laptop browser.
