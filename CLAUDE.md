# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (requires 80% coverage)
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_prompt_builder.py

# Run a single test by name
uv run pytest -k test_character_name_appears_in_prompt

# Run tests without coverage (faster iteration)
uv run pytest --no-cov

# Lint and format
uv run ruff check src/
uv run ruff format src/

# Type check (strict, src/ only — tests/ are excluded)
uv run mypy src/

# Security scan
uv run bandit -c pyproject.toml -r src/

# Run all pre-commit hooks at once
uv run pre-commit run --all-files

# Start the dev server
uv run uvicorn memories.main:app --reload --host 0.0.0.0 --port 8000

# Start the dev server with verbose (DEBUG) logging — surfaces LLM prompts and tool payloads
LOG_LEVEL=DEBUG uv run uvicorn memories.main:app --reload --host 0.0.0.0 --port 8000
```

All commands use `uv run`. Install deps with `uv sync`.

```bash
# Run frontend JS tests (Vitest)
npm test

# Run frontend tests with coverage (enforces 80% threshold on chat.js)
npm run test:coverage

# Frontend tests in watch mode
npm run test:watch

# Lint JS source and test files
npm run lint
```

Install JS dev deps once with `npm install` (creates `node_modules/`, not committed).

## Architecture

### What this project is

A locally-hosted character roleplay chatbot. The core problem it solves: LLMs invent and forget biographical details freely. This project grounds character behaviour in structured, user-defined memory so the character never contradicts itself.

### Memory model (four tiers)

| Tier | Description | Status |
|---|---|---|
| **Facts** | The schema-constrained JSON blob (`character_facts`, one blob per character). Values only ever set by the user, the World Builder, or the Character Evaluator (the latter subject to mutability). The model can never invent new categories — the schema fixes them. | active |
| **Inferences** | The character's beliefs, derived from Facts. Written immediately with no approval gate. Traceable via `source_fact_paths` / `source_inference_ids`. | active |
| **Experiences** | Append-only, immutable episodic log accumulated across sessions. Retrieved per turn by embedding similarity. | active |
| **Decisions** | Per-tool-call audit log across all three passes. Debugging only; not injected into context. | active |

### Three-pass design

Every turn runs three passes, all via `ollama.chat_with_tools` with the same model
(`character.current_model_name or character.modelfile_base`), always `think=False`. The server
drives each tool-call loop, capped at `MAX_TOOL_CALL_ROUNDS` (default 10).

1. **World Builder** (`world_builder.py`, `run_world_builder()`) — runs pre-turn on the user's
   message. Writes any implied facts to the `character_facts` blob via a single `author_set_facts`
   tool call. **Author authority**: no mutability check — it may set any path.
2. **Character LLM** (`chat_service.py`) — generates the in-character prose response (buffered
   server-side). Has the `require_fact` tool for Immutable-but-unset paths it needs a value for.
3. **Character Evaluator** (`evaluator.py`, `run_evaluator()`) — drives a tool-call loop
   (`set_fact`, `propose_inference`, `report_contradiction`, `report_pass`) and returns
   `(EvaluatorResult, updated_facts_blob)`. It checks the buffered response against Facts,
   Inferences, and active Experiences.

**Invariant**: the Character LLM is always followed by a Character Evaluator pass.
`run_contradiction_loop()` repeats the Character-LLM + Evaluator pair until the evaluator does
not report a contradiction, or `MAX_CONTRADICTION_RETRIES` (default 3, env-overridable) is
exhausted. A contradiction suppresses the response and triggers regeneration.

### Code layout

```
src/memories/
  main.py              # FastAPI app, lifespan (opens DB, warms up Ollama models, sets deps._db);
                       #   calls configure_logging() at import; global HTTPException handler logs 4xx/5xx
  deps.py              # get_db() and get_ollama() FastAPI dependencies; set_db() called by lifespan
  database.py          # All SQL — schema DDL + all repository functions; row_factory on every connection
  logging_config.py    # configure_logging(level) — one handler on the `memories` logger; LOG_LEVEL env
  models/__init__.py   # Pydantic models: Character, Session, Message, Decision, Inference, Experience
  exceptions.py        # NotFoundError, SessionEndedError
  routers/
    schema.py          # GET /api/schema (the fact schema rendered for the UI)
    characters.py      # GET/POST /api/characters
    facts.py           # GET /api/characters/{id}/facts       (the full fact blob)
                       #   GET /api/characters/{id}/inferences (list; ?status=active|stale|…)
                       #   PUT /api/characters/{id}/facts      (patch one leaf by dot-path + value)
    inferences.py      # POST /api/characters/{id}/inferences/generate  (eager pass)
                       #   POST .../inferences/revalidate               (cascade_on_fact_edit)
                       #   DELETE .../inferences/{id}
    sessions.py        # POST /api/sessions, POST /api/sessions/{id}/end, GET /api/sessions/{id}/messages
                       #   end calls session-end evaluator: writes closing_journal + returns Experience proposals
    chat.py            # POST /api/sessions/{id}/messages → text/event-stream (SSE)
    require_fact.py    # POST /api/sessions/{id}/turns/{turn_id}/require-fact/respond (resolve require_fact gate)
    fact_approval.py   # POST /api/sessions/{id}/turns/{turn_id}/set-fact/respond    (resolve set_fact gate)
    experiences.py     # POST /api/characters/{id}/experiences (approve proposal; embeds and writes to DB)
                       #   GET /api/characters/{id}/experiences
                       #   DELETE /api/characters/{id}/experiences/{id}
    decisions.py       # GET /api/sessions/{id}/decisions
  services/
    ollama_client.py   # Async httpx wrapper; chat_with_tools() (server-driven tool loop,
                       #   MAX_TOOL_CALL_ROUNDS cap); strips special tokens; warmup()
    sse_events.py      # SSEEvent, EventCallback types
    tool_gate.py       # per-turn asyncio.Queue gate keyed by (session_id, turn_id); create/await/resolve/cleanup
    schema_loader.py   # load_schema(), render_schema_for_prompt(), apply_mask(),
                       #   check_write_permitted(), iter_populated_leaves(), _collect_leaves()
    prompt_builder.py  # build_system_prompt(character, facts_blob, inferences, experiences) → str
    world_builder.py   # run_world_builder(): author_set_facts pre-turn pass
    evaluator.py       # run_evaluator() → (EvaluatorResult, updated_facts_blob)
    inference_service.py  # run_eager_pass(), revalidate_single_inference(),
                          #   cascade_on_fact_edit(), cascade_on_fact_delete(), compute_depth()
    experience_service.py # retrieve_experiences(), embed_and_store(),
                          #   run_session_end_evaluator() → closing journal + Experience proposals
                          #   clear/add/get/remove active experience sets (in-memory, keyed by session_id; replaced each turn)
    chat_service.py    # run_turn(): full per-turn orchestration
                       # run_contradiction_loop(): character + evaluator, retries until clean
  frontend/index.html          # Vue 3 CDN app (no build step); template + thin bootstrap; uses importmap for vue ESM
  frontend/chat-component.js  # Vue component setup() — all reactive state, methods, SSE handlers; tested
  frontend/chat.js             # Pure functions (SSE parsing, notification builder, API helpers); tested
```

### Key patterns

**DB dependency**: `deps.get_db()` yields the single module-level `_db` connection. The lifespan in `main.py` opens it, calls `init_db()`, then `deps.set_db(conn)`. Integration tests override `get_db` with a fixture yielding an in-memory connection.

**Schema is created in full at startup**: `init_db()` creates all seven tables (characters, sessions, character_facts, inferences, experiences, decisions, messages). No migrations. There is **no** `facts` table and **no** `segments` table.

**Facts, Inferences, and Experiences are loaded per turn**: `run_turn()` reloads the fact blob and Inferences from DB on every call. Active Experiences are retrieved by embedding the current user message and querying for all stored Experiences; those scoring at or above `MIN_EXPERIENCE_SCORE` (up to `TOP_K_EXPERIENCES`) are the active set for that turn. The active set is replaced on every turn — nothing carries over from previous turns.

**Inference depth cap**: `MAX_INFERENCE_DEPTH=5` (env-overridable). `compute_depth()` in `inference_service.py` resolves depth from source inference ids at write time. Inferences exceeding the cap are silently discarded.

**Cascade on Fact edit/delete (user-initiated)**: `cascade_on_fact_edit(changed_path)` BFS-walks downstream inferences, calling `revalidate_single_inference()` (an LLM call) for each active one; ones that no longer hold are marked `stale`, and already-stale inferences propagate the cascade without an LLM call. `cascade_on_fact_delete(deleted_path)` is pure DB: marks all transitively-dependent inferences `invalidated`. These are keyed by fact **path** and fire only from user-initiated inference endpoints — the World Builder does **not** trigger them.

**Tool-gate suspension**: `tool_gate.py` holds a per-turn `asyncio.Queue` keyed by
`(session_id, turn_id)`, created at turn start. A blocking tool handler (the Character LLM's
`require_fact`, or the Character Evaluator's mutable / immutable-unset `set_fact`) emits a
`sidechannel` event and then `await`s the gate, suspending the tool-call loop. The
`require_fact` / `fact_approval` endpoints call `resolve_gate()` to deliver the user's response
and let the loop resume.

**SSE event sequence** from the chat endpoint: `status(extracting)` → `status(generating)` → `status(reviewing)` → *(if contradictions occurred)* `sidechannel(contradiction)` + `status(regenerating)` + `status(reviewing)` per retry → *(if think=true)* `thinking` → `message` → `done`. Blocking approval flows emit `sidechannel` events mid-stream and suspend until the matching respond endpoint is called. Sidechannel types: `world_builder_applied`, `fact_update_fluid`, `fact_update_mutable`, `fact_update_immutable_unset`, `require_fact`, `inference_proposed`, `contradiction`. The frontend uses `fetch` + `ReadableStream` rather than `EventSource` (which does not support POST bodies).

**Ollama special-token stripping**: `_SPECIAL_TOKEN_RE` strips chat-template control tokens (e.g., `<|endoftext|>`) that some models emit past their natural stop point.

**Model warmup**: at lifespan start, `_warmup_models()` sends `POST /api/generate` with `keep_alive: 10m` for every model in the DB. Connection/response errors are logged as warnings and do not block startup.

**Logging**: `configure_logging()` (in `logging_config.py`) is called at import time in `main.py`. It attaches a single `StreamHandler` to the `memories` logger — every module uses `logging.getLogger(__name__)`, so all logs fall under `memories.*`. `propagate` stays `True` so pytest's `caplog` keeps working. Verbosity is controlled by the `LOG_LEVEL` env var (default `INFO`). Level conventions:

| Level | Use for |
|---|---|
| `DEBUG` | Full detail for deep debugging: LLM prompts, per-round tool-call args/results, tool-gate create/resolve/cleanup. |
| `INFO` | The normal turn trace — turn start; each fact written; each inference proposed; evaluator verdict; approval card surfaced and its resolution; experiences retrieved; model warmup. |
| `WARNING` | Recoverable degradation — the turn continues but something did not go cleanly (tool-call cap reached, evaluator parse error, World Builder failure, enum/int coercion fallback, contradiction retries exhausted, HTTP 4xx). |
| `ERROR` / `exception` | An unhandled exception that broke a turn (e.g. one escaping the SSE stream generator). |

At `INFO` a reader can reconstruct what each pass decided and what the user was asked to approve, without the prompts; `DEBUG` adds the prompts and raw tool payloads.

**`_set_leaf` type convention**: helper functions that write a value into the blob use `str | int | float | bool | None` for the `value` parameter — never `Any`. Using `Any` triggers ruff `ANN401`. The same convention applies to any function that reads a leaf value out of the blob (`-> str | int | float | bool | None`). See `world_builder.py` and `evaluator.py` for the established pattern.

**`pass_name` bandit false positive**: `store_decision(..., pass_name="character_evaluator", ...)` triggers bandit `B106` ("hardcoded password funcarg"). Add `# nosec B106` on the `pass_name=` line to suppress it. Every call site in the codebase already follows this pattern.

### Test layout

```
tests/
  conftest.py              # db fixture (in-memory aiosqlite) + root client fixture
  unit/
    conftest.py            # character/session/ollama fixtures; make_ollama_ndjson() + make_evaluator_ndjson()
    test_prompt_builder.py
    test_ollama_client.py
    test_chat_service.py
    test_evaluator_service.py
    test_inference_service.py
    test_experience_service.py
    test_logging_config.py
    test_health.py
  integration/
    conftest.py            # overrides get_db and get_ollama dependencies; character/session fixtures
    test_db_init.py
    test_*_repo.py         # one file per DB repository (including test_experiences_repo.py)
    test_api_*.py          # one file per router (including test_api_inference_generation.py, test_api_experiences.py, test_api_chat.py)
  frontend/
    chat.test.js           # Vitest tests for chat.js pure logic (SSE parsing, API helpers, notification building)
    chat-component.test.js # Vitest tests for chat-component.js reactive state (setup() called directly)
```

**Python tests**: Integration tests override both `get_db` (in-memory aiosqlite) and `get_ollama` (client pointing to `http://test-ollama-integration:11434`). Ollama HTTP calls are mocked with `respx`. Use `make_ollama_ndjson()` for character responses and `make_evaluator_ndjson()` for evaluator responses. Coverage threshold is 80% overall; `frontend/` is excluded.

**Frontend tests**: Vitest + jsdom, targeting `tests/frontend/**/*.test.js`. `chat.test.js` covers SSE block parsing, status label mapping, notification object construction, and API call shape/URL. `chat-component.test.js` covers reactive state management via `ChatComponent.setup()` called directly (no mount).

**Rule:** any new logic added to `chat.js` must have corresponding tests in `tests/frontend/`. When adding new SSE event types, notification types, or API endpoints, update both `chat.js` and `chat.test.js` in the same commit.

**Rule:** any new SSE sidechannel type requires four things in the same commit:
1. A case in `buildNotificationFromSidechannel` in `chat.js` — with tests in `chat.test.js`
2. A `v-else-if="msg.scType === '...'"` notification card in `index.html`
3. A handler in the `sendMessage` SSE loop in `chat-component.js`
4. A test in `chat-component.test.js` covering the handler behaviour

### Configurable limits (environment variables)

| Variable | Default | Effect |
|---|---|---|
| `MAX_CONTRADICTION_RETRIES` | `3` | Max times the contradiction loop retries before giving up |
| `MAX_TOOL_CALL_ROUNDS` | `10` | Max server-driven tool-call rounds per `chat_with_tools` pass |
| `MAX_INFERENCE_DEPTH` | `5` | Max hops from root Facts in an inference chain |
| `MAX_INFERENCE_BREADTH` | `5` | Max inferences generated per eager pass |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MEMORIES_DB_PATH` | `memories.db` | SQLite database file path |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for Experience embedding and retrieval |
| `TOP_K_EXPERIENCES` | `5` | Experiences retrieved per turn via similarity search |
| `MIN_EXPERIENCE_SCORE` | `0.0` | Minimum cosine similarity score for an experience to be injected into context |
| `LOG_LEVEL` | `INFO` | Verbosity of the `memories` logger (`DEBUG` surfaces LLM prompts and tool payloads) |

### What's deferred

- **Optimistic streaming** (`docs/streaming-plan.md`): the response is currently buffered server-side until the evaluator clears it; token-by-token streaming is not yet implemented.
- **Phase 7a/7b (Context budget & compression)**: token counting, `captured_by` annotation, and compression passes are not yet implemented. Revisiting this will **rebuild** the dropped `segments` table.
