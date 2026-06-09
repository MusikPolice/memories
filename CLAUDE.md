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
| **Facts** | Ground truths set by the user. Never invented by the model. | Phase 1 — active |
| **Inferences** | Logical/probabilistic conclusions derived from Facts. Traceable derivation graph. | Phase 3 — active |
| **Experiences** | Episodic memory accumulated across sessions. Retrieved by embedding similarity. | Phase 5 — active |
| **Decisions** | Evaluator audit log. Debugging only; not injected into context. | Phase 2 — active |

### Two-LLM design

Every turn involves two sequential Ollama calls:
1. **Character LLM** — generates an in-character response (buffered server-side)
2. **Evaluator LLM** — checks the buffered response against Facts, Inferences, and active Experiences; returns a structured JSON verdict

Both calls use the same model (`character.current_model_name or character.modelfile_base`), always with `think=False`. The contradiction loop repeats both calls until the evaluator returns a non-contradiction verdict or `MAX_CONTRADICTION_RETRIES` (default 3, env-overridable) is exhausted.

Six evaluator verdicts: `pass`, `new_inference_logical`, `new_inference_probabilistic`, `implication`, `contradiction`, `experience_update`. Contradictions suppress the response and trigger automatic regeneration. See `docs/plan.md` for the full flow.

### Code layout

```
src/memories/
  main.py              # FastAPI app, lifespan (opens DB, warms up Ollama models, sets deps._db)
  deps.py              # get_db() and get_ollama() FastAPI dependencies; set_db() called by lifespan
  database.py          # All SQL — schema DDL + all repository functions; row_factory on every connection
  models/__init__.py   # Pydantic models: Character, Fact, Session, Segment, Message, Decision, Inference, Experience
  exceptions.py        # NotFoundError, SessionEndedError
  routers/
    characters.py      # GET/POST /api/characters
    facts.py           # GET/POST/PUT/DELETE /api/characters/{id}/facts
                       #   category ("user"|"character"|"setting") and mutability ("immutable"|"low"|"high") fields on all writes
                       #   PUT triggers cascade_on_fact_edit; DELETE triggers cascade_on_fact_delete
    inferences.py      # POST /api/characters/{id}/inferences/generate (eager pass)
                       #   POST .../revalidate, DELETE/PATCH .../inferences/{id}
                       #   POST .../inferences/{id}/promote (creates Fact from Inference, deletes source Inference)
    sessions.py        # POST /api/sessions, POST /api/sessions/{id}/end, GET /api/sessions/{id}/messages
                       #   end calls session-end evaluator: writes closing_journal + returns Experience proposals
    chat.py            # POST /api/sessions/{id}/messages → text/event-stream (SSE)
    implication.py     # POST .../turns/{turn_id}/accept-implication  (creates Fact, optionally regenerates)
                       #   POST .../turns/{turn_id}/ignore-implication (no-op; client dismisses)
                       #   POST .../turns/{turn_id}/accept-inference   (stores probabilistic inference)
                       #   POST .../turns/{turn_id}/ignore-inference   (no-op)
    experiences.py     # POST /api/characters/{id}/experiences (approve proposal; embeds and writes to DB)
                       #   GET /api/characters/{id}/experiences
                       #   DELETE /api/characters/{id}/experiences/{id}
    decisions.py       # GET /api/sessions/{id}/decisions
  services/
    ollama_client.py   # Async httpx wrapper; stream:true buffered; strips special tokens; warmup()
    prompt_builder.py  # build_system_prompt(character, facts, inferences, experiences) → str
    evaluator.py       # build_evaluator_prompt(), run_evaluator() → EvaluatorResult
    inference_service.py  # run_eager_pass(), revalidate_single_inference(),
                          #   cascade_on_fact_edit(), cascade_on_fact_delete(), compute_depth()
    experience_service.py # retrieve_experiences(), cold_start_retrieve(), embed_and_store()
                          #   run_session_end_evaluator() → closing journal + Experience proposals
                          #   get/add/remove active experience sets (in-memory, keyed by session_id)
    chat_service.py    # run_turn(): full per-turn orchestration
                       # run_contradiction_loop(): character + evaluator, retries until clean
  frontend/index.html          # Vue 3 CDN app (no build step); template + thin bootstrap; uses importmap for vue ESM
  frontend/chat-component.js  # Vue component setup() — all reactive state, methods, SSE handlers; tested
  frontend/chat.js             # Pure functions (SSE parsing, notification builder, API helpers); tested
```

### Key patterns

**DB dependency**: `deps.get_db()` yields the single module-level `_db` connection. The lifespan in `main.py` opens it, calls `init_db()`, then `deps.set_db(conn)`. Integration tests override `get_db` with a fixture yielding an in-memory connection.

**Schema is created in full at startup**: `init_db()` creates all eight tables (characters, sessions, facts, inferences, experiences, decisions, segments, messages). No migrations.

**Every session starts with a segment**: `create_session()` also inserts a `segments` row with `boundary_reason="session_start"`. All messages link to their segment via `segment_id`.

**Facts, Inferences, and Experiences are loaded per turn**: `run_turn()` reloads Facts and Inferences from DB on every call. Active Experiences are retrieved by embedding the current user message and querying for the top-k most similar stored Experiences; newly retrieved ones are added to the session's in-memory active set (managed in `experience_service.py`) and stay in context for the remainder of the session.

**Inference depth cap**: `MAX_INFERENCE_DEPTH=5` (env-overridable). `compute_depth()` in `inference_service.py` resolves depth from source inference ids at write time. Inferences exceeding the cap are silently discarded. The same cap applies to lazy discovery in `run_turn()`.

**Cascade on Fact edit**: `cascade_on_fact_edit()` BFS-walks downstream inferences, calling `revalidate_single_inference()` (an LLM call) for each active one. Ones that no longer hold are marked `stale`. Already-stale inferences propagate the cascade without an LLM call. `cascade_on_fact_delete()` is pure DB: marks all transitively-dependent inferences `invalidated`.

**SSE event sequence** from the chat endpoint: `status(generating)` → `status(reviewing)` → *(if contradictions occurred)* `sidechannel(contradiction)` + `status(regenerating)` + `status(reviewing)` per retry → *(if think=true)* `thinking` → `message` → *(if implication/probabilistic)* `sidechannel` → *(if experience_update)* `sidechannel(experience_update)` → `done`. The frontend uses `fetch` + `ReadableStream` rather than `EventSource` because `EventSource` does not support POST bodies.

**Implication acceptance** (`POST .../accept-implication`): creates the Fact, then by default regenerates the assistant response through the full contradiction loop. When `regenerate=false` (user accepted the character's exact value), no regeneration is needed — the existing response is already correct.

**Ollama special-token stripping**: `_SPECIAL_TOKEN_RE` strips chat-template control tokens (e.g., `<|endoftext|>`) that some models emit past their natural stop point.

**Model warmup**: at lifespan start, `_warmup_models()` sends `POST /api/generate` with `keep_alive: 10m` for every model in the DB. Connection/response errors are logged as warnings and do not block startup.

### Test layout

```
tests/
  conftest.py              # db fixture (in-memory aiosqlite) + root client fixture
  unit/
    conftest.py            # character/session/fact/ollama fixtures; make_ollama_ndjson() + make_evaluator_ndjson()
    test_prompt_builder.py
    test_ollama_client.py
    test_chat_service.py
    test_evaluator_service.py
    test_inference_service.py
    test_experience_service.py
    test_health.py
  integration/
    conftest.py            # overrides get_db and get_ollama dependencies; character/session/fact fixtures
    test_db_init.py
    test_*_repo.py         # one file per DB repository (including test_experiences_repo.py)
    test_api_*.py          # one file per router (including test_api_inference_generation.py, test_api_inference_promotion.py, test_api_implication.py, test_api_experiences.py)
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
| `MAX_INFERENCE_DEPTH` | `5` | Max hops from root Facts in an inference chain |
| `MAX_INFERENCE_BREADTH` | `5` | Max inferences generated per eager pass |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `MEMORIES_DB_PATH` | `memories.db` | SQLite database file path |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama model used for Experience embedding and retrieval |
| `TOP_K_EXPERIENCES` | `5` | Experiences retrieved per turn via similarity search |
| `MIN_EXPERIENCE_SCORE` | `0.0` | Minimum cosine similarity score for an experience to be injected into context |

### What's deferred

- **Phase 7a/7b (Context budget & compression)**: `segments` table exists; all messages go into a single `session_start` segment. Token counting, `captured_by` annotation, and compression passes are not yet implemented.
