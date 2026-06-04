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

## Architecture

### What this project is

A locally-hosted character roleplay chatbot. The core problem it solves: LLMs invent and forget biographical details freely. This project grounds character behaviour in structured, user-defined memory so the character never contradicts itself.

### Memory model (four tiers)

| Tier | Description | Status |
|---|---|---|
| **Facts** | Ground truths set by the user. Never invented by the model. | Phase 1 — active |
| **Inferences** | Logical/probabilistic conclusions derived from Facts. Traceable derivation graph. | Phase 3 |
| **Experiences** | Episodic memory accumulated across sessions. Retrieved by embedding similarity. | Phase 4 |
| **Decisions** | Evaluator audit log. Debugging only; not injected into context. | Phase 2 |

### Two-LLM design

Every turn involves two sequential Ollama calls:
1. **Character LLM** — generates a response in-character (stream buffered server-side)
2. **Evaluator LLM** — checks the buffered response against Facts/Inferences/Experiences; returns a structured JSON verdict

The evaluator is not yet wired in (Phase 2). Phase 1 buffers the Ollama response and forwards it directly.

Six evaluator verdicts: `pass`, `new_inference_logical`, `new_inference_probabilistic`, `implication`, `contradiction`, `experience_update`. Contradictions suppress the response and trigger automatic regeneration; other verdicts deliver with or without a sidechannel notification. See `docs/plan.md` for the full flow.

### Code layout

```
src/memories/
  main.py              # FastAPI app, lifespan (opens DB, sets deps._db), mounts routers + static files
  deps.py              # get_db() and get_ollama() FastAPI dependencies; set_db() called by lifespan
  database.py          # All SQL — schema DDL + repository functions; row_factory set on every connection
  models/__init__.py   # Pydantic models: Character, Fact, Session, Segment, Message
  exceptions.py        # NotFoundError, SessionEndedError
  routers/
    characters.py      # GET/POST /api/characters
    facts.py           # GET/POST/PUT/DELETE /api/characters/{id}/facts
    sessions.py        # POST /api/sessions, POST /api/sessions/{id}/end, GET /api/sessions/{id}/messages
    chat.py            # POST /api/sessions/{id}/messages → text/event-stream (SSE)
  services/
    ollama_client.py   # Async httpx wrapper for POST /api/chat; stream:true, think:false
    prompt_builder.py  # build_system_prompt(character, facts) → str
    chat_service.py    # run_turn(): load → build prompt → store user msg → call Ollama → store reply
  frontend/index.html  # Vue 3 CDN app (no build step); two-panel chat + facts sidechannel
```

### Key patterns

**DB dependency**: `deps.get_db()` is an async generator that yields the single module-level `_db` connection. The lifespan in `main.py` opens the connection, calls `init_db()`, and calls `deps.set_db(conn)`. Integration tests override `get_db` with a fixture that yields an in-memory connection.

**Schema is created in full at Phase 1**: `init_db()` creates all eight tables (including `inferences`, `experiences`, `decisions`). Tables unused in later phases sit empty. No migrations needed between phases.

**Every session starts with a segment**: `create_session()` also inserts a `segments` row with `boundary_reason="session_start"`. All Phase 1 messages go into this segment. Phase 5b will add boundary logic on top.

**Facts are loaded per turn**: `run_turn()` reloads facts from DB on every call so mid-conversation fact edits take effect on the next message without a session restart.

**SSE event sequence** from the chat endpoint: `status` (generating) → `message` (full assistant content + turn_id) → `done`. The frontend uses `fetch` + `ReadableStream` rather than `EventSource` because `EventSource` does not support POST bodies.

**Ollama options**: both character and evaluator calls set `"options": {"think": false}`. This is intentional — the harness provides the structured reasoning context, so thinking mode adds latency without benefit.

### Test layout

```
tests/
  conftest.py              # db fixture (in-memory aiosqlite) + root client fixture
  unit/
    conftest.py            # character/session/fact/ollama fixtures; make_ollama_ndjson() helper
    test_prompt_builder.py
    test_ollama_client.py
    test_chat_service.py
  integration/
    conftest.py            # overrides get_db and get_ollama dependencies; character/session/fact fixtures
    test_db_init.py
    test_*_repo.py         # one file per DB repository
    test_api_*.py          # one file per router
```

Integration tests override both `get_db` (with the `db` fixture connection) and `get_ollama` (with a client pointing to `http://test-ollama-integration:11434`). Ollama HTTP calls are mocked with `respx`. No real Ollama instance required to run tests.

Coverage threshold is 80% overall (enforced by `--cov-fail-under=80`). The `frontend/` directory is excluded from coverage.

### What's deferred

The database schema and `OllamaClient` are already structured to support future phases. The evaluator call slots in between "buffer complete" and "deliver to client" in `chat_service.run_turn()`. Inferences, Experiences, and Decisions tables exist but are empty. Segment boundary logic (Phase 5b) will extend `get_active_segment()`.
