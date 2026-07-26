# Step 10 — Documentation Update & Logging Pass

## Overview

Step 10 is the final step of the Facts v2 refactor. The plan (`plan-v2.md`) scopes it as
documentation-only: rewrite the architecture section of `CLAUDE.md` and the `README.md` to
describe the delivered three-pass, tool-calling, schema-constrained system rather than the
old two-LLM / flat-fact design the docs still describe. Steps 1–9 changed the code out from
under both documents; neither has been touched since (Step 9 explicitly deferred all doc work
to here). Large parts of both files are now actively wrong — they reference a `Fact` model, a
`segments` table, an `implication.py` router, six evaluator verdicts, and a JSON-verdict
evaluator, none of which still exist.

This step adds a second, code-bearing concern at the user's request: **a pass over
application logging.** The motivation is that manual testing begins immediately after Step 10,
and good logging is the difference between reading a stack trace and guessing. The codebase
already has scattered `logging` calls, but there is **no central logging configuration
anywhere** — no `basicConfig`, no `dictConfig`, no handler setup. The practical consequence is
that every `_log.info(...)` and `log.debug(...)` in the codebase is silently dropped: with no
configuration, the `memories.*` loggers propagate to the root logger, whose default
"handler of last resort" emits only `WARNING` and above. So today the informational turn-trace
logging that already exists in `chat_service.py` never appears, and there is no way to raise
verbosity for a debugging session. This step adds a single logging-setup module, wires it in
at startup, makes verbosity configurable via a new `LOG_LEVEL` environment variable, and fills
the most valuable gaps in the turn trace (fact writes, evaluator verdicts, approval-gate
lifecycle, and unhandled SSE-stream exceptions).

The transitional state this leaves is the final state: after Step 10 the Facts v2 refactor is
complete, documented, and observable. There is no Step 11.

**Success criterion:** `tests/unit/test_logging_config.py` passes (new file); the full suite
(`uv run pytest`), `uv run ruff check src/`, `uv run mypy src/`, and `npm run test:coverage`
stay green; and `CLAUDE.md` / `README.md` contain no reference to the removed `Fact` model,
`segments` table, `implication` verdict, `experience_update` verdict, or JSON-verdict
evaluator.

---

## What Steps 8 and 9 Delivered

The system this step documents and instruments:

- **Three passes per turn**, all in `src/memories/services/`:
  - `world_builder.py` — `run_world_builder()` runs pre-turn on the user's message; writes any
    implied facts to the `character_facts` blob via a single `author_set_facts` tool call
    (author authority: no mutability check).
  - `chat_service.py` — `run_contradiction_loop()` invokes the Character LLM
    (`ollama.chat_with_tools` with the `require_fact` tool) then the Character Evaluator, and
    retries on contradiction; `run_turn()` orchestrates the whole turn.
  - `evaluator.py` — `run_evaluator()` drives the Character Evaluator's tool-call loop
    (`set_fact`, `propose_inference`, `report_contradiction`, `report_pass`) and returns
    `(EvaluatorResult, updated_facts_blob)`.
- **Tool-calling substrate**: `ollama_client.py` `chat_with_tools()` (server-driven tool loop,
  `MAX_TOOL_CALL_ROUNDS` cap), `tool_gate.py` (per-turn `asyncio.Queue` gate for blocking
  approval flows), `sse_events.py` (`SSEEvent`, `EventCallback`).
- **Schema-constrained fact blob**: `fact_schema.json` + `schema_loader.py`
  (`load_schema()`, `render_schema_for_prompt()`, `apply_mask()`, `check_write_permitted()`,
  `iter_populated_leaves()`, `_collect_leaves()`); DB `character_facts` table (one JSON blob
  per character); repo helpers `get_facts()`, `set_facts()`, `patch_fact()`.
- **DB schema — seven tables** (`database.py` `init_db()`): `characters`, `sessions`,
  `character_facts`, `inferences`, `experiences`, `decisions`, `messages`. **No** `facts`
  table, **no** `segments` table.
- **`decisions` table is a per-tool-call log**: columns `pass_name`, `tool_name`,
  `tool_args` (JSON), `user_input` (JSON, nullable). `store_decision()` writes one row per tool
  invocation across all three passes.
- **Models** (`models/__init__.py`): `Character`, `Session`, `Message`, `Decision`,
  `Inference` (`source_inference_ids`, `source_fact_paths` — no `source_fact_ids`),
  `Experience`. **No** `Fact`, **no** `Segment`.
- **Routers** (`main.py` mounts): `schema` (`GET /api/schema`), `characters`, `facts`
  (blob path-based GET/PUT/DELETE), `sessions`, `chat` (SSE), `decisions`, `inferences`
  (generate / revalidate / delete — no promote, no PATCH-status), `experiences`,
  `require_fact` (`POST .../turns/{turn_id}/require-fact/respond`), `fact_approval`
  (`POST .../turns/{turn_id}/set-fact/respond`). **No** `implication` router.
- **Existing logging** (all via `logging.getLogger(__name__)`, i.e. under the `memories.*`
  namespace): `main.py` (warmup + a global `HTTPException` handler that logs 4xx/5xx at
  `WARNING`), `chat_service.py` (turn verdict at `INFO`, several `WARNING`s),
  `evaluator.py`/`world_builder.py`/`experience_service.py`/`sessions.py` (`WARNING`s),
  `ollama_client.py` (`DEBUG` per-round tool-call trace). None of it is currently configured to
  emit below `WARNING`.

---

## What This Step Does NOT Change

- **No behavioural code changes.** The only executable code added is the logging-setup module,
  its one-line call in `main.py`, and additional log statements. No control flow, no tool
  handler, no prompt, no DB write changes. A log statement must never alter what the
  surrounding code does (no new exceptions, no changed return values).
- **`fact_schema.json`, `schema_loader.py`, `database.py` DDL, all models.** Untouched.
- **The frontend** (`index.html`, `chat.js`, `chat-component.js`). No SSE event types,
  notification cards, or API endpoints are added, so the four-part sidechannel commit rule is
  not triggered. `npm run test:coverage` must stay green but no JS changes are made.
- **`docs/plan-v2.md` and prior `docs/stepN.md`.** Historical record; left as-is.
- **Existing log statements' wording**, except where a level is explicitly changed below. Do
  not churn messages that already work.
- **The `ollama_client.py` `DEBUG` trace.** It is already the right level and content; it
  simply becomes visible when `LOG_LEVEL=DEBUG`. No change.

---

## Detailed Design

Ordered lowest-level first: the logging module and its wiring, then the per-file log
additions, then the two documentation rewrites.

### Part A — `src/memories/logging_config.py` (new file)

A single module owning all logging setup. One public function:

```python
from __future__ import annotations

import logging
import os

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATEFMT = "%H:%M:%S"


def configure_logging(level: str | None = None) -> None:
    """Configure the ``memories`` package logger.

    Level resolution: the explicit ``level`` argument if given, else the ``LOG_LEVEL``
    environment variable, else ``INFO``. An unrecognised level name falls back to ``INFO``.
    Idempotent: a second call updates the level but does not attach a duplicate handler.
    """
    resolved = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric = logging.getLevelNamesMapping().get(resolved, logging.INFO)

    logger = logging.getLogger("memories")
    logger.setLevel(numeric)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(handler)
```

Design decisions, each load-bearing:

- **Configure the `memories` logger, not root.** Every module already calls
  `logging.getLogger(__name__)`, which yields names under `memories.*` (e.g.
  `memories.services.chat_service`). Attaching one handler to the parent `memories` logger
  catches all of them without touching root or fighting uvicorn's own `uvicorn.*` handler
  configuration.
- **`propagate` is left at its default (`True`).** This is deliberate so pytest's `caplog`
  fixture — which captures via the root logger — continues to work for any behavioural test
  that wants to assert on a log line. Under `uvicorn` the root logger has no handler (uvicorn
  configures only its named loggers), so no duplicate emission occurs in production. See Edge
  Cases.
- **Idempotency via `if not logger.handlers`.** Guards against duplicate handlers if
  `configure_logging()` is called more than once in a single process (e.g. imported by a test
  and then by app startup). `uvicorn --reload` spawns a fresh process per reload, so handlers
  never stack across reloads regardless.
- **`getLevelNamesMapping()`** (Python 3.11+; the project targets 3.12) returns
  `{"DEBUG": 10, "INFO": 20, ...}`; `.get(resolved, logging.INFO)` gives clean
  invalid-name-falls-back-to-INFO behaviour with no exception path.
- **Short `%H:%M:%S` timestamp** keeps each line readable in a local terminal; the date is
  rarely relevant for an interactive local session.

### Part B — `src/memories/main.py`: call `configure_logging()` at import

Add the import and invoke it once at module top level, immediately after the existing
`_log = logging.getLogger(__name__)` line ([main.py:32](src/memories/main.py#L32)) and before
`app = FastAPI(...)`:

```python
from memories.logging_config import configure_logging

_log = logging.getLogger(__name__)
configure_logging()
```

Module top level (not inside `lifespan`) is required so the configuration is active when
`uvicorn memories.main:app` imports the module — before the lifespan warmup logs fire, and
regardless of whether lifespan runs at all. Add one startup `INFO` line inside `lifespan`
(after `db_path` is resolved) so a fresh run announces its configuration:

```python
_log.info("starting Memories — db=%s ollama=%s", db_path, ollama.base_url)
```

### Part C — Logging level conventions

Apply these consistently for every new log statement, and document them verbatim in
`CLAUDE.md` (Part E):

| Level | Use for | Examples |
|---|---|---|
| `DEBUG` | Full detail for deep debugging: LLM prompts, per-round tool-call args and results. | `ollama_client.chat_with_tools` round trace (already present); `tool_gate` create/resolve. |
| `INFO` | The normal turn trace — enough to follow the shape of a turn end to end. | turn start; each fact written; each inference proposed; evaluator verdict; approval card surfaced and its resolution; experiences retrieved; model warmup. |
| `WARNING` | Recoverable degradation — the turn continues but something did not go cleanly. | tool-call cap reached; evaluator parse error; World Builder LLM failure; enum/int coercion fallback; contradiction retries exhausted; HTTP 4xx (existing global handler). |
| `ERROR` / `exception` | An unhandled exception that broke a turn. | uncaught exception escaping the SSE stream generator. |

Guiding rule for the audit: **at `INFO` a reader should be able to reconstruct what each pass
decided and what the user was asked to approve, without the prompts.** `DEBUG` adds the
prompts and raw tool payloads.

### Part D — Log-statement audit (per file)

Each addition below uses the module's existing `_log`/`log` object. Keep messages
single-line, use `%`-style lazy args (never f-strings in the log call — ruff/convention), and
truncate free text (user messages, statements) to a bounded prefix.

#### D1. `chat_service.py` — turn lifecycle

- **`run_turn()`**, right after `turn_id` is resolved and before `create_gate(...)`
  ([chat_service.py:350](src/memories/services/chat_service.py#L350)): add a turn-start `INFO`:
  ```python
  _log.info(
      "turn start session=%d turn=%d think=%s msg=%r",
      session_id, turn_id, think, user_content[:120],
  )
  ```
- The existing end-of-turn `INFO` verdict line
  ([chat_service.py:437](src/memories/services/chat_service.py#L437)) and the
  `max_retries_exceeded` `WARNING` stay as-is.
- **`_handle_require_fact`** ([chat_service.py:140](src/memories/services/chat_service.py#L140)):
  add an `INFO` immediately before `await await_gate(...)`
  ([chat_service.py:178](src/memories/services/chat_service.py#L178)) —
  `"require_fact awaiting user input session=%d turn=%d path=%s", session_id, turn_id, path`
  — and an `INFO` immediately after the gate returns reporting the resolution
  (`path`, and whether a value was provided). These bracket the SSE suspension so a hung turn
  is visible in the log.

#### D2. `world_builder.py` — fact writes and per-entry errors

Inside `_handle_author_set_facts` ([world_builder.py:164](src/memories/services/world_builder.py#L164)),
after the `for entry in entries:` loop completes and `written`/`errors` are known:

- If `written`: `INFO` — `"world_builder session=%d turn=%d wrote %d fact(s): %s"` with a compact
  `path=value` join.
- If `errors`: `WARNING` — `"world_builder session=%d turn=%d rejected %d entr(y/ies): %s"`.
  These per-entry failures (unknown path, invalid enum, non-integer) are currently returned
  only to the LLM as a tool result and are invisible to the operator; surfacing them explains
  why an expected fact did not stick. Place after the `db_set_facts` write so a logged "wrote"
  reflects a committed write.

#### D3. `evaluator.py` — per-tool decisions and approval-gate lifecycle

The Character Evaluator is the pass most worth tracing. Add `INFO` at each decision point;
keep the existing `WARNING`s (coercion fallbacks, cap-with-no-terminal) unchanged.

- **Fluid `set_fact`** (after the write, near
  [evaluator.py:348](src/memories/services/evaluator.py#L348)):
  `INFO` — `"evaluator set_fact (fluid) turn=%d %s=%r"`.
- **Mutable / immutable-unset `set_fact`**: add an `INFO` immediately before each
  `await await_gate(...)` ([evaluator.py:390](src/memories/services/evaluator.py#L390) and
  [evaluator.py:480](src/memories/services/evaluator.py#L480)) announcing the card surfaced
  (`turn`, `path`, tier), and an `INFO` after the gate returns reporting `action` and, for
  edits, the stored value. This mirrors D1's bracketing for the two evaluator-side blocking
  flows.
- **`_handle_propose_inference`** (after `create_inference`,
  [evaluator.py:537](src/memories/services/evaluator.py#L537)): `INFO` —
  `"evaluator propose_inference turn=%d id=%d stmt=%r"` with the statement truncated.
- **`_handle_report_contradiction`** ([evaluator.py:574](src/memories/services/evaluator.py#L574)):
  `INFO` — `"evaluator report_contradiction turn=%d: %s"` (the per-attempt contradiction is
  already logged in `chat_service`, but logging it at the source ties it to the evaluator pass).
- `_handle_report_pass` needs no log (the turn-level verdict line already covers the clean case).

#### D4. `routers/chat.py` — SSE stream lifecycle and uncaught exceptions

This is the highest-value single addition for manual debugging. Today, if `run_turn` raises,
the exception is re-raised inside the `_stream` generator
([chat.py:64-66](src/memories/routers/chat.py#L64-L66)) and surfaces to the client only as a
truncated stream, with nothing in the log. Add a module logger and wrap the failure path:

```python
import logging
_log = logging.getLogger(__name__)
```

- At the top of `_stream` (or right after the task is created): `INFO` —
  `"chat stream open session=%d"`.
- Where the task exception is detected
  ([chat.py:64](src/memories/routers/chat.py#L64)), before re-raising: `_log.exception(
  "run_turn failed session=%d", session_id)` so the full traceback is captured. Still re-raise
  — behaviour is unchanged; only visibility is added.

#### D5. `routers/require_fact.py` and `routers/fact_approval.py` — user decisions

Both routers currently have no logger. Add `_log = logging.getLogger(__name__)` and one `INFO`
per successful `resolve_gate` recording the user's decision:

- `require_fact.py`: `"require-fact resolved session=%d turn=%d provided=%s", ..., body.value is not None`.
- `fact_approval.py`: `"set-fact resolved session=%d turn=%d action=%s", ..., body.action`.

The `404`/`409` error branches are already logged by the global `HTTPException` handler in
`main.py`; do not double-log them.

#### D6. `tool_gate.py` — gate lifecycle (DEBUG)

Optional but cheap and useful when a suspension misbehaves. Add
`_log = logging.getLogger(__name__)` and `DEBUG` lines in `create_gate`, `resolve_gate`, and
`cleanup_gate` recording the `(session_id, turn_id)` key. Keep them `DEBUG` — they are noise at
`INFO`. Do not log inside `await_gate` (it would fire on every blocking call and duplicate D1/D3).

### Part E — Rewrite the `## Architecture` section of `CLAUDE.md`

Replace lines [55](CLAUDE.md#L55)–[201](CLAUDE.md#L201) (the entire `## Architecture` section,
from `### What this project is` through `### What's deferred`). The `## Commands` section above
it is still accurate and stays, except add the two logging/dev notes below. Target content:

1. **`### What this project is`** — keep; still accurate.
2. **`### Memory model (four tiers)`** — keep the four-tier table but correct the descriptions:
   Facts are now the **schema-constrained JSON blob** (`character_facts`), values only ever set
   by the user, the World Builder, or the Character Evaluator subject to mutability; Inferences
   are the character's beliefs, written immediately with no approval; Experiences are an
   **append-only immutable** episodic log; Decisions are a **per-tool-call** audit log. Drop the
   stale "Phase N" column or relabel it "Status: active".
3. **Replace `### Two-LLM design` with `### Three-pass design`** — describe, in order, the World
   Builder (pre-turn, author authority, `author_set_facts`), the Character LLM (generates prose,
   `require_fact` tool), and the Character Evaluator (tool-call loop:
   `set_fact` / `propose_inference` / `report_contradiction` / `report_pass`). State that all
   three use `ollama.chat_with_tools`, the same model, `think=False`, and that the server drives
   the tool loop with a `MAX_TOOL_CALL_ROUNDS` cap. State the invariant that the Character LLM
   is always followed by a Character Evaluator pass. **Remove** the six-verdict list entirely.
4. **`### Code layout`** — regenerate the tree to match the current
   `src/memories/` (see the file list under "What Steps 8 and 9 Delivered"). Specifically: drop
   `Fact`/`Segment` from the models line; `facts.py` is blob path-based (GET/PUT/DELETE by
   dot-path, no category/mutability write fields); remove `implication.py`; `inferences.py` is
   generate / revalidate / delete only (no promote, no PATCH); add `schema.py`, `require_fact.py`,
   `fact_approval.py` routers; add `world_builder.py`, `tool_gate.py`, `sse_events.py`,
   `logging_config.py` and `chat_with_tools` to services; correct `prompt_builder` /
   `evaluator` / `inference_service` / `experience_service` signatures to their blob/path forms.
5. **`### Key patterns`** — update: seven tables created at startup (list them; note **no
   migrations**); **remove** the "Every session starts with a segment" pattern; rewrite the SSE
   event-sequence pattern to the current events (`status(extracting)` → `status(generating)` →
   `status(reviewing)` → optional contradiction/regenerating loop → optional `thinking` →
   `message` → `done`, plus `sidechannel` events: `world_builder_applied`, `fact_update_fluid`,
   `fact_update_mutable`, `fact_update_immutable_unset`, `require_fact`, `inference_proposed`,
   `contradiction`); **remove** "Implication acceptance"; rewrite the cascade pattern to be
   path-keyed (`cascade_on_fact_edit(changed_path)` / `cascade_on_fact_delete(deleted_path)`,
   user-initiated only, not fired by the World Builder); **add** a "tool-gate suspension"
   pattern (per-turn `asyncio.Queue` in `tool_gate.py` keyed by `(session_id, turn_id)`, awaited
   by blocking tool handlers, resolved by the `require_fact` / `fact_approval` endpoints).
   **Keep** the `_set_leaf` type-convention and `pass_name` bandit-nosec patterns verbatim.
   **Add** a "Logging" pattern paragraph reproducing the Part C level table and stating that
   `configure_logging()` is called at import in `main.py`, controlled by `LOG_LEVEL`.
6. **`### Test layout`** — correct the tree: remove `test_facts_repo.py`,
   `test_api_inference_promotion.py`, `test_api_implication.py` from the examples; add
   `test_logging_config.py`; note the `fact` fixture is gone.
7. **`### Configurable limits`** — add rows for **`MAX_TOOL_CALL_ROUNDS`** (default `10`,
   currently undocumented) and **`LOG_LEVEL`** (default `INFO`). Keep the existing eight rows.
8. **`### What's deferred`** — replace the segments/7a-7b bullet with the accurate deferred list
   from `plan-v2.md`: optimistic streaming (`docs/streaming-plan.md`), and Phase 7a/7b context
   budget + compression (which will **rebuild** the dropped `segments` table when revisited).

Also, in `## Commands`, add a one-line note that `LOG_LEVEL=DEBUG uv run uvicorn ...` raises log
verbosity for a debugging session.

### Part F — Rewrite `README.md`

The README is user-facing (assume a reader who is setting the app up, not editing it). Targeted
edits, not a full rewrite:

- **`## How it works`** ([README.md:5-14](README.md#L5-L14)) — replace the "two sequential LLM
  calls" framing with the three passes, described for a non-contributor: the World Builder reads
  your message and updates the world; the Character LLM replies in character; the Evaluator
  checks the reply against the established facts before you see it, regenerating on a
  contradiction. Mention that facts live in a fixed schema the character cannot invent new
  categories in. Keep the Experiences paragraph.
- **Model-selection guidance** ([README.md:48-50](README.md#L48-L50)) — the "returns a specific
  JSON structure with a fixed vocabulary of verdict strings" text is now wrong. Replace with:
  the evaluator and the other passes drive **tool calls**, so the model must support Ollama tool
  calling and follow tool schemas reliably; recommend a tool-calling-capable model. The
  hardware-sizing and thinking-model notes stay.
- **`## Environment variables`** ([README.md:123-133](README.md#L123-L133)) — bring the table to
  parity with the code: add `MAX_TOOL_CALL_ROUNDS` (`10`), `MAX_INFERENCE_DEPTH` (`5`),
  `MAX_INFERENCE_BREADTH` (`5`), `MIN_EXPERIENCE_SCORE` (`0.0`), and `LOG_LEVEL` (`INFO`).
- **Add a short `## Logging` subsection** (after Environment variables) explaining that logs go
  to stderr at `INFO` by default and that `LOG_LEVEL=DEBUG` surfaces full LLM prompts and
  tool-call payloads, useful when a character behaves unexpectedly.

---

## Transitional State After Step 10

This is the terminal state of the Facts v2 plan.

- `CLAUDE.md` and `README.md` describe the three-pass, tool-calling, schema-constrained system
  that actually exists. No reference remains to `Fact`, `Segment`, the `facts`/`segments`
  tables, `implication`, `experience_update`, or the JSON-verdict evaluator.
- `src/memories/logging_config.py` exists and is called once at `main.py` import. Running the
  server emits an `INFO` turn trace to stderr by default; `LOG_LEVEL=DEBUG` adds prompts and
  tool payloads. The turn trace covers: turn start, World Builder writes/rejections, Character
  Evaluator per-tool decisions, approval-card surfacing and resolution, the turn verdict, and
  any uncaught SSE-stream exception (with traceback).
- No functional behaviour changed. The full test suite, `ruff`, `mypy`, `bandit`, and the
  frontend tests are green.
- Manual testing can begin. Nothing further is planned in this plan; the next work items are
  the deferred streaming and context-budget phases, tracked separately.

---

## Test Plan

Only the new logging module carries code and therefore tests. The documentation rewrite has no
tests. The added log statements sit on already-covered branches (fact-write, evaluator-decision,
approval, and error paths all have existing tests), so they do not by themselves require new
tests — with the one exception of the SSE-exception path (D4), which gets a dedicated test.

### `tests/unit/test_logging_config.py` — new file

Uses a fixture that snapshots and restores the `memories` logger's `level`, `handlers`, and
`propagate` around each test, so the global logger singleton does not leak configuration
between tests (and does not leave a stderr handler attached for the rest of the suite):

```python
@pytest.fixture
def _restore_memories_logger():
    logger = logging.getLogger("memories")
    saved = (logger.level, list(logger.handlers), logger.propagate)
    yield
    logger.setLevel(saved[0])
    logger.handlers = saved[1]
    logger.propagate = saved[2]
```

Tests (all take `_restore_memories_logger`, and `monkeypatch` where env is involved):

- **`test_configure_logging_defaults_to_info`** — with `LOG_LEVEL` unset
  (`monkeypatch.delenv("LOG_LEVEL", raising=False)`), `configure_logging()` sets the `memories`
  logger level to `logging.INFO`.
- **`test_configure_logging_reads_log_level_env`** — `monkeypatch.setenv("LOG_LEVEL", "DEBUG")`;
  `configure_logging()` sets level to `logging.DEBUG`.
- **`test_configure_logging_explicit_arg_overrides_env`** —
  `monkeypatch.setenv("LOG_LEVEL", "ERROR")`; `configure_logging("WARNING")` sets level to
  `logging.WARNING` (the argument wins).
- **`test_configure_logging_invalid_level_falls_back_to_info`** —
  `configure_logging("NOTALEVEL")` sets level to `logging.INFO` and raises nothing.
- **`test_configure_logging_is_case_insensitive`** — `configure_logging("debug")` sets level to
  `logging.DEBUG`.
- **`test_configure_logging_attaches_single_handler`** — after `configure_logging()` the
  `memories` logger has exactly one `StreamHandler`.
- **`test_configure_logging_is_idempotent`** — calling `configure_logging()` twice leaves
  exactly one handler (no duplicate).
- **`test_configure_logging_emits_at_configured_level`** — a behavioural check: with
  `caplog.set_level(logging.INFO, logger="memories")`, call `configure_logging("INFO")` then
  `logging.getLogger("memories.test").info("hello")`; assert `"hello"` is in `caplog.text`.
  (Confirms records propagate and are captured — validates the `propagate=True` decision.)

### `tests/integration/test_api_chat.py` — one addition (D4)

- **Add `test_stream_logs_traceback_when_run_turn_raises`** — patch `run_turn` (in the
  `chat` router's namespace) to raise a distinctive exception; POST to
  `/api/sessions/{id}/messages`; assert the exception surfaces (stream errors / non-clean
  completion) **and** that `caplog` (set to `ERROR` on `memories.routers.chat`) contains the
  `"run_turn failed"` message with traceback. This covers the new `_log.exception` branch so
  coverage does not regress. If asserting on the raised exception through the SSE transport is
  awkward under `ASGITransport`, assert only on `caplog` — the log call is the code under test.

### Existing tests — expected to stay green unchanged

- All current unit and integration tests: the added `INFO`/`WARNING` log statements do not
  change return values or control flow. Confirm none currently assert on `caplog` in a way that
  a new adjacent log line would break (grep `caplog` in `tests/` — expected: few or none).
- `npm run test:coverage`: no JS changed; must stay at its threshold.

---

## Edge Cases

- **`propagate=True` and pytest `caplog`.** `caplog` captures via a handler on the **root**
  logger, reached only through propagation. The module deliberately leaves
  `memories.propagate = True` so `caplog` keeps working; had it set `propagate = False` (a
  tempting "clean" choice), every `caplog`-based assertion on a `memories.*` logger would
  silently capture nothing. This is the single most important non-obvious decision in the step —
  the `test_configure_logging_emits_at_configured_level` test pins it.
- **Duplicate handler on repeated configuration.** The `memories` logger is a process-global
  singleton. Without the `if not logger.handlers` guard, importing the module in a test and then
  configuring at app startup (or a test calling `configure_logging()` directly) would stack
  handlers and multiply every emitted line. The guard makes the call idempotent; the
  `_restore_memories_logger` fixture prevents cross-test leakage of the handler.
- **No double emission under uvicorn.** With a handler on `memories` **and** `propagate=True`,
  double emission would occur only if the root logger also had a handler. Under
  `uvicorn memories.main:app` root has none (uvicorn configures `uvicorn.*` loggers, not root),
  so each line is emitted once. Under pytest the root has `caplog`'s handler, so lines are both
  emitted to stderr (captured by pytest) and captured by `caplog` — no correctness problem, and
  `caplog` counts each record once regardless.
- **Log statements must not raise.** Every added statement uses `%`-style lazy formatting with
  arguments that are already in scope and already valid (`path`, `session_id`, an already-parsed
  `action`, etc.). None call a function that could fail. Truncations use sl; e.g.
  `user_content[:120]` is safe on any string. A logging call that itself raised would convert a
  successful turn into a failure — explicitly out of bounds.
- **Truncation of user/model text.** User messages and inference statements can be long and can
  contain newlines. Truncate to a bounded prefix (`[:120]` for messages, similar for statements)
  and rely on `%r` so embedded newlines are escaped into a single readable line rather than
  breaking the log into multiple lines.
- **No SSE / ASGITransport concurrency hazard in this step's tests.** The one new integration
  test (D4) makes a single request and asserts on `caplog`; it does not open a stream and hit a
  second endpoint concurrently, so the `ASGITransport` buffering limitation
  (`project_asgi_transport_streaming`) does not apply. If the raised-exception assertion proves
  awkward through the buffered transport, fall back to asserting on `caplog` alone.
- **Not an enum-validation change.** This step adds no Enum fact validation and no
  error-message-to-LLM text, so the enum-loop prompt risk
  (`project_enum_validation_prompt_risk`) is not in play. The World Builder per-entry error
  logging (D2) only mirrors, at `WARNING`, text already returned to the LLM; it does not change
  what the LLM receives.
- **Documentation accuracy is the deliverable, not prose quality.** The rewrite's correctness
  bar is that every symbol, table, router, verdict, and env var named in `CLAUDE.md` /
  `README.md` exists in the code as described. A useful final check: grep the finished docs for
  `Fact\b`, `segment`, `implication`, `experience_update`, `new_inference`, `promote`,
  `accept-implication` — any hit outside a deliberate "removed in Facts v2" note is a stale
  reference.

---

## Post-Implementation Cleanup Tasks

(To be populated by `/review-step` after implementation.)
