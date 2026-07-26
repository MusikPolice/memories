# Step 9 — Inference Path Migration & Legacy Fact-Layer Removal

## Overview

Step 9 is the final code step of the Facts v2 refactor (Step 10 is documentation only). It
does two related things: it finishes migrating the inference layer from integer fact ids to
schema paths, and it removes the entire legacy flat-fact layer that the migration renders
dead. After this step the whole application — World Builder, Character LLM, Character
Evaluator, inference generation/cascade, chat, and the session-end evaluator — runs purely
on the `character_facts` JSON blob and `fact_schema.json`. There is no `facts` table, no
`Fact` model, and no code path that reads or writes a flat key/value fact.

**The inference migration.** Inferences carry two parallel source columns: `source_fact_ids`
(integer ids into the legacy `facts` table, original design) and `source_fact_paths`
(dot-notation schema paths, added transitionally in Step 3). The Character Evaluator's
`propose_inference` handler has written `source_fact_paths` since Step 5, but the *eager
pass* and the *cascade* functions in `inference_service.py` still match on `source_fact_ids`
and read facts from the flat table via `get_fact_rows()`. This step removes `source_fact_ids`
entirely and rewrites the eager pass, revalidation, and both cascade functions to match on
paths and read facts from the blob.

**The consequential removals.** Dropping the integer-id column forces three removals that
together retire the legacy fact layer. (1) The `POST .../inferences/{id}/promote` endpoint —
already slated for deletion under Facts v2 — goes. (2) The `implication.py` router is
removed: its endpoints are dead since Step 8 (no frontend caller — confirmed by grep), and
it is the only remaining caller of `cascade_on_fact_edit()` with an integer fact id and of
the flat-fact write functions; because Step 9 changes that cascade's signature to a path
string, `implication.py` cannot compile and adapting it is meaningless. (3) With the promote
endpoint and `implication.py` gone, the flat `facts` table has exactly one remaining
consumer — the session-end evaluator — which this step migrates to the blob. That clears the
way to delete the `facts` table, the `Fact` model, the five flat-fact repository helpers, the
shared `fact` test fixture, and `test_facts_repo.py`. Finally, since this is the last code
step, it sweeps up adjacent dead inference-management code Step 8 left unwired — three unused
`chat.js` API helpers and the orphaned PATCH-status endpoint (Part I).

None of this is a backward-compat exercise: the DB is wiped on first run of the new schema
(Open Question 7), and every removed symbol is either dead or migrated in the same step. The
application stays working and the full suite stays green at the end of Step 9 — the extra
scope buys a codebase with **no dead legacy fact machinery**, which is the correct end state
for the last code step of the refactor.

**Success criterion:** `tests/unit/test_inference_service.py`,
`tests/integration/test_inferences_repo.py`,
`tests/integration/test_api_inference_generation.py`, and
`tests/unit/test_experience_service.py` pass against the path/blob implementation;
`tests/integration/test_api_inference_promotion.py`,
`tests/integration/test_api_implication.py`,
`tests/integration/test_api_extraction_resolution.py`, and
`tests/integration/test_facts_repo.py` are deleted; the removed promote and implication
routes return 404; and the full suite (`uv run pytest`), `uv run ruff check src/`,
`uv run mypy src/`, and `npm run test:coverage` are all green with no reference to `Fact`,
`create_fact`, `get_fact_rows`, or `source_fact_ids` anywhere in `src/`.

---

## What Steps 3, 5, 7, and 8 Delivered

- **`source_fact_paths TEXT` column** on `inferences`
  ([database.py:76](src/memories/database.py#L76)), added Step 3 alongside the legacy
  `source_fact_ids TEXT` ([database.py:74](src/memories/database.py#L74)). `_parse_inference`
  ([database.py:502](src/memories/database.py#L502)) already decodes all three source
  columns; `create_inference` ([database.py:514](src/memories/database.py#L514)) already
  accepts `source_fact_paths`.
- **`Inference` model** ([models/__init__.py:57](src/memories/models/__init__.py#L57)) has
  `source_fact_ids`, `source_inference_ids`, and `source_fact_paths`, all defaulting to `[]`.
- **`propose_inference` handler** in `evaluator.py` (Step 5) already writes
  `source_fact_paths` and never touches `source_fact_ids`. **`evaluator.py` is not changed.**
- **`character_facts` blob** repository: `get_facts(db, character_id) -> dict` (schema-masked,
  [database.py:280](src/memories/database.py#L280)), `set_facts(db, character_id, blob)`.
- **`schema_loader.py`**: `load_schema()`, `_collect_leaves(node, prefix="") ->
  list[tuple[str, dict]]`, `check_write_permitted(path, schema) -> str` (returns mutability;
  raises `ValueError` on unknown/grouping path), `render_current_fact_values(facts_blob)`.
- **Frontend (Step 8)** references no `implication.py` endpoint (grep of
  `src/memories/frontend/` for `accept-implication`, `accept-inference`, `undo-user-fact`,
  `accept-implicit`, etc. is empty). The inference panel exposes only expand + delete.
- **Only two test files use the shared `fact: Fact` fixture** —
  `tests/unit/test_inference_service.py` and
  `tests/integration/test_api_inference_generation.py` — both rewritten in this step; plus
  `test_api_extraction_resolution.py`, which is deleted. So the fixture has no surviving
  consumer once the inference tests are migrated.
- **`delete_fact` / `patch_fact_row` are already gone** (Step 8 CT-4's first half). The flat
  fact helpers remaining in `database.py` are exactly: `create_fact`, `get_fact_rows`,
  `get_fact`, `update_fact`, `get_fact_by_category_key`.

---

## What This Step Does NOT Change

- **`evaluator.py`, `world_builder.py`, `chat_service.py`, `prompt_builder.py`.** The
  Character Evaluator already writes `source_fact_paths`; the character system prompt already
  renders the blob. No change.
- **`GET /api/characters/{id}/facts` and `PUT /api/characters/{id}/facts`**
  (`routers/facts.py`). These operate on the `character_facts` blob, not the flat table, and
  are unchanged. In particular, `PUT /facts` still does **not** auto-cascade inferences —
  path-based revalidation remains an explicit separate call via the `revalidate` endpoint
  (see § E3), preserving the existing two-call pattern rather than wiring cascade into the
  fact-write path.
- **`compute_depth()`** — unchanged. It operates on `source_inference_ids` (integer
  inference→inference references, which stay) and is independent of the fact-source
  migration.
- **The Experiences layer beyond the session-end prompt** — retrieval, embedding, and the
  `experiences` table are untouched. Only `build_session_end_prompt` /
  `run_session_end_evaluator`'s `facts` parameter changes.
- **`CLAUDE.md` / `README.md`.** Documentation is Step 10.

---

## Detailed Design

Ordered by dependency: lowest-level (DB, schema helper, models) first.

### Part A — `src/memories/database.py`: drop `source_fact_ids`

#### A1. `inferences` DDL

Delete the `source_fact_ids TEXT,` line from the `CREATE TABLE inferences` statement
([database.py:74](src/memories/database.py#L74)). `source_inference_ids` and
`source_fact_paths` remain. No migration is written (Open Question 7).

#### A2. `_parse_inference`

Remove the `d["source_fact_ids"] = json.loads(...) if ... else []` line
([database.py:504](src/memories/database.py#L504)). Keep the `source_inference_ids` and
`source_fact_paths` decode lines.

#### A3. `create_inference`

Remove the `source_fact_ids` parameter, the `fact_ids_json` assignment, and its entries in
the column list and value tuple.

**Before → After signature:**

```python
# Before
async def create_inference(
    db, *, character_id, statement, derivation,
    source_fact_ids: list[int] | None = None,
    source_inference_ids: list[int] | None = None,
    source_fact_paths: list[str] | None = None,
    depth: int = 1, inference_type: str = "logical",
) -> Inference:

# After
async def create_inference(
    db, *, character_id, statement, derivation,
    source_inference_ids: list[int] | None = None,
    source_fact_paths: list[str] | None = None,
    depth: int = 1, inference_type: str = "logical",
) -> Inference:
```

The INSERT column list becomes `(character_id, statement, derivation, source_inference_ids,
source_fact_paths, depth, inference_type)` with the matching 7-value tuple.

(The flat-fact helper functions in this file — `create_fact`, `get_fact_rows`, `get_fact`,
`update_fact`, `get_fact_by_category_key` — are removed in **Part G**, after their consumers
are gone.)

### Part B — `src/memories/models/__init__.py`

Remove the `source_fact_ids: list[int] = []` field from `Inference`. (The `Fact` model is
removed in **Part G**.)

**Pydantic silent-drop warning (CLAUDE.md pitfall):** `Inference` uses the default config
(`extra="ignore"`). After this field is removed, `Inference(..., source_fact_ids=[…])` does
**not** raise — the kwarg is silently discarded — while `inf.source_fact_ids` now raises
`AttributeError`. Eager-pass JSON items that still carry a `"source_fact_ids"` key are
likewise silently ignored (parsed as `source_fact_paths == []`). The Test Plan enumerates
every affected construction and mock payload; a dedicated test pins the silent-drop of the
legacy JSON key so it is intentional, not accidental.

### Part C — `src/memories/schema_loader.py`: shared blob-walk helper

Add a public helper that both `inference_service.py` (eager pass + revalidation prompts) and
`experience_service.py` (session-end prompt) use to render populated leaves, so the
blob-walk lives in exactly one place:

```python
def iter_populated_leaves(
    facts_blob: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> list[tuple[str, str | int | float | bool]]:
    """Return (dot.path, value) for every schema leaf that has a Value in the blob,
    in schema declaration order. Unset leaves are skipped."""
    if schema is None:
        schema = load_schema()
    result: list[tuple[str, str | int | float | bool]] = []
    for path, _leaf in _collect_leaves(schema):
        node: Any = facts_blob
        found = True
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                found = False
                break
            node = node[part]
        if not found or not isinstance(node, dict):
            continue
        value = node.get("Value")
        if value is not None:
            result.append((path, value))
    return result
```

This mirrors the traversal already inside `render_current_fact_values`; that function may
optionally be refactored to call `iter_populated_leaves`, but is not required to change.
(Return-leaf type follows CLAUDE.md's `_set_leaf` convention: `str | int | float | bool`,
never `Any`.)

### Part D — `src/memories/services/inference_service.py`: path/blob migration

#### D1. Imports

```python
# Before
from memories.database import (create_inference, get_character, get_fact_rows,
                               get_inferences, update_inference_status)
from memories.models import Character, Fact, Inference

# After
from typing import Any
from memories.database import (create_inference, get_character, get_facts,
                               get_inferences, update_inference_status)
from memories.models import Character, Inference
from memories.schema_loader import iter_populated_leaves
```

Also add `def _coerce_str_list(raw: object) -> list[str]` next to the existing
`_coerce_id_list` (returns `[v for v in raw if isinstance(v, str)]`), used to parse
`source_fact_paths` out of eager-pass JSON.

#### D2. `build_eager_pass_prompt` — blob param, cite paths

- Signature: `facts: list[Fact]` → `facts_blob: dict[str, Any]`.
- Render the "Current Facts" block from the shared helper:
  ```python
  lines.append("## Current Facts (path: value)")
  for path, value in iter_populated_leaves(facts_blob):
      lines.append(f"{path}: {value}")
  ```
- In the Rules/JSON template, replace `"source_fact_ids": [int, ...]` with
  `"source_fact_paths": ["Character.Identity.Age", ...]` and change "Cite source Facts and
  Inferences by id" to "Cite source Facts by their schema path and source Inferences by id."
  `source_inference_ids` stays (integer).

#### D3. `run_eager_pass` — blob param, parse paths

- Signature: `facts` → `facts_blob: dict[str, Any]`; pass it to `build_eager_pass_prompt`.
- Per item: `src_paths = _coerce_str_list(item.get("source_fact_paths", []))`;
  `src_inf_ids = _coerce_id_list(item.get("source_inference_ids", []))`.
- Cross-reference guard, depth cap, breadth cap unchanged (they use `src_inf_ids`).
- `create_inference` call passes `source_fact_paths=src_paths` (drop `source_fact_ids=`).

#### D4. `build_revalidation_prompt` — blob param, reference paths

- Signature: `facts` → `facts_blob: dict[str, Any]`; render "Current Facts" from
  `iter_populated_leaves(facts_blob)`.
- Change the "Original sources" line to reference `inference.source_fact_paths`:
  ```python
  f"Original sources: Facts {inference.source_fact_paths}, "
  f"Inferences {inference.source_inference_ids}",
  ```

#### D5. `revalidate_single_inference` — blob param

Signature `facts` → `facts_blob: dict[str, Any]`; pass through to `build_revalidation_prompt`.
Body otherwise identical (LLM call, parse, conservative fallback).

#### D6. `cascade_on_fact_edit` — path-keyed

```python
# Before
async def cascade_on_fact_edit(db, character_id, changed_fact_id: int, ollama):
    ...
    facts = await get_fact_rows(db, character_id)
    worklist = [inf for inf in non_invalidated if changed_fact_id in inf.source_fact_ids]
    holds = await revalidate_single_inference(inference, facts, remaining_active, ollama, model=model)

# After
async def cascade_on_fact_edit(db, character_id, changed_path: str, ollama):
    ...
    facts_blob = await get_facts(db, character_id)
    worklist = [inf for inf in non_invalidated if changed_path in inf.source_fact_paths]
    holds = await revalidate_single_inference(inference, facts_blob, remaining_active, ollama, model=model)
```

The BFS worklist, stale-propagation, and already-stale-seed logic (Bug 6 fix) are otherwise
unchanged.

#### D7. `cascade_on_fact_delete` — path-keyed

Rename `deleted_fact_id: int` → `deleted_path: str`; seed comprehension becomes
`deleted_path in inf.source_fact_paths`. Transitive expansion over `source_inference_ids`
unchanged; no LLM call, as before.

### Part E — `src/memories/routers/inferences.py`

#### E1. Imports / dead helpers

- Drop imports: `_parse_inference`, `_row`, `get_fact_rows`, `Fact`, `Literal`.
- Add imports: `get_facts` (database), `check_write_permitted`, `load_schema` (schema_loader).
- Delete `_fact_exists` ([inferences.py:47-49](src/memories/routers/inferences.py#L47-L49)).

#### E2. `generate_inferences_endpoint`

`facts = await get_fact_rows(db, character_id)` → `facts_blob = await get_facts(db,
character_id)`; pass `facts_blob` to `run_eager_pass`.

#### E3. `revalidate_inferences_endpoint`

```python
# Before
class _RevalidateBody(BaseModel):
    changed_fact_id: int
...
    if not await _fact_exists(db, body.changed_fact_id):
        raise HTTPException(status_code=404, detail="Fact not found")
    stale = await cascade_on_fact_edit(db, character_id, body.changed_fact_id, ollama)

# After
class _RevalidateBody(BaseModel):
    changed_path: str
...
    try:
        check_write_permitted(body.changed_path, load_schema())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    stale = await cascade_on_fact_edit(db, character_id, body.changed_path, ollama)
```

Unknown or grouping path → 422 (consistent with `PUT /facts`); a valid leaf with no
dependent inferences → 200 with `stale_inferences: []`.

#### E4. Remove the promote endpoint

Delete `_PromoteBody` and `promote_inference_endpoint`
([inferences.py:40-45](src/memories/routers/inferences.py#L40-L45),
[138-215](src/memories/routers/inferences.py#L138-L215)) — the only user of
`aiosqlite.IntegrityError`, the raw `facts` INSERT, and `_parse_inference`/`_row` in this
router.

### Part F — Remove `implication.py`

- Delete `src/memories/routers/implication.py` in full.
- `src/memories/main.py`: remove `implication` from the `from memories.routers import (...)`
  tuple ([main.py:25](src/memories/main.py#L25)) and delete the mount line
  ([main.py:84](src/memories/main.py#L84)).

Grep confirms no other production importer (the only other match is a prose comment in
`world_builder.py`).

### Part G — Migrate the session-end evaluator to the blob

**File: `src/memories/services/experience_service.py`**

`build_session_end_prompt` and `run_session_end_evaluator` take `facts: list[Fact]` and
render `[{f.id}] {f.key}: {f.value}`. Migrate both to the blob:

```python
# build_session_end_prompt — Before
def build_session_end_prompt(character, facts: list[Fact], inferences, messages) -> str:
    ...
    if facts:
        for f in facts:
            parts.append(f"[{f.id}] {f.key}: {f.value}")
    else:
        parts.append("(none)")

# After
def build_session_end_prompt(character, facts_blob: dict[str, Any], inferences, messages) -> str:
    ...
    fact_lines = iter_populated_leaves(facts_blob)
    if fact_lines:
        for path, value in fact_lines:
            parts.append(f"{path}: {value}")
    else:
        parts.append("(none)")
```

- `run_session_end_evaluator`: rename its `facts: list[Fact]` param to
  `facts_blob: dict[str, Any]`; pass it to `build_session_end_prompt`.
- Imports: drop `Fact` from `from memories.models import ...`; add
  `from memories.schema_loader import iter_populated_leaves`. (`Any` is already imported.)

**File: `src/memories/routers/sessions.py`**

- Import: `get_fact_rows` → `get_facts`.
- In `end_session_endpoint`: `facts = await get_fact_rows(db, session.character_id)` →
  `facts_blob = await get_facts(db, session.character_id)`; call
  `run_session_end_evaluator(character, facts_blob, inferences, messages, ollama)`.

### Part H — Delete the flat-fact layer

With `implication.py`, the promote endpoint, and the session-end `list[Fact]` consumer all
gone, nothing reads the flat `facts` table. Remove:

- **`src/memories/database.py`**: the five helpers `create_fact`, `get_fact_rows`,
  `get_fact`, `update_fact`, `get_fact_by_category_key`; and drop `Fact` from the
  `from memories.models import (...)` import ([database.py:21](src/memories/database.py#L21)).
  Leave the `facts` `CREATE TABLE` DDL removal to the note below.
- **The `facts` table DDL** in `init_db()`'s schema string (and its
  `CREATE INDEX idx_facts_character`). The DB is wiped on first run; no migration.
- **`src/memories/models/__init__.py`**: remove the `Fact` class (and any `__all__` entry).

After this, `grep -rw Fact src/` returns only substring matches like `character_facts` /
`## Fact Schema` / `Fact.Schema`-in-prose — no `Fact` model reference remains.

### Part I — Remove dead inference-management vestiges

Post-Step-8 the inference side-panel is **delete-only**: `index.html` has no generate,
revalidate, promote, or status-toggle control, and `chat-component.js` imports only
`apiDeleteInference`. Three `chat.js` helpers and one backend endpoint are leftover
Phase-3/4 inference-management machinery with no live caller.

**Backend — `src/memories/routers/inferences.py`:**

- Remove `patch_inference_endpoint` (PATCH `/{character_id}/inferences/{inference_id}`) and
  its `_PatchBody` model. Under Facts v2 inferences are steered by **deletion only**; manual
  status toggling (active / stale / invalidated) is a vestige of the old inference-management
  UI with no live caller. Drop the now-unused `update_inference_status` import from this
  router (it stays in `database.py` — the cascade functions still use it — only the HTTP
  endpoint goes).
- **Keep** `generate` (eager pass), `revalidate` (cascade), and `delete` — all
  plan-sanctioned user-triggered operations.

**Frontend — `src/memories/frontend/chat.js`:**

- Remove `apiGenerateInferences`, `apiRevalidateInferences`, and `apiPatchInferenceStatus`
  (definitions and their entries in any export/import list). None are imported by
  `chat-component.js`; none have a control in `index.html`. `apiRevalidateInferences`
  additionally posts the stale `{changed_fact_id}` body (the endpoint now takes
  `changed_path`), and `apiPatchInferenceStatus` targets the endpoint being removed above.
  The backend `generate` / `revalidate` endpoints remain as API surface; a future
  fact-management UI that wires them up should add fresh, correctly-shaped helpers at that
  point rather than resurrecting these.

---

## Transitional State After Step 9

- Inferences carry only `source_inference_ids` (integer) and `source_fact_paths` (schema
  paths). `source_fact_ids` is gone from the DB, the model, and all code.
- Inference generation, revalidation, both cascades, and the session-end evaluator read
  facts from the `character_facts` blob. `cascade_on_fact_edit`/`_delete` and the
  `revalidate` endpoint take a `changed_path` string.
- The promote endpoint, the PATCH `/inferences/{id}` status endpoint, and `implication.py`
  are gone; their routes 404. The inference API is now generate / revalidate / delete only.
- **There is no flat-fact layer.** No `facts` table, no `Fact` model, no `create_fact` /
  `get_fact_rows` / etc. The session-end evaluator now sees the real Facts v2 blob (it
  previously read an always-empty flat table).
- The Facts v2 refactor is functionally complete. Step 10 updates `CLAUDE.md` / `README.md`
  to describe the delivered system.

---

## Test Plan

### Files to DELETE

- `tests/integration/test_api_inference_promotion.py` — promote endpoint removed.
- `tests/integration/test_api_implication.py` — `implication.py` removed.
- `tests/integration/test_api_extraction_resolution.py` — tests `implication.py`'s Phase-6
  endpoints.
- `tests/integration/test_facts_repo.py` — tests only the removed flat-fact repo helpers
  (`create_fact`, `get_fact_rows`, `update_fact`, `get_fact_by_category_key`).

### `tests/integration/test_inferences_repo.py` — changes

- **Delete** `test_source_fact_ids_stored_and_retrieved_as_list` — field removed (reading
  `inf.source_fact_ids` would `AttributeError`).
- **Update** `test_create_inference_stores_statement_and_derivation` — its
  `create_inference(..., source_fact_ids=[1])` now raises `TypeError`; change the kwarg to
  `source_fact_paths=["Character.Identity.Age"]`.
- **Keep** the three Step-3 `source_fact_paths` tests unchanged.

### `tests/unit/test_inference_service.py` — changes

**Module fixtures:** replace `_FACTS: list[Fact]` with a blob constant `_FACTS_BLOB =
{"Character": {"Identity": {"Age": {"Value": 33}}}, "Setting": {"Temporal":
{"Current-Year": {"Value": 2026}}}}`; drop the `Fact` import. In `_EXISTING_INFERENCES`,
`_REVALIDATION_INFERENCE`, `_OTHER_INFERENCE`, replace `source_fact_ids=[…]` with
`source_fact_paths=[…]` using real paths. In `_EAGER_PASS_ITEM`, replace the JSON key
`"source_fact_ids"` with `"source_fact_paths"` and use path values.

- **`build_eager_pass_prompt` tests** (`_includes_all_facts`, `_includes_character_name`,
  `_lists_existing_inferences`, `_no_existing_inferences_uses_fallback`,
  `_instructs_max_breadth`) — pass `_FACTS_BLOB`; `_includes_all_facts` asserts a path
  substring like `"Character.Identity.Age"` and `"33"`.
- **Add** `test_eager_pass_prompt_requests_source_fact_paths` — asserts the JSON template
  contains `"source_fact_paths"` and not `"source_fact_ids"`.
- **`run_eager_pass` tests** (all eleven: `_parses_returned_inferences`,
  `_stores_inferences_to_db`, `_returns_empty_list_on_empty_json_array`, `_applies_breadth_cap`,
  `_rejects_same_pass_cross_reference`, `_computes_depth_one_for_fact_only_source`,
  `_computes_depth_from_source_inference`, `_discards_inference_exceeding_max_depth`,
  `_raises_on_non_json_response`, `_request_sends_format_json`, `_request_sends_think_false`)
  — drop the `fact` fixture; call `run_eager_pass(db, character, _FACTS_BLOB, …)`; items that
  cited `source_fact_ids=[fact.id]` now cite `source_fact_paths=["Character.Identity.Age"]`.
- **Add** `test_eager_pass_persists_source_fact_paths` — item cites a path; assert the stored
  inference's `source_fact_paths == ["Character.Identity.Age"]`.
- **Add** `test_eager_pass_ignores_legacy_source_fact_ids_key` — an item carrying only the
  old `"source_fact_ids": [1]` key is stored with `source_fact_paths == []` (pins the
  silent-drop).
- **`build_revalidation_prompt` tests** (`_includes_inference_statement`,
  `_includes_inference_derivation`, `_includes_current_facts`,
  `_includes_other_active_inferences`) — pass `_FACTS_BLOB`; `_includes_current_facts`
  asserts a path substring + `"33"`.
- **`revalidate_single_inference` tests** (`_returns_true_when_inference_holds`,
  `_returns_false_when_inference_does_not_hold`, `_defaults_to_true_on_parse_error`) — pass
  `_FACTS_BLOB`.
- **`cascade_on_fact_edit` tests** (all seven) — create inferences with
  `source_fact_paths=["Character.Identity.Occupation"]`; call
  `cascade_on_fact_edit(db, character.id, "Character.Identity.Occupation", ollama)`; the two
  raw-SQL `INSERT ... source_fact_ids, source_inference_ids ...` statements switch the column
  to `source_fact_paths` with `json.dumps(["Character.Identity.Occupation"])`; the unrelated
  and transitive tests use distinct paths / `source_inference_ids` as before.
- **`cascade_on_fact_delete` tests** (all five) — create inferences with
  `source_fact_paths=[…]`; call `cascade_on_fact_delete(db, character.id,
  "Character.Identity.Occupation")`; unrelated uses a different path.
- **`compute_depth` tests** — unchanged.

### `tests/integration/test_api_inference_generation.py` — changes

- **Generate tests** — `_DEFAULT_EAGER_ITEM`'s `"source_fact_ids": []` key → `"source_fact_paths": []`;
  where items cited `source_fact_ids=[fact.id]`, use `source_fact_paths=["Character.Identity.Age"]`;
  the `fact` fixture dependency is dropped (the endpoint reads the blob; the LLM is mocked so
  blob contents do not affect what is stored).
- **Revalidate tests** — body becomes `{"changed_path": "Character.Identity.Occupation"}`;
  inferences cite matching `source_fact_paths`; `test_revalidate_does_not_affect_unrelated_inferences`
  uses a distinct path; `test_revalidate_unknown_character_returns_404` keeps 404 (character
  check first); **rename** `test_revalidate_unknown_fact_returns_404` →
  `test_revalidate_unknown_path_returns_422` (POST `{"changed_path": "Nope.Not.Here"}` → 422).
- **Delete management tests** (`test_delete_inference_*`) — unchanged.
- **Delete the PATCH-status tests** — `test_patch_inference_status_to_active_returns_200`,
  `test_patch_inference_status_to_stale_returns_200`, `test_patch_inference_status_updates_db`,
  and `test_patch_inference_unknown_id_returns_404` are removed with the endpoint (Part I).
- Drop the now-unused `Fact` import from the module header.

### `tests/unit/test_experience_service.py` — changes

- Replace `_FACTS = [Fact(...)]` with `_FACTS_BLOB = {"Character": {"Identity": {"Occupation":
  {"Value": "surgeon"}}}}`; drop the `Fact` import.
- Every `build_session_end_prompt(_CHARACTER, _FACTS, …)` and
  `run_session_end_evaluator(_CHARACTER, _FACTS, …)` call passes `_FACTS_BLOB`.
- `test_session_end_prompt_includes_all_facts` — assert `"Character.Identity.Occupation"` and
  `"surgeon"` (was lowercase `"occupation"`).
- `test_session_end_prompt_no_facts_shows_fallback` — pass `{}` instead of `[]`; still `"(none)"`.
- The inference constant that used `source_fact_ids=[1]` → `source_fact_paths=[…]`.

### `tests/integration/test_api_chat.py` — changes

- **Delete** `test_accept_implication_on_high_mutability_fact_preserves_mutability`
  ([line 995](tests/integration/test_api_chat.py#L995)) — it exercises the removed
  `accept-implication` endpoint and `get_fact_rows`.
- **Update** `test_send_message_system_message_includes_inferences`
  ([line 479](tests/integration/test_api_chat.py#L479)) — remove the dead
  `create_fact(age=33)` seed; the test asserts on the inference in the system prompt, which is
  unaffected.
- **Delete or defang** `test_chat_system_prompt_groups_user_and_character_facts`
  ([line 615](tests/integration/test_api_chat.py#L615)) — its two `create_fact` seeds are dead
  (the "User"/"Character" substrings come from the schema tree, not flat facts). It is now a
  near-duplicate of `test_chat_system_prompt_renders_all_schema_sections`; **delete it** (the
  schema-section test already covers the assertion), or if kept, remove the two `create_fact`
  seeds.
- Remove the now-unused `create_fact` and `get_fact_rows` imports.

### `tests/unit/test_chat_service.py` — changes

- **Update** `test_run_turn_loads_inferences_for_character`
  ([line 478](tests/unit/test_chat_service.py#L478)) — remove the dead `create_fact(age=33)`
  seed ([line 482](tests/unit/test_chat_service.py#L482)); the test asserts on the inference.
- Remove the now-unused `create_fact` import.

### `tests/unit/test_prompt_builder.py` / `tests/unit/test_evaluator_service.py` / `tests/integration/test_api_facts.py` — changes

- **`test_prompt_builder.py`** (lines 142, 166, 178, 190, 326): bare `Inference(...,
  source_fact_ids=[…])` constructions — rename each kwarg to `source_fact_paths=[…]` (they are
  silently dropped otherwise; assertions don't read the field).
- **`test_evaluator_service.py`** (~line 107): module-level `Inference(...,
  source_fact_ids=[1])` → `source_fact_paths=["Character.Identity.Age"]`.
- **`test_api_facts.py`** (~line 146-150): the `create_inference(db, …, source_fact_ids=[])`
  setup call now raises `TypeError`; change the kwarg to `source_fact_paths=[]` (or drop it).

### `tests/frontend/chat.test.js` — changes (Part I)

- Remove `apiGenerateInferences`, `apiRevalidateInferences`, and `apiPatchInferenceStatus`
  from the import block ([lines 7-10](tests/frontend/chat.test.js#L7-L10)).
- Delete their five tests: `apiGenerateInferences_posts_to_correct_url`,
  `apiRevalidateInferences_posts_to_correct_url`,
  `apiRevalidateInferences_sends_changed_fact_id_in_body`,
  `apiPatchInferenceStatus_sends_patch_to_correct_url`, and
  `apiPatchInferenceStatus_sends_status_in_body`.
- Keep the `apiDeleteInference` test and all schema-tree / notification / experience helper
  tests. `npm run test:coverage` must stay green at the 80% threshold on `chat.js`.

### Fixture removals

- **`tests/integration/conftest.py`** — delete the `fact` fixture
  ([line 63](tests/integration/conftest.py#L63)) and drop `Fact` from the
  `from memories.models import ...` line ([line 16](tests/integration/conftest.py#L16)).
- **`tests/unit/conftest.py`** — delete the `fact` fixture
  ([line 222](tests/unit/conftest.py#L222)) and drop `Fact` from the
  `from memories.models import ...` line ([line 14](tests/unit/conftest.py#L14)).
- Confirm (grep `fact: Fact`) no remaining test requests the fixture after the inference-test
  rewrite.

---

## Edge Cases

- **Pydantic silent-drop of the legacy `source_fact_ids` key (highest-risk pitfall).** After
  the model field is removed, eager-pass JSON items or `Inference(...)` constructions that
  still carry `source_fact_ids` are accepted with **no error** and no fact provenance.
  `create_inference(...)` calls fail loudly (`TypeError`) and are easy to fix; bare
  `Inference(...)` constructions and raw JSON mock items fail **silently** — the Test Plan
  enumerates each, and `test_eager_pass_ignores_legacy_source_fact_ids_key` locks the
  behaviour in.
- **Session-end prompt now renders real facts.** Before Step 9 the session-end evaluator read
  an always-empty flat table, so `## Character Facts` was effectively always `(none)`. After
  migration it renders the populated blob leaves as `path: value`. Existing session-end tests
  mock Ollama and assert on parsed output, not prompt fact content, so the change is
  behaviour-preserving for them once `_FACTS_BLOB` replaces `_FACTS`.
- **`revalidate` with a valid path but no dependent inferences** → 200 with
  `stale_inferences: []`. This differs from the old endpoint, which 404'd on a nonexistent
  integer `changed_fact_id`. Under the blob model there is no "fact existence" to check (any
  leaf may be unset), so the only validation is that the path is a real schema leaf (422
  otherwise). The 422 body carries a single `check_write_permitted` message — a plain
  validation error, not an enum constraint list, so the enum-loop risk
  (`project_enum_validation_prompt_risk`) does not apply.
- **Cascade matches by exact path string.** `changed_path in inf.source_fact_paths` is exact
  equality — editing `Character.Identity.Age` does not cascade inferences sourced from other
  paths, and grouping paths never appear as sources. This mirrors the old exact-id matching.
- **No SSE endpoints touched — ASGITransport limitation not applicable.** `generate`,
  `revalidate`, and `POST /sessions/{id}/end` are plain request/response; there is no
  streaming and no concurrent-request-during-stream scenario, so the `ASGITransport` buffering
  hazard (`project_asgi_transport_streaming`) does not arise in this step's tests.
- **`grep`-clean invariant.** A useful implementation check: after the step, `grep -rn
  "source_fact_ids\|get_fact_rows\|create_fact\|\bFact\b" src/` should return only innocuous
  substring hits (`character_facts`, `## Fact Schema`, prose). Any real symbol hit is an
  unfinished removal.
- **Dead inference-management helpers (removed in Part I).** `chat.js`'s
  `apiGenerateInferences` / `apiRevalidateInferences` / `apiPatchInferenceStatus` and the
  backend PATCH-status endpoint have no live caller post-Step-8 and are deleted in this step.
  `apiRevalidateInferences` in particular posted the now-stale `{changed_fact_id}` body. The
  `generate` / `revalidate` backend endpoints remain as API surface; a future fact-management
  UI that wires them up adds fresh, correctly-shaped helpers rather than resurrecting these.

---

## Post-Implementation Cleanup Tasks

No cleanup tasks identified. Implementation matches the spec.

All nine Parts (A–I) are implemented exactly as specified. `grep` for `source_fact_ids`,
`get_fact_rows`, `create_fact`, and the `Fact` model in `src/` returns only innocuous
substring hits; all cascade callers pass `changed_path` strings; the Pydantic silent-drop of
the legacy `source_fact_ids` key is pinned by a dedicated test; and the removed promote,
PATCH-status, and implication routes 404. The full suite (684 passed, 4 skipped, 93.93 %
coverage), `ruff check src/`, `mypy src/`, and `npm run test:coverage` (100 % on `chat.js`)
are all green.

One process note, already resolved in the feat commit: the Test Plan omitted
`tests/integration/test_db_init.py`, whose `facts`-table assertions broke when the DDL was
removed. The implementer caught this gap and retired those assertions in `feat(step9)`; no
action remains.
