# Step 9 — Inference Path Migration

## Overview

Step 9 finishes the inference layer's migration from the old flat-fact model to the
Facts v2 schema-path model. Inferences currently carry two parallel source columns:
`source_fact_ids` (integer ids into the legacy `facts` table, from the original design)
and `source_fact_paths` (dot-notation schema paths, added transitionally in Step 3). The
Character Evaluator's `propose_inference` handler has written `source_fact_paths` since
Step 5, but the *eager pass* and the *cascade* functions in `inference_service.py` still
read and match on `source_fact_ids` and still read facts from the legacy flat `facts`
table via `get_fact_rows()`. This step removes `source_fact_ids` entirely and rewrites the
eager pass, revalidation, and both cascade functions to work against schema paths and the
`character_facts` JSON blob.

Removing the integer-id column forces two consequential removals. First, the
`POST /api/characters/{id}/inferences/{id}/promote` endpoint — already slated for deletion
in the plan (inference promotion is removed under Facts v2) — is deleted. Second, and less
obviously, the legacy `implication.py` router must be removed. Its endpoints
(`accept-implication`, `accept-inference`, and the Phase-6 extraction-resolution handlers)
are dead since Step 8 (the UI no longer calls any of them, confirmed by grep), and they are
the only remaining callers of `cascade_on_fact_edit()` with an integer fact id and of the
legacy flat-fact write functions. Because Step 9 changes `cascade_on_fact_edit()`'s
signature from `changed_fact_id: int` to `changed_path: str`, `implication.py` cannot
compile against the new signature, and adapting it is meaningless (its cascade could never
match anything once inferences no longer carry integer ids). Deleting it is the only
coherent option, and both Step 7 and Step 8 explicitly deferred its removal to "Step 9+".

The legacy flat `facts` table and its read helper `get_fact_rows()` are **retained**: the
session-end evaluator (`sessions.py` → `run_session_end_evaluator`) still consumes
`list[Fact]`, and the shared integration `fact` fixture still calls `create_fact()`.
Migrating the session-end evaluator to the blob is out of scope (the plan does not schedule
it), so the flat-fact machinery survives this step as documented dead-ish weight; only its
`implication.py`-specific consumers go.

**Success criterion:** `tests/unit/test_inference_service.py`,
`tests/integration/test_inferences_repo.py`, and
`tests/integration/test_api_inference_generation.py` pass against the path-based
implementation; `tests/integration/test_api_inference_promotion.py`,
`tests/integration/test_api_implication.py`, and
`tests/integration/test_api_extraction_resolution.py` are deleted; a `GET` to the removed
promote endpoint returns 404; and the full suite (`uv run pytest`) plus
`uv run ruff check src/` and `uv run mypy src/` are green.

---

## What Steps 3, 5, 7, and 8 Delivered

Concrete foundations this step builds on (all verified present in the current tree):

- **`source_fact_paths TEXT` column** on the `inferences` table
  ([database.py:76](src/memories/database.py#L76)), added in Step 3 alongside the existing
  `source_fact_ids TEXT` ([database.py:74](src/memories/database.py#L74)). `_parse_inference`
  ([database.py:502](src/memories/database.py#L502)) already JSON-decodes all three source
  columns. `create_inference` ([database.py:514](src/memories/database.py#L514)) already
  accepts `source_fact_paths: list[str] | None`.
- **`Inference` model** ([models/__init__.py:57](src/memories/models/__init__.py#L57))
  carries `source_fact_ids: list[int]`, `source_inference_ids: list[int]`, and
  `source_fact_paths: list[str]`, all defaulting to `[]`.
- **`propose_inference` handler** in `evaluator.py` (Step 5) already writes
  `source_fact_paths` (from the tool's `source_paths` argument) and passes
  `source_inference_ids=[]`; it never touches `source_fact_ids`. **No change to
  `evaluator.py` in this step.**
- **`character_facts` blob** repository: `get_facts(db, character_id) -> dict`
  ([database.py:280](src/memories/database.py#L280), schema-masked) and
  `set_facts(db, character_id, blob)`.
- **`schema_loader.py`** helpers: `load_schema()`, `_collect_leaves(node, prefix="") ->
  list[tuple[str, dict]]` (flattens the schema to `(dot.path, leaf_dict)` pairs),
  `check_write_permitted(path, schema) -> str` (returns the mutability string; raises
  `ValueError` for an unknown path or a grouping path).
- **Frontend (Step 8)** no longer references any `implication.py` endpoint (grep of
  `src/memories/frontend/` for `accept-implication`, `accept-inference`, `undo-user-fact`,
  `accept-implicit`, etc. returns nothing). The inference side-panel offers only expand and
  delete; the promote-to-fact button was removed in Step 8.

---

## What This Step Does NOT Change

- **`evaluator.py`, `world_builder.py`, `chat_service.py`, `prompt_builder.py`.** The
  Character Evaluator already writes `source_fact_paths`; no service under `services/` other
  than `inference_service.py` is touched.
- **The legacy flat `facts` table and its helpers `create_fact`, `get_fact`,
  `get_fact_by_category_key`, `update_fact`, `get_fact_rows`, `delete_fact`,
  `patch_fact_row`.** These are deliberately retained. After `implication.py` is removed
  they have no production callers except `sessions.py` (`get_fact_rows`, for the session-end
  evaluator) and the integration `fact` fixture (`create_fact`). Fully removing them —
  Step 8's CT-4 follow-up — requires migrating the session-end evaluator and the shared
  `fact` fixture, which is a separate change. **Out of scope.** The `Fact` model stays.
- **`GET /api/characters/{id}/facts` and `PUT /api/characters/{id}/facts`**
  (`routers/facts.py`). Fact reads/writes are unchanged. In particular, `PUT /facts` still
  does **not** trigger an inference cascade — path-based revalidation remains an explicit,
  separate call through the `revalidate` endpoint (see Detailed Design § D), preserving the
  existing two-call pattern rather than wiring cascade into the fact-write path.
- **`compute_depth()`** — unchanged. It operates on `source_inference_ids` (inference→
  inference references, which are still integer ids) and is independent of the fact-source
  migration.
- **`sessions.py`, `experiences.py`, `experience_service.py`.** The session-end evaluator
  keeps reading flat facts via `get_fact_rows`.
- **`CLAUDE.md` / `README.md`.** Documentation is Step 10.

---

## Detailed Design

### Part A — `src/memories/database.py`

#### A1. Drop `source_fact_ids` from the `inferences` DDL

**Before** ([database.py:69–81](src/memories/database.py#L69-L81)):

```sql
CREATE TABLE IF NOT EXISTS inferences (
    id                    INTEGER PRIMARY KEY,
    character_id          INTEGER REFERENCES characters(id),
    statement             TEXT NOT NULL,
    derivation            TEXT NOT NULL,
    source_fact_ids       TEXT,
    source_inference_ids  TEXT,
    source_fact_paths     TEXT,
    depth                 INTEGER NOT NULL DEFAULT 1,
    inference_type        TEXT NOT NULL DEFAULT 'logical',
    status                TEXT NOT NULL DEFAULT 'active',
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**After** — delete the `source_fact_ids TEXT,` line. `source_inference_ids` and
`source_fact_paths` remain. No migration is written (Open Question 7: the DB is wiped on
first run of the new schema).

#### A2. `_parse_inference` — drop the `source_fact_ids` decode

**Before** ([database.py:502–511](src/memories/database.py#L502-L511)):

```python
def _parse_inference(row: aiosqlite.Row) -> Inference:
    d = _row(row)
    d["source_fact_ids"] = json.loads(d["source_fact_ids"]) if d.get("source_fact_ids") else []
    d["source_inference_ids"] = (
        json.loads(d["source_inference_ids"]) if d.get("source_inference_ids") else []
    )
    d["source_fact_paths"] = (
        json.loads(d["source_fact_paths"]) if d.get("source_fact_paths") else []
    )
    return Inference.model_validate(d)
```

**After** — remove the `d["source_fact_ids"] = …` line only. The other two lines stay.

#### A3. `create_inference` — drop the `source_fact_ids` parameter

**Before** ([database.py:514–544](src/memories/database.py#L514-L544)):

```python
async def create_inference(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    statement: str,
    derivation: str,
    source_fact_ids: list[int] | None = None,
    source_inference_ids: list[int] | None = None,
    source_fact_paths: list[str] | None = None,
    depth: int = 1,
    inference_type: str = "logical",
) -> Inference:
    fact_ids_json = json.dumps(source_fact_ids or [])
    inf_ids_json = json.dumps(source_inference_ids or [])
    paths_json = json.dumps(source_fact_paths or [])
    cursor = await db.execute(
        """INSERT INTO inferences
               (character_id, statement, derivation, source_fact_ids, source_inference_ids,
                source_fact_paths, depth, inference_type)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            character_id,
            statement,
            derivation,
            fact_ids_json,
            inf_ids_json,
            paths_json,
            depth,
            inference_type,
        ),
    )
    ...
```

**After:**

```python
async def create_inference(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    statement: str,
    derivation: str,
    source_inference_ids: list[int] | None = None,
    source_fact_paths: list[str] | None = None,
    depth: int = 1,
    inference_type: str = "logical",
) -> Inference:
    inf_ids_json = json.dumps(source_inference_ids or [])
    paths_json = json.dumps(source_fact_paths or [])
    cursor = await db.execute(
        """INSERT INTO inferences
               (character_id, statement, derivation, source_inference_ids,
                source_fact_paths, depth, inference_type)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            character_id,
            statement,
            derivation,
            inf_ids_json,
            paths_json,
            depth,
            inference_type,
        ),
    )
    ...
```

(Remove `source_fact_ids` from the signature, the `fact_ids_json` assignment, the column
list, and the value tuple. The trailing fetch-and-return block is unchanged.)

### Part B — `src/memories/models/__init__.py`

Remove the `source_fact_ids` field from `Inference`:

```python
class Inference(BaseModel):
    id: int
    character_id: int
    statement: str
    derivation: str
    source_inference_ids: list[int] = []
    source_fact_paths: list[str] = []
    depth: int = 1
    inference_type: str = "logical"
    status: str = "active"
    created_at: datetime
```

**Pydantic silent-drop warning:** `Inference` uses the default model config (`extra="ignore"`).
After this field is removed, any test that still constructs `Inference(..., source_fact_ids=[…])`
does **not** raise — the kwarg is silently discarded. And any code/assertion that reads
`inf.source_fact_ids` now raises `AttributeError`. Both consequences are handled in the Test
Plan; the same silent-drop hazard applies to eager-pass mock payloads (see Part C).

### Part C — `src/memories/services/inference_service.py`

This is the substantive rewrite: eager pass, revalidation, and both cascades move from
integer fact ids + the flat `facts` table to schema paths + the blob.

#### C1. Imports

**Before** ([inference_service.py:8–18](src/memories/services/inference_service.py#L8-L18)):

```python
import aiosqlite

from memories.database import (
    create_inference,
    get_character,
    get_fact_rows,
    get_inferences,
    update_inference_status,
)
from memories.models import Character, Fact, Inference
from memories.services.ollama_client import OllamaClient
```

**After:**

```python
from typing import Any

import aiosqlite

from memories.database import (
    create_inference,
    get_character,
    get_facts,
    get_inferences,
    update_inference_status,
)
from memories.models import Character, Inference
from memories.schema_loader import _collect_leaves, load_schema
from memories.services.ollama_client import OllamaClient
```

(`get_fact_rows` → `get_facts`; drop the `Fact` model import; add `_collect_leaves` /
`load_schema` and `typing.Any`.)

#### C2. New helper — populated leaf values from the blob

Add near the top of the module (after `_coerce_id_list`, before `run_eager_pass`):

```python
def _coerce_str_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [v for v in raw if isinstance(v, str)]


def _populated_leaf_values(
    facts_blob: dict[str, Any],
) -> list[tuple[str, str | int | float | bool]]:
    """Return (dot.path, value) for every schema leaf that has a Value in the blob.

    Walks the schema's leaf paths and reads each leaf's stored Value out of the
    blob, skipping unset leaves. Path order follows schema declaration order.
    """
    result: list[tuple[str, str | int | float | bool]] = []
    for path, _leaf in _collect_leaves(load_schema()):
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

`_coerce_str_list` mirrors the existing `_coerce_id_list` and is used to parse
`source_fact_paths` out of eager-pass JSON. `_populated_leaf_values` is the shared blob→
`(path, value)` renderer used by both prompt builders. (The `Any` return-leaf convention
matches CLAUDE.md's `_set_leaf` note: read/write leaf helpers use
`str | int | float | bool | None`, never `Any`.)

#### C3. `build_eager_pass_prompt` — facts param becomes the blob; cite paths

**Before** ([inference_service.py:46–100](src/memories/services/inference_service.py#L46-L100)):
signature `build_eager_pass_prompt(character, facts: list[Fact], existing_inferences,
max_breadth, max_depth=…)`; renders `[{f.id}] {f.key}: {f.value}`; the JSON template asks
for `"source_fact_ids": [int, ...]`.

**After:**

- Signature: `facts` → `facts_blob: dict[str, Any]`.
- The "Current Facts" section renders path/value pairs from `_populated_leaf_values`:

  ```python
  lines.append("## Current Facts (path: value)")
  for path, value in _populated_leaf_values(facts_blob):
      lines.append(f"{path}: {value}")
  ```

- Update the "Rules" and JSON template: replace the `"source_fact_ids": [int, ...]` line
  with `"source_fact_paths": ["Character.Identity.Age", ...]`, and change "Cite source Facts
  and Inferences by id" to "Cite source Facts by their schema path and source Inferences by
  id." `source_inference_ids` stays as-is (integer ids).

#### C4. `run_eager_pass` — read blob, parse paths

**Before** ([inference_service.py:119–184](src/memories/services/inference_service.py#L119-L184)):
`facts: list[Fact]` param; per item `src_fact_ids = _coerce_id_list(item.get("source_fact_ids", []))`;
passes `source_fact_ids=src_fact_ids` to `create_inference`.

**After:**

- Signature: `facts` → `facts_blob: dict[str, Any]`; pass `facts_blob` to
  `build_eager_pass_prompt`.
- Per item:

  ```python
  src_paths = _coerce_str_list(item.get("source_fact_paths", []))
  src_inf_ids = _coerce_id_list(item.get("source_inference_ids", []))
  ```

- The same-pass cross-reference guard, depth cap, and breadth cap are unchanged (they use
  `src_inf_ids`).
- The `create_inference` call passes `source_fact_paths=src_paths` (drop the
  `source_fact_ids=` argument).

#### C5. `build_revalidation_prompt` — facts param becomes the blob; reference paths

**Before** ([inference_service.py:187–220](src/memories/services/inference_service.py#L187-L220)):
`facts: list[Fact]`; renders `[{f.id}] {f.key}: {f.value}`; prints
`f"Original sources: Facts {inference.source_fact_ids}, Inferences {inference.source_inference_ids}"`.

**After:**

- Signature: `facts` → `facts_blob: dict[str, Any]`; render the "Current Facts" block from
  `_populated_leaf_values(facts_blob)` (same `path: value` format as C3).
- Change the "Original sources" line to reference `inference.source_fact_paths`:

  ```python
  f"Original sources: Facts {inference.source_fact_paths}, "
  f"Inferences {inference.source_inference_ids}",
  ```

#### C6. `revalidate_single_inference` — facts param becomes the blob

**Before** ([inference_service.py:223–251](src/memories/services/inference_service.py#L223-L251)):
`facts: list[Fact]`.

**After:** signature `facts` → `facts_blob: dict[str, Any]`; pass it through to
`build_revalidation_prompt`. Body otherwise unchanged (the LLM call, parse, and
conservative fallback stay identical).

#### C7. `cascade_on_fact_edit` — path-keyed

**Before** ([inference_service.py:254–310](src/memories/services/inference_service.py#L254-L310)):

```python
async def cascade_on_fact_edit(
    db: aiosqlite.Connection,
    character_id: int,
    changed_fact_id: int,
    ollama: OllamaClient,
) -> list[Inference]:
    ...
    facts = await get_fact_rows(db, character_id)
    ...
    worklist = [inf for inf in non_invalidated if changed_fact_id in inf.source_fact_ids]
    ...
    holds = await revalidate_single_inference(inference, facts, remaining_active, ollama, model=model)
```

**After:**

```python
async def cascade_on_fact_edit(
    db: aiosqlite.Connection,
    character_id: int,
    changed_path: str,
    ollama: OllamaClient,
) -> list[Inference]:
    ...
    facts_blob = await get_facts(db, character_id)
    ...
    worklist = [inf for inf in non_invalidated if changed_path in inf.source_fact_paths]
    ...
    holds = await revalidate_single_inference(inference, facts_blob, remaining_active, ollama, model=model)
```

The BFS worklist, stale-propagation, and already-stale-seed logic (the Bug 6 fix) are
**unchanged** — only the seed predicate (`source_fact_paths` membership), the facts source
(`get_facts`), and the argument passed to `revalidate_single_inference` change.

#### C8. `cascade_on_fact_delete` — path-keyed

**Before** ([inference_service.py:313–338](src/memories/services/inference_service.py#L313-L338)):

```python
async def cascade_on_fact_delete(
    db: aiosqlite.Connection,
    character_id: int,
    deleted_fact_id: int,
) -> list[Inference]:
    active = await get_inferences(db, character_id, status="active")
    to_invalidate: set[int] = {inf.id for inf in active if deleted_fact_id in inf.source_fact_ids}
    ...
```

**After:** rename `deleted_fact_id: int` → `deleted_path: str`; change the seed comprehension
to `deleted_path in inf.source_fact_paths`. The transitive-expansion loop (which walks
`source_inference_ids`) is unchanged. No LLM call, as before.

### Part D — `src/memories/routers/inferences.py`

#### D1. Imports and dead helpers

**Before** ([inferences.py:11–24](src/memories/routers/inferences.py#L11-L24)) imports
`_parse_inference`, `_row`, `get_fact_rows`, plus `Fact` and `Literal`; defines
`_fact_exists` ([inferences.py:47–49](src/memories/routers/inferences.py#L47-L49)) against
the `facts` table.

**After:**

- Drop imports: `_parse_inference`, `_row`, `get_fact_rows`, `Fact` (models), `Literal`.
- Add imports: `get_facts` (from `memories.database`), `check_write_permitted` (from
  `memories.schema_loader`).
- Delete `_fact_exists`.

#### D2. `generate_inferences_endpoint` — read the blob

**Before** ([inferences.py:64–68](src/memories/routers/inferences.py#L64-L68)):

```python
    facts = await get_fact_rows(db, character_id)
    existing_inferences = await get_inferences(db, character_id, status="all")
    try:
        new_inferences = await run_eager_pass(db, character, facts, existing_inferences, ollama)
```

**After:**

```python
    facts_blob = await get_facts(db, character_id)
    existing_inferences = await get_inferences(db, character_id, status="all")
    try:
        new_inferences = await run_eager_pass(db, character, facts_blob, existing_inferences, ollama)
```

#### D3. `revalidate_inferences_endpoint` — body carries a path

**Before** ([inferences.py:32–33](src/memories/routers/inferences.py#L32-L33),
[77–94](src/memories/routers/inferences.py#L77-L94)):

```python
class _RevalidateBody(BaseModel):
    changed_fact_id: int
...
    if not await _fact_exists(db, body.changed_fact_id):
        raise HTTPException(status_code=404, detail="Fact not found")
    stale = await cascade_on_fact_edit(db, character_id, body.changed_fact_id, ollama)
```

**After:**

```python
class _RevalidateBody(BaseModel):
    changed_path: str
...
    try:
        check_write_permitted(body.changed_path, load_schema())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    stale = await cascade_on_fact_edit(db, character_id, body.changed_path, ollama)
```

(Import `load_schema` alongside `check_write_permitted`. An unknown or grouping path returns
422, consistent with `PUT /facts` in `facts.py`; a valid leaf path with no dependent
inferences returns 200 with an empty `stale_inferences` list.)

#### D4. Remove the promote endpoint

Delete `_PromoteBody` ([inferences.py:40–45](src/memories/routers/inferences.py#L40-L45))
and the entire `promote_inference_endpoint`
([inferences.py:138–215](src/memories/routers/inferences.py#L138-L215)). This is the only
place `aiosqlite.IntegrityError`, raw `facts` INSERT, and `_parse_inference`/`_row` were
used in this router, so the import cleanup in D1 is complete after this deletion.

### Part E — Remove `implication.py`

- **Delete** `src/memories/routers/implication.py` in full.
- **`src/memories/main.py`:** remove `implication` from the `from memories.routers import
  (...)` tuple ([main.py:25](src/memories/main.py#L25)) and delete the mount line
  ([main.py:84](src/memories/main.py#L84)):
  `app.include_router(implication.router, prefix="/api/sessions", tags=["implication"])`.

No other production module imports `implication` (grep confirms: the only other match is a
prose comment in `world_builder.py`). After removal, `create_fact`, `get_fact`,
`get_fact_by_category_key`, and `update_fact` in `database.py` have no production callers but
are retained (see "What This Step Does NOT Change").

---

## Transitional State After Step 9

- Inferences carry only `source_inference_ids` (integer, inference→inference) and
  `source_fact_paths` (schema paths). `source_fact_ids` is gone from the DB, the model, and
  all code.
- The eager pass and both cascade functions operate on schema paths and read facts from the
  `character_facts` blob. `POST /api/characters/{id}/inferences/generate` and
  `POST .../revalidate` (now `{"changed_path": …}`) are the user-triggered entry points.
- The promote endpoint and the entire `implication.py` router are removed. Their routes 404.
- Editing a fact through `PUT /api/characters/{id}/facts` still does not auto-cascade; a
  client that wants revalidation after an edit calls the `revalidate` endpoint with the
  edited path. (Wiring cascade directly into `PUT /facts` remains deliberately unshipped.)
- The legacy flat `facts` table, `Fact` model, and read/write helpers still exist. The
  session-end evaluator still reads flat facts (which are effectively always empty under
  Facts v2 — a pre-existing, accepted gap not addressed here). Fully retiring the flat-fact
  layer is a later cleanup (Step 8 CT-4).

---

## Test Plan

### `tests/integration/test_api_inference_promotion.py` — DELETE

The promote endpoint is removed; delete the whole file (28 tests). No replacement.

### `tests/integration/test_api_implication.py` — DELETE

`implication.py` is removed; every test targets a deleted endpoint. Delete the whole file.

### `tests/integration/test_api_extraction_resolution.py` — DELETE

Tests the Phase-6 extraction-resolution endpoints (`undo-user-fact`, `accept-implicit-fact`,
`ignore-implicit-fact`) that lived in `implication.py`. Delete the whole file.

### `tests/integration/test_inferences_repo.py` — changes

- **Delete** `test_source_fact_ids_stored_and_retrieved_as_list` — the field no longer
  exists (reading `inf.source_fact_ids` would `AttributeError`).
- **Update** `test_create_inference_stores_statement_and_derivation` — its
  `create_inference(..., source_fact_ids=[1])` call now raises `TypeError`; change the kwarg
  to `source_fact_paths=["Character.Identity.Age"]` (the assertions on statement/derivation
  are unaffected).
- **Keep** the three Step-3 `source_fact_paths` tests
  (`test_source_fact_paths_stored_and_retrieved_as_list`,
  `test_source_fact_paths_empty_list_when_not_set`,
  `test_source_fact_paths_persisted_on_read_back`) unchanged.

### `tests/unit/test_inference_service.py` — changes

**Module-level fixtures to update:**
- Replace the `_FACTS: list[Fact]` constant with a blob constant
  `_FACTS_BLOB = {"Character": {"Identity": {"Age": {"Value": 33}}, ...},
  "Setting": {"Temporal": {"Current-Year": {"Value": 2026}}}}` and drop the `Fact` import.
- In `_EXISTING_INFERENCES`, `_REVALIDATION_INFERENCE`, and `_OTHER_INFERENCE`, replace
  `source_fact_ids=[…]` with `source_fact_paths=[…]` (using real schema paths, e.g.
  `["Character.Identity.Age", "Setting.Temporal.Current-Year"]`).
- In `_EAGER_PASS_ITEM`, replace the `"source_fact_ids": [1, 2]` JSON key with
  `"source_fact_paths": ["Character.Identity.Age", "Setting.Temporal.Current-Year"]`.

**`build_eager_pass_prompt` tests to update** (pass the blob; assert on paths):
- `test_eager_pass_prompt_includes_all_facts` → assert
  `"Character.Identity.Age: 33"` (or path + value substrings) appears.
- `test_eager_pass_prompt_includes_character_name`,
  `test_eager_pass_prompt_lists_existing_inferences`,
  `test_eager_pass_prompt_no_existing_inferences_uses_fallback`,
  `test_eager_pass_prompt_instructs_max_breadth` — change the `facts` argument to
  `_FACTS_BLOB`; assertions otherwise hold.
- **Add** `test_eager_pass_prompt_requests_source_fact_paths` — assert the rendered JSON
  template contains `"source_fact_paths"` and not `"source_fact_ids"`.

**`run_eager_pass` tests to update** (drop the `fact` fixture dependency; pass a blob;
eager-pass items cite `source_fact_paths`):
- `test_eager_pass_parses_returned_inferences`,
  `test_eager_pass_stores_inferences_to_db`,
  `test_eager_pass_returns_empty_list_on_empty_json_array`,
  `test_eager_pass_applies_breadth_cap`,
  `test_eager_pass_rejects_same_pass_cross_reference`,
  `test_eager_pass_computes_depth_one_for_fact_only_source`,
  `test_eager_pass_computes_depth_from_source_inference`,
  `test_eager_pass_discards_inference_exceeding_max_depth`,
  `test_eager_pass_raises_on_non_json_response`,
  `test_eager_pass_request_sends_format_json`,
  `test_eager_pass_request_sends_think_false` — call `run_eager_pass(db, character,
  _FACTS_BLOB, …)` instead of `[fact]`; where items previously used
  `source_fact_ids=[fact.id]`, use `source_fact_paths=["Character.Identity.Age"]`.
- **Add** `test_eager_pass_persists_source_fact_paths` — item cites
  `source_fact_paths=["Character.Identity.Age"]`; assert the stored inference's
  `source_fact_paths == ["Character.Identity.Age"]`.
- **Add** `test_eager_pass_ignores_legacy_source_fact_ids_key` — an item carrying only the
  old `"source_fact_ids": [1]` key (no `source_fact_paths`) is stored with
  `source_fact_paths == []` (documents the silent-drop of the legacy key).

**`build_revalidation_prompt` tests to update** (pass the blob; source paths):
- `test_revalidation_prompt_includes_inference_statement`,
  `test_revalidation_prompt_includes_inference_derivation`,
  `test_revalidation_prompt_includes_current_facts`,
  `test_revalidation_prompt_includes_other_active_inferences` — pass `_FACTS_BLOB`; the
  current-facts assertions check for path substrings (e.g. `"Character.Identity.Age"` and
  `"33"`).

**`revalidate_single_inference` tests to update** (pass the blob):
- `test_revalidate_returns_true_when_inference_holds`,
  `test_revalidate_returns_false_when_inference_does_not_hold`,
  `test_revalidate_defaults_to_true_on_parse_error` — call with `_FACTS_BLOB`.

**`cascade_on_fact_edit` tests to update** (create inferences with `source_fact_paths`; call
with a path; raw-SQL inserts switch column):
- `test_cascade_edit_marks_directly_dependent_inference_stale`,
  `test_cascade_edit_returns_stale_inferences`,
  `test_cascade_edit_does_not_mark_if_revalidation_returns_true` — create the inference with
  `source_fact_paths=["Character.Identity.Occupation"]`; call
  `cascade_on_fact_edit(db, character.id, "Character.Identity.Occupation", ollama)`.
- `test_cascade_edit_leaves_unrelated_inference_active` — unrelated inference uses
  `source_fact_paths=["Setting.Location.Name"]`; cascade on a different path.
- `test_cascade_edit_transitively_marks_chained_inference_stale` — `inf_a` uses
  `source_fact_paths=["Character.Identity.Occupation"]`, `inf_b` uses
  `source_inference_ids=[inf_a.id]`; cascade on the occupation path.
- `test_cascade_edit_skips_already_stale_inferences`,
  `test_cascade_edit_propagates_through_stale_intermediary` — the raw
  `INSERT INTO inferences (…, source_fact_ids, source_inference_ids, …)` statements change to
  `(…, source_fact_paths, source_inference_ids, …)` with a JSON path array
  (`json.dumps(["Character.Identity.Occupation"])`); cascade on that path.

**`cascade_on_fact_delete` tests to update** (paths; call with a path string):
- `test_cascade_delete_marks_directly_dependent_inference_invalidated`,
  `test_cascade_delete_returns_all_invalidated_inferences`,
  `test_cascade_delete_no_llm_call_made` — create inferences with
  `source_fact_paths=["Character.Identity.Occupation"]`; call
  `cascade_on_fact_delete(db, character.id, "Character.Identity.Occupation")`.
- `test_cascade_delete_leaves_unrelated_inference_active` — unrelated uses a different path.
- `test_cascade_delete_transitively_marks_chained_inference_invalidated` — chained via
  `source_inference_ids`.

**`compute_depth` tests** — unchanged (they never referenced fact sources).

### `tests/integration/test_api_inference_generation.py` — changes

**Generate endpoint tests to update** (items cite `source_fact_paths`; the `fact` fixture is
no longer meaningful — the endpoint reads the blob, which is empty unless seeded, but the
LLM is mocked so blob contents do not affect what is stored):
- `test_generate_inferences_returns_200`,
  `test_generate_inferences_returns_new_inferences_list`,
  `test_generate_inferences_empty_response_returns_empty_list`,
  `test_generate_inferences_on_parse_error_returns_warning`,
  `test_generate_inferences_unknown_character_returns_404` — change `_DEFAULT_EAGER_ITEM`'s
  `"source_fact_ids": []` key to `"source_fact_paths": []`.
- `test_generate_inferences_stores_to_db`,
  `test_generate_inferences_applies_breadth_cap` — items use
  `source_fact_paths=["Character.Identity.Age"]` in place of `source_fact_ids=[fact.id]`.
- `test_generate_inferences_respects_depth_cap` — unchanged apart from the item's key rename
  (it exercises `source_inference_ids`).

**Revalidate endpoint tests to update** (body `{"changed_path": …}`; inferences cite paths):
- `test_revalidate_returns_200`,
  `test_revalidate_returns_stale_inferences`,
  `test_revalidate_marks_stale_in_db` — create the inference with
  `source_fact_paths=["Character.Identity.Occupation"]`; POST
  `{"changed_path": "Character.Identity.Occupation"}`.
- `test_revalidate_does_not_affect_unrelated_inferences` — unrelated uses a different path.
- `test_revalidate_unknown_character_returns_404` — body becomes
  `{"changed_path": "Character.Identity.Occupation"}`; still 404 (character check runs first).
- **Rename/replace** `test_revalidate_unknown_fact_returns_404` →
  `test_revalidate_unknown_path_returns_422` — POST `{"changed_path": "Nope.Not.Here"}` to a
  valid character → 422 (path not in schema).

**Delete/patch management tests** (`test_delete_inference_*`, `test_patch_inference_*`) —
unchanged.

### `tests/unit/test_prompt_builder.py` — changes

Lines 142, 166, 178, 190, 326 construct `Inference(..., source_fact_ids=[…])`. These are
bare model constructions (not `create_inference`), so the kwarg is silently dropped rather
than erroring, but update each to `source_fact_paths=[…]` for correctness (the surrounding
assertions do not read the field). No test is added or deleted.

### `tests/unit/test_evaluator_service.py` — changes

The module-level inference constant at line ~103–107 constructs `Inference(...,
source_fact_ids=[1])`; change to `source_fact_paths=["Character.Identity.Age"]`. No
behavioural change to the evaluator tests.

### `tests/unit/test_experience_service.py` — changes

The inference constructed at line ~48–52 uses `source_fact_ids=[1]`; change to
`source_fact_paths=["Character.Identity.Occupation"]`.

### `tests/integration/test_api_facts.py` — changes

The `create_inference(db, …, source_fact_ids=[])` setup call at line ~146–150 now raises
`TypeError`; change the kwarg to `source_fact_paths=[]` (or drop it — it defaults to `[]`).

---

## Edge Cases

- **Pydantic silent-drop of the legacy `source_fact_ids` key (highest-risk pitfall).**
  Removing the model field means eager-pass JSON items that still carry `"source_fact_ids"`
  are parsed with `source_fact_paths == []` and **no error** — the inference is stored with
  no fact provenance. Every eager-pass mock item and every `Inference(...)` /
  `create_inference(...)` call in the suite must switch the key/kwarg to `source_fact_paths`.
  `create_inference` calls fail loudly (`TypeError`) and are easy to catch; bare
  `Inference(...)` constructions and raw JSON items fail **silently** and are the real hazard
  — the Test Plan enumerates each. `test_eager_pass_ignores_legacy_source_fact_ids_key`
  pins this behaviour so it is intentional, not accidental.
- **`revalidate` with a valid path but no dependent inferences.** Returns 200 with
  `stale_inferences: []`. This differs from the old endpoint, which 404'd when the integer
  `changed_fact_id` did not exist in the `facts` table. Under the blob model there is no
  "fact existence" to check — any schema leaf may be unset — so the only validation is that
  the path is a real schema leaf (422 otherwise).
- **`revalidate` path validation returns a single message.** `check_write_permitted` raises
  one `ValueError` naming the offending path; the 422 body carries that one message. This is
  a plain validation error, not an enum constraint list, so the enum-loop risk
  (`project_enum_validation_prompt_risk`) does not apply here.
- **Cascade seed matches by exact path string.** `cascade_on_fact_edit`/`_delete` match
  `changed_path in inf.source_fact_paths` by exact equality. There is no prefix/parent
  matching — editing `Character.Identity.Age` does not cascade inferences sourced from
  `Character.Identity` (a grouping, which can never be a source path anyway). This mirrors
  the old exact-id matching and is intended.
- **Blob read in the cascade.** `cascade_on_fact_edit` now calls `get_facts` (schema-masked
  blob) to build the revalidation prompt. If the blob is empty the revalidation prompt lists
  no current facts; the LLM decision is mocked in tests, so this does not affect the suite,
  and in production an empty blob simply yields a sparse prompt.
- **No SSE endpoints touched — ASGITransport limitation not applicable.** `generate` and
  `revalidate` are plain request/response POSTs; there is no streaming and no concurrent-
  request-during-stream scenario, so the `ASGITransport` buffering hazard
  (`project_asgi_transport_streaming`) does not arise in this step's tests.
- **Frontend `apiRevalidateInferences` body shape.** If `chat.js` retains an
  `apiRevalidateInferences` helper that posts `{changed_fact_id}`, it is now stale — but
  Step 8 wired no UI control to it (the inference panel exposes only delete), so no live UI
  path breaks. Updating that helper's body to `{changed_path}` is a frontend follow-up, not
  required for this backend step; flag it if a revalidate control is added later.

---

## Post-Implementation Cleanup Tasks
