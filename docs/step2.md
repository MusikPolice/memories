# Step 2 — DB: `character_facts` Table, Updated `decisions` Table, and Repositories

## Why this step exists

Step 3 (prompt changes) needs to render the full fact tree by merging the schema with a
character's stored values. Step 4 (World Builder) needs to write facts to a blob.
Step 5 (Character Evaluator tool loop) needs to read and patch individual paths in that
blob. All three need stable repository functions to call.

None of that can be built until the `character_facts` table exists and its repository
functions — `get_facts`, `set_facts`, `patch_fact` — are wired up and tested. Similarly,
the `decisions` table must be restructured before Step 5 can log tool-call events, and the
`segments` table must be removed before any service code can be simplified (its phantom
presence makes `store_message` require a `segment_id` that nothing meaningful provides).

This step makes all those structural changes in one place: the DDL in `database.py`, the
Pydantic models in `models/__init__.py`, and the repository functions that bridge the two.
It also includes the mechanical updates to every existing caller that is broken by the
signature changes — `chat_service.py`, `implication.py`, and the integration conftest.

**Success criterion:** all tests in `tests/integration/test_db_init.py`,
`tests/integration/test_character_facts_repo.py`, `tests/integration/test_decisions_repo.py`,
`tests/integration/test_messages_repo.py`, and `tests/integration/test_sessions_repo.py`
pass. All existing tests continue to pass. The dev server starts cleanly against a fresh DB.

---

## What this step does NOT change

- The old `facts` table — it remains in the DDL and its row-based CRUD functions
  (`create_fact`, `get_fact_rows`, `get_fact`, `update_fact`, `delete_fact`,
  `get_fact_by_category_key`) remain in `database.py` to keep the existing `facts.py`
  router working. The old `facts.py` router is not touched in this step.
- Evaluator logic — `store_decision` callers in `chat_service.py` and `implication.py`
  receive temporary stub arguments (see Part D) to restore compilability; the real
  tool-call-based decision logging is Step 5's concern.
- Any prompt, SSE, or service logic beyond what is strictly required to resolve the
  removed columns and table.
- The frontend.

---

## Part A — DDL changes in `database.py`

The `_DDL` constant in `database.py` is rewritten. All changes take effect on the next
fresh database initialisation; no migration script is required. The project convention is
that `memories.db` is wiped when the DDL changes incompatibly, so no backward-compatible
migration code is added.

### `character_facts` table (new)

One row per character. The `facts_json` column stores the blob described in
`docs/plan-v2.md` — a sparse nested JSON object where populated leaves carry a `Value`
key. The column defaults to `'{}'` so a character with no facts set yet has a valid empty
blob. `updated_at` is overwritten on every write.

### `decisions` table (replaced)

The three existing columns — `reasoning TEXT NOT NULL`, `verdict TEXT NOT NULL`,
`violations TEXT` — are replaced by four new columns:

- `pass_name TEXT NOT NULL` — identifies which pass produced the decision
  (`"world_builder"`, `"character_llm"`, or `"character_evaluator"`)
- `tool_name TEXT NOT NULL` — identifies the tool that was called
  (`"author_set_facts"`, `"set_fact"`, `"propose_inference"`, `"report_pass"`,
  `"report_contradiction"`, etc.)
- `tool_args TEXT NOT NULL` — the arguments passed to the tool, serialised as a JSON
  object; stored as TEXT and parsed on read
- `user_input TEXT` — the user's decision when the tool required approval (mutable fact
  update, immutable-unset card); `NULL` when no user input was required; stored as TEXT
  and parsed on read when non-null

The `id`, `character_id`, `session_id`, `turn_id`, and `created_at` columns are
unchanged. The index on `(session_id, turn_id)` is retained.

### `messages` table (columns removed)

Three columns are dropped from the `messages` DDL:

- `segment_id INTEGER REFERENCES segments(id)` — segments no longer exist
- `captured_by TEXT` — deferred Phase 7a annotation; premature dead write
- `ungrounded_implications TEXT` — replaced by the tool-call approach (Step 5)

The table continues to hold `id`, `character_id`, `session_id`, `role`, `content`,
`turn_id`, and `created_at`. The index on `(session_id, turn_id)` is retained.

### `segments` table (removed)

The entire `segments` table DDL is deleted. The index `idx_segments_session` is also
removed. Segments will be rebuilt in Phase 7b when context compression is implemented.

### `facts` table (unchanged)

The legacy `facts` table DDL remains in `_DDL` so that:
- The old `facts.py` router can still read and write rows
- `init_db()` remains idempotent on any existing DB that has the table

Its CRUD functions are renamed (see Part C) but the table definition is not altered.

---

## Part B — Model changes in `models/__init__.py`

### `Segment` model (removed)

`Segment` is deleted from `models/__init__.py`. All imports of `Segment` (currently only
in `database.py`) are removed. No other file imports `Segment`.

### `Decision` model (updated)

The three old fields (`reasoning: str`, `verdict: str`, `violations: list[dict] | None`)
are replaced by the four new fields mirroring the new DDL columns:

- `pass_name: str`
- `tool_name: str`
- `tool_args: dict[str, Any]` — always a dict after parsing; an empty dict `{}` is valid
- `user_input: dict[str, Any] | None` — `None` when no user input was captured

### `Message` model (updated)

Three fields are removed from the Pydantic model to match the new DDL:

- `segment_id: int`
- `captured_by: list[str] | None`
- `ungrounded_implications: list[dict[str, Any]] | None`

After this change `Message` holds only `id`, `character_id`, `session_id`, `role`,
`content`, `turn_id`, and `created_at`.

---

## Part C — `database.py` function changes

### Renamed: `get_facts` → `get_fact_rows`

The existing `get_facts(db, character_id) -> list[Fact]` function is renamed to
`get_fact_rows` to free the name `get_facts` for the new blob-based function described
below. The old `Fact` model and the function's behaviour are unchanged; only the name
differs. All imports of `get_facts` in `chat_service.py` are updated to `get_fact_rows`.

The old `patch_fact(db, *, fact_id, ...)` function in `database.py` retains its name and
signature — it operates on integer `fact_id` values from the old `facts` table, which is
a completely different signature from the new `patch_fact` described below. The two are
unambiguous in practice; no rename is needed.

### Updated: `create_session`

The INSERT into `segments` (which created the initial `"session_start"` segment) is
removed. `create_session` now only inserts a row into `sessions` and returns the resulting
`Session` object. The `Segment` import and `get_active_segment` import are removed from
the module.

### Updated: `store_message`

The `segment_id: int` parameter and the `ungrounded_implications: list[dict] | None`
parameter are both removed. The INSERT SQL is updated to write only the columns that
remain in the `messages` table. The `_parse_message` helper no longer needs to JSON-parse
`captured_by` or `ungrounded_implications`; that parsing code is deleted. The function
return type remains `Message`.

### Updated: `replace_message_content`

The SQL currently sets `ungrounded_implications = NULL` alongside `content = ?`. That
clause is removed. The update now only sets `content`. No behavioural change beyond the
removed column reference.

### Updated: `store_decision`

Old keyword arguments `reasoning: str`, `verdict: str`, `violations: list[dict] | None`
are replaced by `pass_name: str`, `tool_name: str`, `tool_args: dict[str, Any]`,
`user_input: dict[str, Any] | None = None`.

`tool_args` is serialised to JSON before storage; `user_input` is serialised to JSON when
non-None and stored as `NULL` otherwise.

The `_parse_decision` helper is updated to JSON-parse `tool_args` (always non-null) and
`user_input` (null-checked) on read, producing a `Decision` with the new field shapes.

The `get_decisions` function signature and query are unchanged.

### Removed: `get_active_segment`

The `get_active_segment(db, session_id) -> Segment` function is deleted. It is called
from `chat_service.py`; that call is removed in Part D.

### New: `get_facts(db, character_id) -> dict[str, Any]`

Reads the `character_facts` row for `character_id`. If no row exists, returns an empty
dict `{}`. If a row exists, JSON-parses `facts_json` and passes the result through
`schema_loader.apply_mask()` before returning. The masking ensures stale paths (those
removed from the schema since the blob was written) are silently dropped before the caller
sees them.

This function never raises `NotFoundError`; a missing character row and a character with
no facts set both return `{}`.

### New: `set_facts(db, character_id, blob) -> None`

Writes `blob` as the `facts_json` for `character_id` using an `INSERT OR REPLACE`
statement. `blob` must be a plain Python dict; the function serialises it to JSON before
writing. `updated_at` is set to the current timestamp. After the write, `db.commit()` is
called.

### New: `patch_fact(db, character_id, path_tuple, value) -> None`

`path_tuple` is a sequence of string keys representing the dot-notation path split on
`"."` — for example `("Character", "Identity", "Name")`. The function:

1. Calls `get_facts(db, character_id)` to load the current masked blob (an empty dict
   for a character with no facts yet).
2. Navigates to the parent of the target leaf, creating intermediate grouping dicts as
   needed.
3. Sets the leaf to `{"Value": value}`.
4. Calls `set_facts(db, character_id, updated_blob)` to write back.

`value` must already be validated by the caller (schema type and constraint checks happen
in the `set_fact` / `author_set_facts` handlers, not in this repository function). This
function does no schema validation — it is a pure structural write.

---

## Part D — Mechanical caller updates

These updates are not logic changes; they are the minimum edits required to restore
compilability and test passage after the interface changes in Parts A–C.

### `src/memories/services/chat_service.py`

The import of `get_active_segment` is removed. The import of `get_facts` is replaced with
`get_fact_rows`. The `asyncio.gather` call that fetches `get_active_segment(db, session_id)`
is updated to omit that call; the resulting `segment` variable is removed from its
assignment target. The two `store_message` calls that pass `segment_id=segment.id` have
that argument removed. The block that computes `ungrounded` and passes
`ungrounded_implications=ungrounded` to `store_message` is deleted entirely.

The two `store_decision` calls are updated with temporary stub arguments:

- `pass_name="character_evaluator"` (the only caller that currently exists)
- `tool_name="evaluator_verdict"` — a placeholder that makes the row meaningful as a
  debugging record while Step 5 has not yet introduced per-tool-call logging
- `tool_args={"verdict": eval_result.verdict}` — captures the most useful datum from the
  old schema; the evaluator result's verdict is stored as the sole arg so the decisions
  log remains readable during the transition period
- `user_input=None`

These stubs are explicitly temporary. Step 5 will replace all `store_decision` calls in
`chat_service.py` with per-tool-call logging once the tool-call loop is in place.

### `src/memories/routers/implication.py`

The two `store_decision` calls in this router receive the same stub treatment as above.
Because `implication.py` will be removed in Step 7 (when the old `implication` verdict
path is deleted), no further investment in its decision-logging fidelity is warranted.

### `tests/integration/conftest.py`

The `fact` fixture calls `create_fact(db, ...)`, which writes to the old `facts` table.
That fixture is unchanged — it still works because the old table and `create_fact`
function are retained. The `session` fixture calls `create_session`, which no longer
inserts a segment row; no change to the fixture is needed.

---

## Tests

### Updated: `tests/integration/test_db_init.py`

**`test_all_tables_created`** — The `_EXPECTED_TABLES` set is updated to remove
`"segments"` and add `"character_facts"`. All other expected tables remain.

**`test_segments_table_does_not_exist`** — New test. Queries `sqlite_master` for a table
named `"segments"`. Asserts it is not present.

**`test_character_facts_table_exists`** — New test. Runs `PRAGMA table_info(character_facts)`.
Asserts the result has at least one row.

**`test_character_facts_table_columns`** — New test. Asserts `character_id`, `facts_json`,
and `updated_at` are all present in `PRAGMA table_info(character_facts)`.

**`test_character_facts_json_default_is_empty_object`** — New test. Inserts a character,
then inserts a `character_facts` row without explicitly providing `facts_json`. Reads it
back. Asserts `facts_json` is `"{}"`.

**`test_decisions_table_has_pass_name_column`** — New test. `PRAGMA table_info(decisions)`
includes `pass_name`.

**`test_decisions_table_has_tool_name_column`** — New test. `PRAGMA table_info(decisions)`
includes `tool_name`.

**`test_decisions_table_has_tool_args_column`** — New test. `PRAGMA table_info(decisions)`
includes `tool_args`.

**`test_decisions_table_has_user_input_column`** — New test. `PRAGMA table_info(decisions)`
includes `user_input`.

**`test_decisions_table_no_reasoning_column`** — New test. `PRAGMA table_info(decisions)`
does NOT include `reasoning`.

**`test_decisions_table_no_verdict_column`** — New test. `PRAGMA table_info(decisions)`
does NOT include `verdict`.

**`test_decisions_table_no_violations_column`** — New test. `PRAGMA table_info(decisions)`
does NOT include `violations`.

**`test_messages_table_no_segment_id_column`** — New test. `PRAGMA table_info(messages)`
does NOT include `segment_id`.

**`test_messages_table_no_captured_by_column`** — New test. `PRAGMA table_info(messages)`
does NOT include `captured_by`.

**`test_messages_table_no_ungrounded_implications_column`** — New test.
`PRAGMA table_info(messages)` does NOT include `ungrounded_implications`.

All existing tests in this file that check the old `segments` table or the old
`decisions` columns are removed.

---

### New: `tests/integration/test_character_facts_repo.py`

These tests use the `db` fixture (in-memory SQLite) and the `character` fixture. They
call only the repository functions; no HTTP client is involved.

---

**`test_get_facts_returns_empty_dict_for_new_character`**

Calls `get_facts(db, character.id)` on a character that has never had any facts written.
Asserts the result is `{}`.

---

**`test_get_facts_returns_empty_dict_for_unknown_character_id`**

Calls `get_facts(db, 999999)` with an ID that has no character row. Asserts the result
is `{}`. This verifies the function does not raise on a missing row.

---

**`test_set_facts_then_get_facts_round_trips`**

Calls `set_facts(db, character.id, {"Character": {"Identity": {"Name": {"Value": "Sarah"}}}})`.
Calls `get_facts(db, character.id)`. Asserts the returned dict contains
`Character.Identity.Name.Value == "Sarah"`. (Path in the schema ensures the mask passes
it through.)

---

**`test_set_facts_replaces_existing_blob`**

Calls `set_facts` twice with different blobs. After the second call, `get_facts` returns
only the second blob's contents — no merging occurs. Asserts the first blob's content is
absent.

---

**`test_get_facts_applies_schema_mask`**

Writes a blob that includes a path not in the schema (e.g. `"LegacyCategory"` at the top
level). Calls `get_facts`. Asserts the unknown key is absent from the result. This
verifies that `apply_mask` is called on every read and not just on explicit mask calls.

---

**`test_get_facts_keeps_valid_paths_after_masking`**

Writes a blob with both a valid schema path and an invalid path. Calls `get_facts`.
Asserts the valid path is present and the invalid path is absent in the same result.

---

**`test_patch_fact_updates_single_nested_key`**

Calls `set_facts` with a blob containing two leaf values at different paths. Calls
`patch_fact(db, character.id, ("Character", "Identity", "Name"), "Elena")`. Calls
`get_facts`. Asserts the patched leaf has the new value. Asserts the other leaf is
unchanged — `patch_fact` is non-destructive to other paths.

---

**`test_patch_fact_on_character_with_no_row_creates_row`**

Calls `patch_fact` on a character that has never had `set_facts` called. Asserts `get_facts`
returns a dict containing the single patched leaf. Verifies that `patch_fact` initialises
the blob rather than raising.

---

**`test_patch_fact_creates_intermediate_groupings`**

Writes an empty blob (`set_facts(db, character.id, {})`). Calls `patch_fact` with a
deeply nested path. Asserts `get_facts` returns the full nested structure with the leaf
set. Verifies that intermediate dict keys are created rather than raising on missing nodes.

---

**`test_set_facts_multiple_characters_independent`**

Creates two characters. Calls `set_facts` on each with different blobs. Asserts
`get_facts` for each character returns only that character's blob. Verifies there is no
cross-character contamination.

---

**`test_get_facts_returns_updated_at_is_not_exposed`**

This is a negative test. `get_facts` returns a plain dict; it must not return the
database row's `updated_at` timestamp (which belongs to the storage layer, not the
application layer). Asserts the returned dict does not contain `"updated_at"` as a
top-level key.

---

### Updated: `tests/integration/test_decisions_repo.py`

Existing tests that use the old `store_decision` signature (with `reasoning`, `verdict`,
`violations`) are rewritten to use the new signature. Tests that assert on old field
names in the returned `Decision` object are updated to assert on the new field names.

---

**`test_store_decision_with_tool_call_data`**

Calls `store_decision` with `pass_name="character_evaluator"`, `tool_name="report_pass"`,
`tool_args={}`, `user_input=None`. Asserts the returned `Decision` has `pass_name`,
`tool_name`, and `tool_args` equal to the supplied values. Asserts `user_input` is
`None`.

---

**`test_store_decision_tool_args_round_trips_as_dict`**

Calls `store_decision` with `tool_args={"path": "Character.Identity.Name", "value": "Sarah"}`.
Reads back via `get_decisions`. Asserts `tool_args` is a `dict`, not a raw JSON string.
Asserts the keys and values are intact.

---

**`test_store_decision_with_user_input`**

Calls `store_decision` with `user_input={"action": "accept", "value": "Sarah"}`.
Reads back. Asserts `user_input` is a `dict` with the expected keys.

---

**`test_store_decision_user_input_null`**

Calls `store_decision` with `user_input=None`. Reads back. Asserts `user_input` is
`None`, not an empty dict or a string `"null"`.

---

**`test_get_decisions_returns_in_reverse_chronological_order`**

Stores two decisions for the same session with different `turn_id` values. Calls
`get_decisions`. Asserts the one with the higher `turn_id` appears first. This verifies
the ordering is unchanged from the old schema.

---

**`test_decision_has_no_reasoning_field`**

Asserts that the `Decision` Pydantic model returned by `store_decision` does not have a
`reasoning` attribute. Specifically, `hasattr(decision, "reasoning")` must be `False`.

---

**`test_decision_has_no_verdict_field`**

Same for `verdict`.

---

### Updated: `tests/integration/test_messages_repo.py`

Existing tests that pass `segment_id` to `store_message` are updated to omit that
argument. Tests that pass `ungrounded_implications` are updated to omit that argument.
Tests that assert on `message.segment_id`, `message.captured_by`, or
`message.ungrounded_implications` are removed or replaced.

---

**`test_store_message_does_not_require_segment_id`**

Calls `store_message` without a `segment_id` argument. Asserts the returned `Message`
stores correctly and can be retrieved by `get_messages`. This verifies the parameter is
gone (not merely optional with a default).

---

**`test_stored_message_has_no_segment_id_attribute`**

Asserts `hasattr(message, "segment_id")` is `False` on a freshly stored message.

---

**`test_stored_message_has_no_ungrounded_implications_attribute`**

Asserts `hasattr(message, "ungrounded_implications")` is `False`.

---

**`test_stored_message_has_no_captured_by_attribute`**

Asserts `hasattr(message, "captured_by")` is `False`.

---

**`test_replace_message_content_updates_content`**

Stores a user message and an assistant message for the same `(session_id, turn_id)`.
Calls `replace_message_content` with a new content string. Asserts `get_messages` returns
the updated content for the assistant message. This is a regression guard to confirm
`replace_message_content` still functions after the column removal.

---

### Updated: `tests/integration/test_sessions_repo.py`

**`test_create_session_does_not_create_segment`**

Calls `create_session`. Queries `sqlite_master` to confirm the `segments` table does not
exist, or (if the table is somehow present) queries `SELECT COUNT(*) FROM segments` and
asserts zero rows. The preferred assertion is the first (table absent), since the DDL no
longer includes it.

The existing `test_create_session_creates_opening_segment` test (if one exists) is
removed.

---

## Files changed by this step

| Action | File | Notes |
|---|---|---|
| Modify | `src/memories/database.py` | DDL rewrite; rename `get_facts` → `get_fact_rows`; remove `get_active_segment`; update `create_session`, `store_message`, `_parse_message`, `replace_message_content`, `store_decision`, `_parse_decision`; add `get_facts`, `set_facts`, `patch_fact` |
| Modify | `src/memories/models/__init__.py` | Remove `Segment`; update `Decision` and `Message` |
| Modify | `src/memories/services/chat_service.py` | Remove `get_active_segment` call; `get_facts` → `get_fact_rows`; remove `segment_id` and `ungrounded_implications` from `store_message` calls; stub `store_decision` with new signature |
| Modify | `src/memories/routers/implication.py` | Stub `store_decision` with new signature |
| Modify | `tests/integration/test_db_init.py` | Update expected tables; add new column/table assertions; remove old assertions |
| Add | `tests/integration/test_character_facts_repo.py` | All repository tests for the new blob layer |
| Modify | `tests/integration/test_decisions_repo.py` | Update to new `Decision` shape and `store_decision` signature |
| Modify | `tests/integration/test_messages_repo.py` | Remove `segment_id`, `captured_by`, `ungrounded_implications` references |
| Modify | `tests/integration/test_sessions_repo.py` | Assert no segment created |

No changes to `schema_loader.py` (already exists from Step 1).
No changes to any router except `implication.py` (mechanical stub only).
No changes to the frontend.
No changes to any other service file.

---

## Dependency order relative to other steps

Step 2 requires Step 1 to be complete, because `get_facts` calls `schema_loader.apply_mask`
and `schema_loader` must exist.

Step 2 must be complete before:
- **Step 3** (prompt changes) — `build_system_prompt()` will call `get_facts` to get the
  blob; the evaluator prompt builder will also need the blob values to render populated
  leaves
- **Step 4** (World Builder) — `author_set_facts` handler calls `patch_fact` and
  `set_facts`
- **Step 5** (Character Evaluator tool loop) — `set_fact` handler calls `patch_fact`;
  `store_decision` is called with real per-tool-call arguments

Steps 2 and 0b are independent and can proceed in parallel.
