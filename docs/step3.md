# Step 3 — Prompt Changes and Evaluator Model Cleanup

## Overview

Step 3 is a refactoring pass. It wires the schema-constrained fact blob (delivered by Step 2) into the prompts that the Character LLM and Character Evaluator receive, strips the evaluator model of the two verdicts being retired (`implication` and `experience_update`), and removes the data structures that served those verdicts. No new features are added; no LLM calls are restructured. The goal is a codebase that speaks the v2 vocabulary at every prompt boundary, so that Steps 4 and 5 can add the World Builder and Evaluator tool-call loops on top of a consistent foundation.

When Step 3 is complete, the system is in a transitional state: the character system prompt renders the full schema tree with live values; the evaluator has a narrower vocabulary; but the tool-call loop that eventually replaces the JSON evaluator verdict is not wired yet (Step 5). The system continues to generate responses and evaluate them — it just uses a richer fact representation and fewer verdict types.

---

## What Steps 1 and 2 Delivered

**Step 1** produced:
- `src/memories/fact_schema.json` — the canonical, versioned list of every valid fact path with `Type`, `Mutability`, `Description`, and (for Enum leaves) `Constraint`.
- `src/memories/schema_loader.py` — functions for loading, masking, and rendering the schema: `load_schema()`, `apply_mask()`, `check_write_permitted()`, `render_schema_for_prompt()`.
- `GET /api/schema` — endpoint that returns the schema JSON verbatim.

**Step 2** produced:
- `character_facts` table — one row per character; a single `facts_json` TEXT column holding the fact blob.
- `get_facts(db, character_id) → dict[str, Any]` — reads and schema-masks the blob; returns `{}` if no row exists.
- `set_facts(db, character_id, blob)` — full blob write.
- `patch_fact(db, character_id, path_tuple, value)` — updates a single leaf in place.
- Updated `decisions` table — `pass_name`, `tool_name`, `tool_args`, `user_input` columns.

Neither step touched the prompt-building functions or the evaluator, which still use the old `list[Fact]` API.

---

## Scope of Step 3

Five distinct things happen in this step, in dependency order:

1. **`build_system_prompt()` overhaul** — signature changes from `list[Fact]` to `dict[str, Any]` (the blob); the function renders the full schema tree, merging schema metadata with blob values.
2. **`build_evaluator_prompt()` updates** — adds the `## Fact Schema` section; replaces the old flat fact list with a schema-path representation; removes `implication` and `experience_update` from the vocabulary.
3. **Evaluator model cleanup** — removes `ExperienceUpdate`, `experience_updates`, `Violation.suggested_fact`, `_violation_duplicates_existing_fact()`, and narrows `_VALID_VERDICTS` to four entries.
4. **`chat_service.py` dead-code removal** — the `experience_update` handling block accesses `eval_result.experience_updates`, a field that no longer exists; it must be removed here to keep the code runnable.
5. **Signature cascade** — `run_evaluator()`, `run_contradiction_loop()`, and `run_turn()` all take `list[Fact]` today; they are updated to take `dict[str, Any]`, and `run_turn()` switches its fact-loading call from `get_fact_rows()` to `get_facts()`.

The `extraction_service.py` (predecessor to the World Builder) is left untouched. Step 4 replaces it wholesale. Within Step 3, `run_turn()` still invokes the extractor, but the extractor writes to the legacy `facts` table, not to `character_facts`. This means any facts the extractor discovers during Step 3's transitional window are not reflected in the character system prompt. That is accepted: the extraction service is deprecated and any gap is temporary until Step 4 lands.

---

## Detailed Design

### 1. `build_system_prompt()` — Schema-Tree Rendering

**File:** `src/memories/services/prompt_builder.py`

**Signature change:**

```python
# Before
def build_system_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:

# After
def build_system_prompt(
    character: Character,
    facts_blob: dict[str, Any],
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
```

The function calls `load_schema()` at the top to get the canonical schema. It walks every top-level key in the schema (`Character`, `User`, `Setting`) and renders them as a named section. Within each section it recursively descends groupings, rendering every leaf regardless of whether the blob contains a value for it.

**Preamble block** — added once above all sections:

```
## World State

The following describes everything known about the world right now.
Facts are organised by category and grouped by topic.

Mutability levels:
- IMMUTABLE: fixed permanently once set. You may not invent or change these.
  If a value is missing and you need it, call require_fact() rather than making one up.
- MUTABLE: stable but can change with narrative context. If your response implies a
  change, the evaluator will surface it to the user for approval.
- FLUID: expected to change freely within a session. Update via the evaluator naturally.

For ENUM facts, only the listed values are valid.
```

**Leaf rendering** — each leaf renders as a compact block:

```
[IMMUTABLE] Character.Identity.Name — The character's given name
  Value: Sarah

[FLUID] Character.State-Of-Mind.Mood — The character's dominant emotional state
  Value: Anxious
  Valid values: Calm | Anxious | Angry | Sad | Joyful | Neutral | Guarded | Excited

[IMMUTABLE] Character.Identity.Age — The character's age in years
  Value: (not set)
```

- The full dot-notation path (`Character.State-Of-Mind.Mood`) appears on the label line alongside the mutability tag and description.
- Populated leaves show `Value: <stored value>`.
- Unpopulated leaves show `Value: (not set)`.
- `Enum` leaves always show `Valid values:`, regardless of whether a value is set.
- `String` and `Integer` leaves do not show a `Valid values:` line.

**Rendering helpers to add:**

Two private helpers inside `prompt_builder.py`:

```python
def _lookup_blob_value(
    blob: dict[str, Any],
    path_parts: list[str],
) -> str | int | None:
    """Walk the blob following path_parts; return the leaf Value or None if absent."""
```

This traverses the blob dict following each path segment; at the final segment it looks for a `{"Value": ...}` dict. Returns `None` if any intermediate node is missing or if the `Value` key is absent.

```python
def _render_schema_node(
    schema_node: dict[str, Any],
    blob_node: dict[str, Any],
    prefix: str,
    lines: list[str],
) -> None:
    """Recursively walk schema_node, rendering leaves into lines."""
```

For each key in `schema_node`:
- If the child has a `"Type"` field, it is a leaf → render it (path, mutability, description, value, constraint).
- Otherwise it is a grouping → recurse with the updated prefix and the corresponding blob sub-dict (which may be empty if no values exist under that grouping yet).

This helper mutates `lines` in place, consistent with the existing function's `lines: list[str]` pattern.

**Section-level rendering:**

```
### Character

[IMMUTABLE] Character.Identity.Name — …
…

### User

[IMMUTABLE] User.Identity.Name — …
…

### Setting

[MUTABLE] Setting.Location.Name — …
…
```

Top-level schema keys are rendered as `### <Key>` subsections within the `## World State` block. This preserves the human-readable grouping the current prompt already uses (User / Character / Setting) while using schema-derived names rather than hard-coded strings.

**No-facts case:** Because the function now always renders the full schema tree (with `(not set)` for missing leaves), the "no facts have been established" fallback is removed. An empty blob produces a prompt that lists all schema paths as unset — the character knows the taxonomy but sees no values. The `do not invent biographical details` instruction is moved into the preamble's `IMMUTABLE` mutability explanation.

**Inferences and Experiences sections** — unchanged in format. They follow the schema sections exactly as they do today.

**Imports to add:** `from memories.schema_loader import load_schema`. `dict[str, Any]` requires `from typing import Any`.

---

### 2. `build_evaluator_prompt()` — Schema Section and Fact Format

**File:** `src/memories/services/evaluator.py`

**Signature change:**

```python
# Before
def build_evaluator_prompt(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:

# After
def build_evaluator_prompt(
    character: Character,
    facts_blob: dict[str, Any],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
```

**Schema section** — added at the top of the prompt, using `render_schema_for_prompt()` from `schema_loader.py`. This function already exists and produces the grouped `IMMUTABLE / MUTABLE / FLUID` listing with path descriptions and Enum constraints. No changes to `schema_loader.py` are needed for this.

**Fact-values section** — the current `## Established Facts (id: key: value (category, mutability))` block is replaced. The evaluator receives the current blob values so it can compare the character response against them. The new format uses dot-notation paths and derives mutability from the schema rather than from the old row-level `mutability` column:

```
## Current Fact Values

Character.Identity.Name: "Sarah"
Character.State-Of-Mind.Mood: "Anxious"  [Fluid]
Setting.Location.Name: "Chicago Memorial Hospital"  [Mutable]

(all other schema paths are unset)
```

To build this section the function calls `schema_loader._collect_leaves()` (or an equivalent flat-listing utility) to enumerate all schema leaves, then looks up each path in the blob. Leaves with a value are rendered first; the trailing line summarises unpopulated paths without enumerating them individually, keeping the prompt compact when most paths are unset.

The numeric `[id]` prefix used in the old format is dropped; the evaluator now references facts by path string, not by row id.

**Inference section** — retained as-is. The existing `## Established Inferences (id: statement)` section continues to use integer IDs because the `Inference` model still has them.

**Experience section** — retained as-is.

**Mutability rules and task description** — rewritten to match the v2 vocabulary:

- Remove all mention of `implication` and `experience_update`.
- Remove the `implication` verdict from the example priority list at the end.
- Replace the `LOW-mutability / HIGH-mutability` implication language with:
  - `MUTABLE`: the evaluator should call `set_fact` (Step 5); for now it treats changes as `contradiction` only for `Immutable` paths; changes to `Mutable` paths that the character implies are noted in the `decision_log` but not flagged as violations (they will surface through the approval gate in Step 7).
  - `FLUID`: any change is accepted silently; not flagged at all.
- The evaluator's job at this stage is to catch `Immutable` violations (→ `contradiction`) and derive inferences (→ `new_inference_logical` / `new_inference_probabilistic`) or declare clean (→ `pass`).

**JSON output format** — updated to remove `implication`/`experience_update` fields:

```
Return a JSON object with this exact structure:

{
  "verdict": "<pass|contradiction|new_inference_logical|new_inference_probabilistic>",
  "new_inferences": [
    {
      "inference_type": "logical | probabilistic",
      "statement": "...",
      "derivation": "brief explanation of how this follows from the facts",
      "source_fact_paths": ["Character.Identity.Name", ...],
      "source_inference_ids": []
    }
  ],
  "violations": [
    {
      "type": "contradiction",
      "description": "what immutable fact was contradicted"
    }
  ],
  "decision_log": "One-sentence summary of why you chose this verdict."
}
```

Note that `violations` here only ever has `type: "contradiction"` entries. `source_fact_ids` in `new_inferences` is renamed to `source_fact_paths` (a list of dot-notation path strings) to reflect the new schema-path-based reference system. This aligns with the `inferences` table's `source_fact_paths` column introduced in Step 2.

**Imports to add:** `from memories.schema_loader import load_schema, render_schema_for_prompt`.

---

### 3. Evaluator Model Cleanup

**File:** `src/memories/services/evaluator.py`

#### Remove `ExperienceUpdate`

Delete the `ExperienceUpdate` Pydantic model class entirely:

```python
class ExperienceUpdate(BaseModel):
    contradicted_experience_id: int
    description: str
```

#### Remove `experience_updates` from `EvaluatorResult`

Delete the field from `EvaluatorResult`:

```python
experience_updates: list[ExperienceUpdate] = []  # remove this line
```

#### Remove `suggested_fact` from `Violation`

The `suggested_fact: dict[str, str] | None = None` field on `Violation` is removed. The evaluator no longer proposes new fact key-value pairs inline; implied fact updates will become `set_fact` tool calls in Step 5. For now, violations carry only `type` and `description`.

```python
class Violation(BaseModel):
    type: str
    description: str
    # suggested_fact removed
```

#### Remove `_violation_duplicates_existing_fact()`

This helper existed solely to filter evaluator-proposed `implication` violations whose suggested fact was already stored. Since:
- `implication` is no longer a valid verdict
- `Violation.suggested_fact` no longer exists

The helper has no callers after this step and should be deleted in full.

#### Narrow `_VALID_VERDICTS`

```python
_VALID_VERDICTS = frozenset(
    {
        "pass",
        "contradiction",
        "new_inference_logical",
        "new_inference_probabilistic",
    }
)
```

`implication` and `experience_update` are removed. Any evaluator response containing either will now raise `EvaluatorParseError`, which triggers the existing fallback in `run_contradiction_loop()` (deliver response unverified, log a warning).

#### Remove the contradiction priority override

The existing block that forces `verdict = "contradiction"` when any violation has `type == "contradiction"` is retained; it remains valid and protects against malformed model output. The block that subsequently filtered `implication` violations whose suggested fact duplicated an existing fact is removed (along with its helper).

#### Update `NewInference` model

Rename `source_fact_ids: list[int]` to `source_fact_paths: list[str]` to align with the new schema-path reference system. The existing coercion block that dropped non-integer `source_fact_ids` is updated (or removed, since path strings cannot be mistaken for integers and the coercion issue no longer applies).

---

### 4. `chat_service.py` — Dead-Code Removal and Signature Updates

**File:** `src/memories/services/chat_service.py`

#### Remove `experience_update` handling

Lines 267–288 of the current implementation handle the `experience_update` verdict by calling `delete_experience()` and `remove_active_experience()`. Since `experience_update` is no longer in `_VALID_VERDICTS`, it can never be returned by the evaluator, making this block unreachable dead code. It must be removed now because it accesses `eval_result.experience_updates`, a field that no longer exists on `EvaluatorResult`.

The companion block at line 304 (`if eval_result.verdict in ("new_inference_logical", "experience_update")`) changes to simply `if eval_result.verdict in ("new_inference_logical",)` (or equivalently `if eval_result.verdict == "new_inference_logical"`), since `experience_update` is gone. Logical inferences are still auto-promoted.

#### Retain `implication` verdict handling (for now)

The `implication` verdict branch in `run_turn()` is left in place. Since `implication` was removed from `_VALID_VERDICTS`, it can never be returned by the evaluator, and the branch is effectively dead. But removing it is Step 7's responsibility. Leaving it avoids scope creep in Step 3.

#### Update fact-loading call

`run_turn()` currently calls:

```python
character, facts, inferences, history, turn_id = await asyncio.gather(
    get_character(db, session.character_id),
    get_fact_rows(db, session.character_id),
    ...
)
```

Replace `get_fact_rows` with `get_facts`:

```python
character, facts_blob, inferences, history, turn_id = await asyncio.gather(
    get_character(db, session.character_id),
    get_facts(db, session.character_id),
    ...
)
```

Import `get_facts` instead of `get_fact_rows` from `memories.database`.

The block that processes extraction results and calls `get_fact_rows()` a second time after extraction (lines 213–234) no longer applies. Since `get_facts()` reads from `character_facts` and the extractor still writes to the legacy `facts` table, re-reading after extraction would still return the blob (unchanged by extraction). This block is simplified: the extraction still runs (it is not removed until Step 4), but post-extraction re-reading via the old table is dropped. The call to `get_fact_rows()` inside that block is removed. Facts are not reloaded mid-turn until Step 4's World Builder writes to `character_facts` directly.

#### Update `build_system_prompt` call

```python
system_prompt = build_system_prompt(character, facts_blob, inferences, active or None)
```

#### Update `run_contradiction_loop()` signature

```python
async def run_contradiction_loop(
    model: str,
    base_messages: list[dict[str, str]],
    character: Character,
    facts_blob: dict[str, Any],   # was: facts: list[Fact]
    user_content: str,
    ollama: OllamaClient,
    ...
) -> tuple[str, str, EvaluatorResult]:
```

The `facts_blob` is passed through to `run_evaluator()`.

#### Update `run_evaluator()` call

```python
ev = await run_evaluator(
    character,
    facts_blob,       # was: facts
    user_content,
    content,
    ollama,
    ...
)
```

#### Remove now-unused imports

`get_fact_rows`, `Fact` (if no longer used), `delete_experience`, `ExtractionResult`, and `run_fact_extractor` imports — check each one. `Fact` is used in type annotations inside the extraction processing block; if that block is simplified or removed, `Fact` may become unused. `delete_experience` is used only inside the `experience_update` block being removed.

---

### 5. World Builder Note

Step 3's plan bullet says "Update World Builder prompt to match." The World Builder does not exist yet — it is created in Step 4 as a replacement for `extraction_service.py`. There is nothing to update in `extraction_service.py` that would survive Step 4's rewrite. This bullet is understood to mean: the World Builder, when written in Step 4, must include the `## Fact Schema` section (using `render_schema_for_prompt()`) in its prompt, consistent with how the evaluator prompt is updated here. No changes to `extraction_service.py` are made in Step 3.

---

## Transitional State After Step 3

After Step 3 and before Step 5:

- The character system prompt renders the full schema tree. If `character_facts` is empty (no blob yet), all leaves show `(not set)`. This is correct behaviour for a new character.
- The evaluator classifies character responses and returns `pass`, `contradiction`, `new_inference_logical`, or `new_inference_probabilistic`. Changes to `Mutable` facts that the character implies are not flagged (this is Step 5's domain).
- The extraction service still runs pre-turn and writes extracted facts to the legacy `facts` table — but those writes do not appear in the character system prompt. This is a known gap accepted for the duration of Steps 3–4.
- The `implication` verdict handling in `chat_service.py` is dead code but present.
- `proposed_inferences` continue to flow through the existing `new_inference_*` path (auto-promote logical; require user review for probabilistic) unchanged until Step 5 replaces the evaluator with a tool-call loop.

---

## Test Plan

### Files affected

| File | Action |
|---|---|
| `tests/unit/test_prompt_builder.py` | Heavy revision — most tests use `list[Fact]`; all rewritten for blob input |
| `tests/unit/test_evaluator_service.py` | Partial revision — remove tests for deleted features, update signature |
| `tests/unit/test_chat_service.py` | Update to pass `facts_blob` (a `dict`) where `list[Fact]` was passed |
| `tests/unit/conftest.py` | Remove `experience_updates` parameter from `make_evaluator_ndjson()` |
| `tests/integration/test_api_*.py` | Update any tests relying on `implication` or `experience_update` verdicts |

---

### `test_prompt_builder.py` — changes

**Tests to delete** (they test behaviour that no longer exists):

- All tests in the "Phase 4 additions" section that test category-section header names (e.g. `## Facts About You (Character)`, `## Facts About The User`) and the `[low-mutability`, `[fluid` annotation format — these headers and annotations are replaced by the schema-tree format.
- `test_no_facts_yields_no_invention_instruction` — the no-facts fallback path is removed.
- `test_facts_section_header_present_regardless_of_fact_count` — the specific header names change.
- `test_fact_order_preserved`, `test_all_facts_injected_as_key_value_pairs`, `test_character_facts_appear_under_character_section`, `test_user_facts_appear_under_user_section`, etc. — all tests that pass `list[Fact]` objects or check for the old `key: value` line format.

**Tests to add:**

The new tests validate the schema-tree rendering contract:

- `test_world_state_header_present` — prompt contains `## World State`.
- `test_mutability_preamble_present` — prompt contains `IMMUTABLE`, `MUTABLE`, `FLUID` explanation.
- `test_populated_leaf_shows_value` — given a blob with `{"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}`, the rendered prompt contains `Value: Sarah` on a line below `Character.Identity.Name`.
- `test_unpopulated_leaf_shows_not_set` — given an empty blob `{}`, the rendered prompt contains `Value: (not set)` for every schema leaf.
- `test_enum_leaf_always_shows_valid_values` — regardless of whether `Character.State-Of-Mind.Mood` has a value, the prompt always contains `Valid values: Calm | Anxious | ...`.
- `test_enum_leaf_shows_valid_values_when_set` — when the blob has a Mood value, both `Value: Anxious` and `Valid values:` appear.
- `test_immutable_mutability_tag` — an immutable leaf shows `[IMMUTABLE]`.
- `test_mutable_mutability_tag` — a mutable leaf shows `[MUTABLE]`.
- `test_fluid_mutability_tag` — a fluid leaf shows `[FLUID]`.
- `test_full_dot_path_in_leaf_label` — the leaf label contains the full dot-notation path (e.g. `Character.Appearance.Hair.Colour`).
- `test_character_section_header` — prompt contains a `### Character` section header.
- `test_user_section_header` — prompt contains a `### User` section header.
- `test_setting_section_header` — prompt contains a `### Setting` section header.
- `test_character_section_precedes_user_section` — the `### Character` header appears before `### User` in the rendered prompt.
- `test_user_section_precedes_setting_section` — similarly ordered.
- `test_inferences_section_follows_schema_sections` — `## Your Inferences` appears after the last schema section.
- `test_inferences_section_absent_when_none` — unchanged from existing test; retained.
- `test_experiences_section_follows_inferences` — unchanged; retained.
- `test_character_name_appears_in_prompt` — retained with updated blob-based call signature.

Tests for `inferences` and `experiences` parameters are largely unchanged in assertion content (same section headers and statement rendering); only the call signature changes (`facts_blob={}` instead of `facts=[]`).

---

### `test_evaluator_service.py` — changes

**Tests to delete:**

- `test_evaluator_parses_implication_verdict` — `implication` is no longer valid.
- `test_evaluator_returns_experience_update_verdict_not_coerced` — `experience_update` is no longer valid.
- All `test_evaluator_strips_implication_*` and `test_evaluator_does_not_strip_implication_*` tests — `_violation_duplicates_existing_fact()` is removed.
- `test_evaluator_partial_filter_keeps_genuine_violations` — same.
- `test_evaluator_duplicate_filter_does_not_apply_to_contradiction_verdict` — same.
- `test_evaluator_prompt_warns_against_re_proposing_existing_facts` — the instruction is removed.
- `test_evaluator_result_model_has_experience_updates_field` — field removed.
- All Phase 5 tests for `experience_update` verdict parsing (`test_run_evaluator_returns_experience_update_verdict`, `test_run_evaluator_parses_experience_updates_list`, etc.).
- `test_contradiction_takes_priority_over_experience_update` — the scenario no longer applies.
- Phase 4 tests for `implication`/`high`/`low` mutability instructions that reference the removed vocabulary — **check each individually**: some reference immutable contradiction logic that is retained; only delete the ones that specifically test the `implication`-for-mutable-facts or `experience_update` instructions.

**Tests to update (signature change only):**

All tests calling `build_evaluator_prompt(_CHARACTER, _FACTS, ...)` change to `build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, ...)` where `_FACTS_BLOB` is a dict. Define a module-level fixture:

```python
_FACTS_BLOB: dict[str, Any] = {
    "Character": {
        "Identity": {"Occupation": {"Value": "surgeon"}},
        "Background": {"Hometown": {"Value": "Reykjavik"}},
    }
}
```

(Paths chosen to match the schema's `Character.Identity.Occupation` and `Character.Background.Hometown` leaves.)

Tests asserting that the prompt contains specific fact values now check for path-based rendering (e.g. `"Character.Identity.Occupation"` and `"surgeon"` in prompt) rather than the old `"occupation: surgeon"` format.

**Tests to add:**

- `test_evaluator_prompt_includes_schema_section` — `## Fact Schema` appears in the prompt output.
- `test_evaluator_prompt_includes_immutable_paths` — at least one `IMMUTABLE` path label appears in the prompt.
- `test_evaluator_prompt_includes_fact_values_section` — `## Current Fact Values` (or the chosen header) appears.
- `test_evaluator_prompt_renders_populated_path` — a blob with a value produces a `Character.Identity.Occupation: "surgeon"` line in the fact-values section.
- `test_evaluator_prompt_empty_blob_produces_unset_note` — an empty blob produces an `(all other schema paths are unset)` line or equivalent.
- `test_evaluator_raises_parse_error_on_implication_verdict` — the evaluator now raises `EvaluatorParseError` when the model returns `implication`. This is a regression-guard test confirming the verdict is no longer accepted.
- `test_evaluator_raises_parse_error_on_experience_update_verdict` — same for `experience_update`.
- `test_new_inference_source_fact_paths_are_strings` — given an evaluator response with `source_fact_paths: ["Character.Identity.Age"]`, the parsed `NewInference.source_fact_paths` is a list of strings.
- `test_violation_has_no_suggested_fact_field` — constructing a `Violation` object and checking it raises `AttributeError` or `ValidationError` if `suggested_fact` is passed.

---

### `test_chat_service.py` — changes

Any test that constructs a `facts: list[Fact]` and passes it to `run_contradiction_loop()` or `run_turn()` must be updated to pass a `facts_blob: dict[str, Any]` instead. In practice, most chat service tests mock the Ollama HTTP layer via `respx` and call `run_turn()` end-to-end; these tests change only their `facts_blob` fixture (an empty dict or a minimal populated dict) rather than building `Fact` objects.

The evaluator mock (`make_evaluator_ndjson()`) in `tests/unit/conftest.py` loses its `experience_updates` parameter. Any test passing `experience_updates=...` to that helper is updated accordingly.

Tests that verify the `experience_update` branch in `run_turn()` (e.g. tests that confirm `delete_experience` is called on an experience_update verdict) are deleted.

---

### Existing tests that are unaffected

- All `test_ollama_client.py` tests.
- All `test_inference_service.py` tests.
- All `test_experience_service.py` tests (the experience service itself is not changed).
- All frontend tests (`chat.test.js`, `chat-component.test.js`).
- All integration tests for facts, inferences, experiences, decisions, schema, and sessions repositories.
- The `GET /api/schema` endpoint test.

---

## Edge Cases

**Empty blob.** `get_facts()` returns `{}` for a character with no entries in `character_facts`. `build_system_prompt()` must handle this gracefully: walk the schema normally, producing `Value: (not set)` for every leaf. No guard clause needed — the blob-lookup function returns `None` for missing keys, and the rendering treats `None` as unset.

**Schema masking.** `get_facts()` already applies `apply_mask()` before returning the blob. Any path in `character_facts` that doesn't match the current schema is silently dropped. `build_system_prompt()` therefore never encounters unexpected paths in the blob — it only ever sees paths that exist in the schema.

**`Enum` type-checking in the prompt.** `build_system_prompt()` does not validate that a stored blob value is a member of the Enum's `Constraint` list. Validation is the job of `check_write_permitted()` at write time. The prompt renders whatever value is stored, even if it is technically out-of-range (this can happen if the schema evolves after a value is written). The character prompt simply shows what's there.

**`Integer` type rendering.** Blob values for `Integer` leaves are stored as Python ints (not strings). `_lookup_blob_value()` should return them as-is; the rendering code converts to string for display. No special handling needed beyond `str(value)`.

**Section ordering.** The schema is a Python dict. In Python 3.7+ dicts preserve insertion order, and `fact_schema.json` is loaded by `json.loads()` which respects object key order. The schema file defines `Character` before `User` before `Setting`, so sections render in that order consistently. No explicit ordering step is required.

**Schema changes between sessions.** If a leaf is added to the schema after a session has started, the character system prompt will show the new leaf as `(not set)`. If a leaf is removed, the blob still holds its value (masked on read, so it's dropped). The character sees the new schema state on the very next turn. No session restart is required.

**`run_turn()` and the extraction gap.** Between Step 3 and Step 4, the extraction service runs and writes to the old `facts` table, but those writes are invisible to the character. This means a user message like "my name is Jon" may not be captured until Step 4's World Builder is in place. Document this as a known transitional limitation in the commit message.

---

## Post-Implementation Cleanup Tasks

Issues found during adversarial review that were deferred rather than fixed in-session.

### CT-1: `source_fact_paths` is unimplemented end-to-end

**Decided:** Fix in a follow-up before Step 4 begins.

The spec says Step 2 introduced a `source_fact_paths` column on the `inferences` table. It was not added. The gap means:

- `NewInference.source_fact_paths` (the evaluator output field) is silently dropped on every auto-promote in `run_turn()` — `create_inference()` has no `source_fact_paths` parameter.
- `Inference` model (`models/__init__.py`) and `_parse_inference()` in `database.py` still use `source_fact_ids: list[int]`, not path strings.
- `_AcceptInferenceBody` in `implication.py` still has `source_fact_ids: list[int]` — the accept-inference API now speaks a different language than what the evaluator emits.

**What to do:**

1. Add `source_fact_paths TEXT` column to the `inferences` DDL in `database.py`.
2. Update `_parse_inference()` to parse the column (JSON array of strings; default `[]`).
3. Add `source_fact_paths: list[str] = []` to `Inference` in `models/__init__.py`.
4. Add `source_fact_paths: list[str] | None = None` parameter to `create_inference()`, stored as JSON.
5. Pass `source_fact_paths=inf.source_fact_paths` in the auto-promote block in `chat_service.py`.
6. Update `_AcceptInferenceBody.source_fact_ids` → `source_fact_paths: list[str] = []` in `implication.py`, and update `create_inference()` call at line 206 accordingly.
7. Add integration tests for the new column (write, read-back, accept-inference endpoint).

### CT-2: `test_run_turn_passes_tier1_and_tier2_facts_to_character` is a false positive

**Decided:** Delete and replace with a correct test.

The test asserts `"Chicago" in system_content` after the extractor writes `meeting_location: Chicago` to the legacy `facts` table. It passes because `"Chicago"` appears in the schema description text for `Setting.Location.Name` (`"e.g. 'Chicago Memorial Hospital'"`) — not because the extraction result reached the system prompt. The test's docstring is the opposite of the correct step 3 behavior.

**What to do:**

1. Delete `test_run_turn_passes_tier1_and_tier2_facts_to_character` from `tests/unit/test_chat_service.py`.
2. Add a replacement test: after the extractor writes a new fact, assert the character system prompt is unchanged (i.e., the blob-derived prompt does not contain the extracted value). Name it `test_run_turn_extraction_does_not_affect_system_prompt` to make the transitional intent explicit.

### CT-3: `_lookup_blob_value()` in `prompt_builder.py` is dead code

**Decided:** Delete it.

`_lookup_blob_value` ([prompt_builder.py:26-38](src/memories/services/prompt_builder.py#L26)) was specified as the helper that `_render_schema_node` should call to resolve a blob leaf. The implementation instead does the lookup inline inside `_render_schema_node` (lines 53-54), so `_lookup_blob_value` is never called. The inline approach is correct in context (the recursive pattern already passes the current sub-dict as `blob_node`, so a top-down path-walk helper isn't needed).

**What to do:** Delete `_lookup_blob_value` from `prompt_builder.py`. No callers to update.

### CT-4: Mock inference payloads in `test_chat_service.py` use the old `source_fact_ids` field name

**Decided:** Update all payloads to use `source_fact_paths`.

Every `new_inferences` payload in `test_chat_service.py` (e.g. lines 444–450, 469–476, 582–588, 619–626, 643–650) contains `"source_fact_ids": []`. `NewInference` now has `source_fact_paths: list[str] = []`. Pydantic silently ignores the old key and defaults `source_fact_paths` to `[]`, so the tests pass without exercising the renamed field.

**What to do:**

1. Replace every `"source_fact_ids": []` with `"source_fact_paths": []` in all `new_inferences` mock dicts in `test_chat_service.py`.
2. In tests that specifically exercise the path-string content (e.g. depth tests, auto-promote tests), use a realistic value such as `"source_fact_paths": ["Character.Identity.Age"]` to confirm the field flows through.

### CT-5: `decision_log` check in `run_evaluator` contradicts the Pydantic model

**Decided:** Remove the default from `EvaluatorResult.decision_log` so Pydantic enforces the requirement.

`run_evaluator` manually raises `EvaluatorParseError` if `decision_log` is absent from the raw response dict ([evaluator.py:229-230](src/memories/services/evaluator.py#L229)). But `EvaluatorResult` declares `decision_log: str = ""`, so Pydantic would silently accept an absent field. The two places contradict each other.

**What to do:**

1. Change `decision_log: str = ""` to `decision_log: str` in `EvaluatorResult` (no default).
2. Delete the manual `if "decision_log" not in data` check — Pydantic's `ValidationError` path (already caught and re-raised as `EvaluatorParseError` on line 244) will handle it.
3. Update `test_evaluator_raises_parse_error_on_missing_verdict` (or add a sibling) to confirm a missing `decision_log` also raises `EvaluatorParseError`.
