# Phase 3 Implementation Plan — Inference Generation

## Goals (from plan.md)

- Inject active Inferences into the character system prompt alongside Facts
- Pass active Inferences to the evaluator so established inferences are treated as grounded (not flagged as new)
- **Eager pass**: when a Fact is added or edited, a dedicated LLM call derives all conclusions the current Fact and Inference set can support; new inferences are stored to DB
- **Lazy discovery depth**: compute depth for lazily-discovered inferences before storing; discard those exceeding `MAX_INFERENCE_DEPTH`
- **Cascade on Fact edit**: revalidate every Inference that directly or transitively depends on the changed Fact; mark those that no longer hold as `stale`; surface stale inferences to the user
- **Cascade on Fact delete**: mark all Inferences that depend on the deleted Fact — directly or transitively — as `invalidated`; surface to the user for keep/rewrite/delete

**Deliverable:** The character speaks from a coherent, automatically-maintained inference base. New Facts automatically unlock related conclusions. Changes to Facts propagate cleanly through the inference graph, surfacing broken derivations for user review.

---

## Task List

### T1 — Database additions

The existing `create_inference` and `get_inferences` repository functions carry over from Phase 2. Three new functions are needed to manage inference lifecycle.

**T1.1 — `get_inference`**
```python
get_inference(db, inference_id: int) -> Inference | None
```
Single-row lookup by primary key. Returns `None` if the id does not exist. Needed by the cascade and revalidation endpoints to load individual inferences.

**T1.2 — `update_inference_status`**
```python
update_inference_status(db, inference_id: int, new_status: str) -> Inference
```
Updates the `status` column for a single inference. Valid targets are `"active"`, `"stale"`, and `"invalidated"`. Raises `NotFoundError` if the id does not exist. Returns the updated `Inference`. Used by cascade functions and by the inference management endpoints.

**T1.3 — `delete_inference`**
```python
delete_inference(db, inference_id: int) -> None
```
Hard-deletes the row. Raises `NotFoundError` if the id does not exist. Users call this when they decide a stale or invalidated inference should be removed entirely.

---

### T2 — Prompt builder update

**T2.1 — Updated `build_system_prompt` signature**
```python
def build_system_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference] | None = None,
) -> str
```

The `inferences` parameter defaults to `None`/empty so that existing Phase 1/2 call sites continue to work without modification. Phase 3 callers pass the active inference list explicitly.

**T2.2 — Inferences section in the prompt**

When `inferences` is non-empty, append a new section after the Facts block:

```
## Your Inferences
These conclusions have been derived from your Facts. They are as reliable as the
Facts they came from. Do not contradict them.

{statement} (from: {derivation})
{statement} (from: {derivation})
```

When `inferences` is empty or `None`, the section is omitted entirely — no empty header.

---

### T3 — Evaluator update

The evaluator in Phase 2 receives only Facts. In Phase 3, established Inferences must be included so the evaluator treats them as grounded — otherwise every inference-backed assertion the character makes would be flagged as a new implication.

**T3.1 — Updated `build_evaluator_prompt` signature**
```python
def build_evaluator_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],    # new parameter
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
) -> str
```

**T3.2 — Established Inferences section in the evaluator prompt**

After the Facts block, add:

```
## Established Inferences (id: statement)
[{id}] {statement}  (from: {derivation})
```

When the list is empty, replace with `(no inferences established yet)`.

**T3.3 — Updated verdict instructions**

The grounding section of the prompt is updated to add Inferences as an additional grounded source:

> A detail is grounded if it can be directly looked up in the **Established Facts** list, found verbatim in the **Established Inferences** list, or is a necessary logical consequence of the above. A detail is **not** automatically grounded just because it is consistent with the facts or sounds plausible.

The `new_inference_logical` / `new_inference_probabilistic` verdicts should only fire for conclusions that are not already in the Established Inferences list. The instructions are updated to make this explicit.

**T3.4 — Updated `run_evaluator` signature**
```python
async def run_evaluator(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],    # new parameter
    user_message: str,
    character_response: str,
    ollama: OllamaClient,
    contradiction_hints: list[str] | None = None,
) -> EvaluatorResult
```

`run_contradiction_loop` in `chat_service.py` is updated to accept and forward `inferences`.

---

### T4 — Inference service (new file: `src/memories/services/inference_service.py`)

**T4.1 — Depth computation helper**
```python
def compute_depth(
    source_inference_ids: list[int],
    known_inferences: list[Inference],
) -> int
```

Returns `1` if `source_inference_ids` is empty (derived from Facts only). Otherwise returns `max(inf.depth for inf in known_inferences if inf.id in source_inference_ids) + 1`. If any source id is not found in `known_inferences` (LLM referenced a non-existent inference), it is silently skipped; if no valid source found, returns `1`.

**T4.2 — Eager pass prompt builder**
```python
def build_eager_pass_prompt(
    character: Character,
    facts: list[Fact],
    existing_inferences: list[Inference],
    max_breadth: int,
) -> str
```

Produces a user-message prompt for the eager-pass LLM call. Structure:

```
Character: {name}

## Current Facts (id: key: value)
[{id}] {key}: {value}
...

## Already Established Inferences (do NOT re-derive these)
[{id}] {statement}  (from: {derivation})
...
(or: "(none established yet)")

## Your Task
You are a logical reasoner. Derive up to {max_breadth} NEW conclusions from the
Facts above that are not already listed in Established Inferences.

Rules:
- LOGICAL: only derive what is certain — a strict logical consequence of one or
  more Facts (e.g. birth year from age + current year).
- PROBABILISTIC: a well-founded tendency or likelihood given the Facts, not a
  specific invented detail.
- Do NOT re-derive anything already in the Established Inferences list.
- Cite source Facts and Inferences by id.
- Cross-references within this same response are NOT allowed — only cite Facts
  and Inferences already established before this pass.
- Aim for depth {max_depth} or fewer hops from root Facts.

Return a JSON array (empty array if nothing new to derive):
[
  {
    "inference_type": "logical | probabilistic",
    "statement": "...",
    "derivation": "brief explanation of how this follows",
    "source_fact_ids": [int, ...],
    "source_inference_ids": [int, ...]
  }
]

Return only the JSON array, no other text.
```

**T4.3 — Eager pass execution**
```python
MAX_INFERENCE_DEPTH: int = int(os.getenv("MAX_INFERENCE_DEPTH", "5"))
MAX_INFERENCE_BREADTH: int = int(os.getenv("MAX_INFERENCE_BREADTH", "5"))

class InferenceParseError(Exception):
    """Raised when the eager-pass LLM returns unparseable output."""

async def run_eager_pass(
    db: aiosqlite.Connection,
    character: Character,
    facts: list[Fact],
    existing_inferences: list[Inference],
    ollama: OllamaClient,
    max_depth: int = MAX_INFERENCE_DEPTH,
    max_breadth: int = MAX_INFERENCE_BREADTH,
) -> list[Inference]
```

1. Build the system prompt (a brief role declaration) and user prompt via `build_eager_pass_prompt`.
2. Call `ollama.chat(model, messages, think=False, format="json")`.
3. Strip markdown fences; `json.loads`; raise `InferenceParseError` on failure.
4. Coerce `source_fact_ids` / `source_inference_ids` to `list[int]`, dropping non-integers (same guard as in the evaluator).
5. **Validate sources**: discard any item whose `source_inference_ids` reference an id not in `existing_inferences` — same-pass cross-references are rejected (see D8).
6. **Compute depth**: call `compute_depth(item.source_inference_ids, existing_inferences)`.
7. **Apply depth cap**: discard any item with `computed_depth > max_depth`.
8. **Apply breadth cap**: take at most the first `max_breadth` items that pass the above checks.
9. Store each surviving item via `create_inference(db, ...)` with the computed depth.
10. Return the list of stored `Inference` objects.

**T4.4 — Revalidation prompt builder**
```python
def build_revalidation_prompt(
    inference: Inference,
    facts: list[Fact],
    active_inferences: list[Inference],
) -> str
```

Structure:

```
## Current Facts (id: key: value)
[{id}] {key}: {value}

## Other Active Inferences (context only)
[{id}] {statement}

## Inference to Revalidate
Statement: "{inference.statement}"
Original derivation: "{inference.derivation}"
Original sources: Facts {inference.source_fact_ids}, Inferences {inference.source_inference_ids}

## Your Task
Given the CURRENT facts above, does this inference still hold?
Return JSON exactly:
{"holds": true | false, "reason": "one sentence"}
```

**T4.5 — Single-inference revalidation**
```python
async def revalidate_single_inference(
    inference: Inference,
    facts: list[Fact],
    active_inferences: list[Inference],
    ollama: OllamaClient,
) -> bool
```

Calls the LLM with the revalidation prompt; parses `holds` from the JSON response. On parse failure, defaults to `True` (conservative — don't invalidate if uncertain).

**T4.6 — Cascade on Fact edit**
```python
async def cascade_on_fact_edit(
    db: aiosqlite.Connection,
    character_id: int,
    changed_fact_id: int,
    ollama: OllamaClient,
) -> list[Inference]
```

Returns all inferences that were marked `stale` during the cascade.

Algorithm (breadth-first, iterative):
1. Load all `active` inferences for the character.
2. Load the current facts (the fact has already been updated in DB by the time this runs).
3. Seed a worklist with inferences whose `source_fact_ids` contains `changed_fact_id`.
4. While worklist is non-empty:
   a. Pop one inference.
   b. Call `revalidate_single_inference(inference, facts, remaining_active_inferences)`.
   c. If `holds=False`: call `update_inference_status(db, inference.id, "stale")`; add the inference id to a `newly_stale` set.
   d. Add to the worklist any active inference whose `source_inference_ids` contains the now-stale inference id and has not already been processed.
5. Return the list of stale `Inference` objects (re-fetched after status updates).

`remaining_active_inferences` in step 4b is the active inference set minus those already marked stale in this pass — so each inference is evaluated against the most conservative view of what still holds.

**T4.7 — Cascade on Fact delete**
```python
async def cascade_on_fact_delete(
    db: aiosqlite.Connection,
    character_id: int,
    deleted_fact_id: int,
) -> list[Inference]
```

No LLM call needed — a deleted fact unconditionally invalidates any inference that relied on it.

Algorithm:
1. Load all `active` inferences for the character.
2. Seed a `to_invalidate` set with inferences whose `source_fact_ids` contains `deleted_fact_id`.
3. While new ids were added in the previous iteration:
   - Add any active inference whose `source_inference_ids` intersects `to_invalidate`.
4. Call `update_inference_status(db, id, "invalidated")` for each id in `to_invalidate`.
5. Return the invalidated `Inference` objects (re-fetched from DB).

---

### T5 — Chat service update

**T5.1 — Load and inject inferences per turn**

In `run_turn`, after loading facts:
```python
inferences = await get_inferences(db, session.character_id)  # active only
system_prompt = build_system_prompt(character, facts, inferences)
```

**T5.2 — Pass inferences through the contradiction loop**

`run_contradiction_loop` gains an `inferences: list[Inference]` parameter. It passes inferences to `run_evaluator`.

**T5.3 — Depth-capped lazy inference storage**

When `eval_result.verdict == "new_inference_logical"`, before calling `create_inference`:
```python
for inf in eval_result.new_inferences:
    depth = compute_depth(inf.source_inference_ids, inferences)
    if depth > MAX_INFERENCE_DEPTH:
        continue   # silently discard
    await create_inference(db, ..., depth=depth)
```

`compute_depth` is imported from `inference_service`. The same depth cap applies to `new_inference_probabilistic` inferences accepted by the user (via the accept-inference endpoint in Phase 2's `implication.py` router — the router needs to compute depth when creating the inference).

---

### T6 — API additions and changes

New router: `src/memories/routers/inferences.py`. Mounted at `/api/characters` prefix in `main.py`.

**T6.1 — `POST /api/characters/{character_id}/inferences/generate`**

Triggers the eager pass for a character. Called by the frontend after a Fact is added or edited.

- Validates: character exists; returns 404 otherwise.
- Loads current facts and active inferences from DB.
- Calls `run_eager_pass(db, character, facts, inferences, ollama)`.
- Returns 200 with `{"new_inferences": list[Inference]}`.
- If `run_eager_pass` raises `InferenceParseError` (LLM returned garbage), returns 200 with `{"new_inferences": [], "warning": "Inference pass could not be parsed; try again."}`.

**T6.2 — `POST /api/characters/{character_id}/inferences/revalidate`**

Triggers cascade revalidation after a Fact is edited.

- Body: `{"changed_fact_id": int}`.
- Validates: character exists; Fact with that id exists (returns 404 otherwise).
- Calls `cascade_on_fact_edit(db, character_id, changed_fact_id, ollama)`.
- Returns 200 with `{"stale_inferences": list[Inference]}`.

**T6.3 — `DELETE /api/characters/{character_id}/facts/{key}` (updated)**

The existing endpoint is updated to also run `cascade_on_fact_delete` synchronously before returning. The fact's `id` is retrieved before deletion so it can be passed to the cascade.

Response: **200** (changed from 204) with body `{"invalidated_inferences": list[Inference]}`.

Changing from 204 to 200 requires updating the existing Phase 1 test `test_delete_fact_204` → `test_delete_fact_200_with_cascade`.

**T6.4 — `DELETE /api/characters/{character_id}/inferences/{inference_id}`**

Removes an inference the user has decided to discard (typically after reviewing a stale or invalidated one).

- Validates: character exists; inference exists for this character; returns 404 otherwise.
- Calls `delete_inference(db, inference_id)`.
- Returns 204.

**T6.5 — `PATCH /api/characters/{character_id}/inferences/{inference_id}`**

Allows the user to restore a stale inference to active (when they decide the old derivation is still valid despite the fact change), or to mark an active one as stale/invalidated manually.

- Body: `{"status": "active" | "stale" | "invalidated"}`.
- Validates: character exists; inference exists for this character.
- Calls `update_inference_status(db, inference_id, body.status)`.
- Returns 200 with the updated `Inference`.

**T6.6 — `GET /api/characters/{character_id}/inferences` (updated)**

The existing endpoint in `facts.py` returns only active inferences (no query parameter). Add an optional `?status=` query parameter: `active` (default), `stale`, `invalidated`, or `all` (returns all statuses). Affects only the DB query; no other changes.

---

### T7 — Frontend updates

**T7.1 — Inferences section in sidechannel**

Below the Facts list, add a collapsible Inferences section (expanded by default). Shows all active inferences for the character, grouped by type (Logical / Probabilistic). Each row:
```
[logical]  Born in 1993  (from: age=33, current_year=2026)
```

Refreshed after every turn (lazy discovery may have added new inferences) and after every fact mutation.

**T7.2 — Post-fact-add inference generation flow**

After a successful `POST /facts`:
1. Show a "Deriving inferences…" spinner in the inferences panel.
2. Call `POST /inferences/generate`.
3. On response: refresh the inference list; if `new_inferences` is non-empty, briefly highlight the new rows.

**T7.3 — Post-fact-edit cascade + generation flow**

After a successful `PUT /facts/{key}`:
1. Show a "Revalidating inferences…" spinner.
2. Call `POST /inferences/revalidate` with the edited fact's id.
3. If `stale_inferences` is non-empty, render them in an "Attention needed" section with **Keep** / **Delete** buttons.
4. Then call `POST /inferences/generate` to derive any new inferences enabled by the changed fact.
5. Refresh the inference list.

**T7.4 — Post-fact-delete cascade flow**

After a successful `DELETE /facts/{key}` (now returns 200 with `{invalidated_inferences}`):
1. If `invalidated_inferences` is non-empty, show them in an "Invalidated inferences" section.
2. Each row has **Keep as-is** (PATCH status → active) / **Delete** buttons.
3. Refresh the inference list.

**T7.5 — Stale/invalidated inference management UI**

Stale and invalidated inferences shown with a warning badge. Grouped into an "Attention needed" sub-section above the active list. Three actions per inference:
- **Keep** (PATCH → active): user asserts the inference still holds despite the fact change.
- **Delete** (DELETE): permanently remove the inference.
- (Probabilistic inferences only) an editable text field to rewrite the statement before keeping.

**T7.6 — New API helpers in `chat.js`**

```javascript
export function apiGenerateInferences(characterId)         // POST /generate
export function apiRevalidateInferences(characterId, changedFactId)  // POST /revalidate
export function apiDeleteInference(characterId, inferenceId)         // DELETE
export function apiPatchInferenceStatus(characterId, inferenceId, status) // PATCH
```

These follow the same pattern as the existing `apiAcceptImplication` helpers.

---

## Technology & Infrastructure Decisions

### D1 — Single LLM call per eager pass (not iterative)

The eager pass makes one Ollama call and returns up to `MAX_INFERENCE_BREADTH` new inferences. It does not chain calls to follow derived inferences further. Chaining happens naturally over multiple passes: a second eager pass triggered by the next fact edit can derive inferences from what the first pass stored. This keeps each pass bounded and predictable. If deeper chaining is missed, it surfaces in subsequent sessions.

### D2 — Server-side depth computation (not delegated to the LLM)

The LLM is not trusted to count hops. Depth is computed by the server from `source_inference_ids` against the known inference set. The LLM provides the sources; the server assigns the depth. This makes the depth cap reliable regardless of model quality.

### D3 — Same-pass cross-references rejected

The eager pass prompt instructs the model not to reference inferences returned in the same call. Enforced server-side: any returned inference whose `source_inference_ids` contains an id not in the pre-pass active set is discarded. This avoids ordering problems and keeps depth computation unambiguous.

### D4 — Cycle detection is structural for newly-added inferences

Since inferences are only ever inserted (never updated to add new sources), and SQLite assigns IDs monotonically, an inference newly added in a pass cannot be in any existing inference's source chain — no existing inference knew its ID at creation time. Cycles are therefore structurally impossible for newly-added inferences. The cycle check specified in plan.md is satisfied by this property without an explicit graph walk. A defensive assertion (`inference_id not in source_inference_ids`) is added at write time for safety.

### D5 — Cascade on Fact delete is pure DB (no LLM call)

A deleted Fact unconditionally breaks any Inference that depended on it — there is no revalidation needed. The cascade is a graph traversal over the in-memory inference set, followed by bulk status updates. This is fast (no network IO) and deterministic.

### D6 — Cascade on Fact edit uses a lightweight revalidation prompt (not the full per-turn evaluator)

The full per-turn evaluator checks a character's response against the complete Fact set. The revalidation pass has a narrower question: "does this specific inference still follow from the updated Facts?" A dedicated, simpler prompt is more reliable and cheaper than repurposing the full evaluator. Parse failures default to `holds=True` (conservative — do not mark stale unless certain).

### D7 — Eager pass triggered by a separate REST endpoint, not inline in the Fact mutation

Running the eager pass synchronously inside `POST /facts` or `PUT /facts/{key}` would add multi-second latency to every fact mutation. A separate `POST /inferences/generate` endpoint allows:
1. The fact mutation to return immediately.
2. The frontend to show a distinct "Deriving inferences…" state.
3. The generation to be re-triggered if it fails.
4. Independent testability of the generation endpoint without also testing fact creation.

### D8 — `DELETE /facts/{key}` changed from 204 to 200 with cascade result

The cascade on Fact delete is fast (pure DB) and its result is immediately useful to the frontend (which inferences need attention?). Returning the invalidated inferences in the DELETE response avoids a follow-up GET and gives the frontend the information in a single round-trip. The existing `test_delete_fact_204` test is updated to `test_delete_fact_200_with_cascade`.

### D9 — `MAX_INFERENCE_DEPTH` and `MAX_INFERENCE_BREADTH` read from env (defaults: 5 and 5)

Both are configurable via environment variables, consistent with `MAX_CONTRADICTION_RETRIES` in Phase 2. The defaults from plan.md (5 and 5) are used.

### D10 — `accept-inference` endpoint in Phase 2's `implication.py` updated for depth

When a user accepts a probabilistic inference via `POST .../accept-inference`, the router now calls `compute_depth` before `create_inference`. If the inference would exceed `MAX_INFERENCE_DEPTH`, the endpoint returns 422 with a message explaining the depth cap. This closes the one path in Phase 2 that could store an inference without a depth check.

---

## Test Plan

Tests are written first. The implementation is complete when all Phase 3 tests pass alongside all existing Phase 1 and Phase 2 tests, and overall coverage stays at or above 80% (90% for `services/`).

Ollama HTTP calls in the new `inference_service.py` are mocked with `respx`, using the same pattern as evaluator tests.

---

### Unit tests — `tests/unit/`

#### `test_inference_service.py` (new file)

Tests for the eager pass prompt builder, eager pass execution, revalidation prompt, and cascade functions. All Ollama calls are mocked with `respx`.

**Eager pass prompt**

| Test | Asserts |
|------|---------|
| `test_eager_pass_prompt_includes_all_facts` | Every `key: value` pair from the fact list appears in the built prompt |
| `test_eager_pass_prompt_includes_character_name` | Character name appears in the prompt |
| `test_eager_pass_prompt_lists_existing_inferences` | Each existing inference's statement appears in the "Already Established" section |
| `test_eager_pass_prompt_no_existing_inferences_uses_fallback` | Empty inference list → "(none established yet)" fallback line |
| `test_eager_pass_prompt_instructs_max_breadth` | The `max_breadth` value appears in the prompt |

**Eager pass execution**

| Test | Asserts |
|------|---------|
| `test_eager_pass_parses_returned_inferences` | Valid JSON array → returned as stored `Inference` objects |
| `test_eager_pass_stores_inferences_to_db` | After call, inference rows exist in the DB |
| `test_eager_pass_returns_empty_list_on_empty_json_array` | `[]` response → returns `[]`, no DB writes |
| `test_eager_pass_discards_inference_exceeding_max_depth` | Inference with depth > 5 → not stored, not in return list |
| `test_eager_pass_applies_breadth_cap` | LLM returns 7 inferences → only first `MAX_INFERENCE_BREADTH` (5) stored |
| `test_eager_pass_rejects_same_pass_cross_reference` | `source_inference_ids` references an id not in `existing_inferences` → that item discarded |
| `test_eager_pass_computes_depth_one_for_fact_only_source` | `source_inference_ids=[]` → stored with `depth=1` |
| `test_eager_pass_computes_depth_from_source_inference` | Source inference has `depth=2` → new inference stored with `depth=3` |
| `test_eager_pass_raises_on_non_json_response` | Ollama returns plain text → `InferenceParseError` raised |
| `test_eager_pass_request_sends_format_json` | Captured Ollama request body contains `"format": "json"` |
| `test_eager_pass_request_sends_think_false` | Captured request body has `"think": false` |

**Depth computation helper**

| Test | Asserts |
|------|---------|
| `test_compute_depth_returns_one_for_empty_source_inference_ids` | `source_inference_ids=[]` → 1 |
| `test_compute_depth_returns_source_depth_plus_one` | Source at depth 3 → returns 4 |
| `test_compute_depth_takes_max_of_multiple_sources` | Sources at depth 2 and depth 4 → returns 5 |
| `test_compute_depth_skips_unknown_source_ids` | Unknown id in source list → skipped; remaining source used for calculation |
| `test_compute_depth_returns_one_when_all_sources_unknown` | All ids unknown → returns 1 (fallback) |

**Revalidation prompt**

| Test | Asserts |
|------|---------|
| `test_revalidation_prompt_includes_inference_statement` | Inference statement appears verbatim in the built prompt |
| `test_revalidation_prompt_includes_inference_derivation` | Original derivation text appears in the prompt |
| `test_revalidation_prompt_includes_current_facts` | All current fact key-value pairs appear in the prompt |
| `test_revalidation_prompt_includes_other_active_inferences` | Other active inferences appear under "context only" |

**Revalidation execution**

| Test | Asserts |
|------|---------|
| `test_revalidate_returns_true_when_inference_holds` | LLM returns `{"holds": true, ...}` → returns `True` |
| `test_revalidate_returns_false_when_inference_does_not_hold` | LLM returns `{"holds": false, ...}` → returns `False` |
| `test_revalidate_defaults_to_true_on_parse_error` | LLM returns non-JSON → returns `True` (conservative default) |

**Cascade on Fact edit**

| Test | Asserts |
|------|---------|
| `test_cascade_edit_marks_directly_dependent_inference_stale` | Changed fact id in `source_fact_ids` → inference marked stale in DB |
| `test_cascade_edit_leaves_unrelated_inference_active` | Inference not depending on changed fact → status unchanged |
| `test_cascade_edit_transitively_marks_chained_inference_stale` | Chain: Inf-A depends on changed fact, Inf-B depends on Inf-A → Inf-A and Inf-B both marked stale |
| `test_cascade_edit_returns_stale_inferences` | Return value contains all newly-stale `Inference` objects |
| `test_cascade_edit_skips_already_stale_inferences` | Pre-existing stale inference is not re-processed |
| `test_cascade_edit_does_not_mark_if_revalidation_returns_true` | LLM says inference still holds → not marked stale |

**Cascade on Fact delete**

| Test | Asserts |
|------|---------|
| `test_cascade_delete_marks_directly_dependent_inference_invalidated` | Deleted fact id in `source_fact_ids` → inference marked invalidated |
| `test_cascade_delete_leaves_unrelated_inference_active` | Inference not referencing deleted fact → status unchanged |
| `test_cascade_delete_transitively_marks_chained_inference_invalidated` | Chain: Inf-A → deleted fact, Inf-B → Inf-A → both marked invalidated |
| `test_cascade_delete_returns_all_invalidated_inferences` | Return list contains all newly-invalidated `Inference` objects |
| `test_cascade_delete_no_llm_call_made` | No Ollama request made during cascade delete (pure DB) |

---

#### `test_prompt_builder.py` (additions to existing file)

| Test | Asserts |
|------|---------|
| `test_inferences_section_included_when_inferences_present` | Non-empty `inferences` arg → `## Your Inferences` section appears in prompt |
| `test_inference_statement_appears_verbatim` | Inference statement text appears in the prompt |
| `test_inference_derivation_appears_in_prompt` | Inference derivation text also included |
| `test_inferences_section_absent_when_no_inferences` | `inferences=[]` → no inferences header in prompt |
| `test_inferences_section_absent_when_inferences_is_none` | `inferences=None` → no inferences header in prompt |
| `test_multiple_inferences_all_appear` | Three inferences in list → all three statements present |

---

#### `test_evaluator_service.py` (additions to existing file)

| Test | Asserts |
|------|---------|
| `test_evaluator_prompt_includes_established_inferences` | Active inferences appear in the prompt under their own section |
| `test_evaluator_prompt_no_inferences_uses_fallback` | Empty inference list → `(no inferences established yet)` fallback line |
| `test_evaluator_prompt_includes_inference_ids` | Inference ids appear in the prompt so the LLM can cite them in source lists |
| `test_evaluator_accepts_inferences_parameter` | `run_evaluator(inferences=[...])` does not raise |

---

#### `test_chat_service.py` (additions to existing file)

| Test | Asserts |
|------|---------|
| `test_run_turn_loads_inferences_for_character` | When active inferences exist, `get_inferences` is called and inferences are included in the system message |
| `test_run_turn_system_message_includes_inference_text` | System message content contains a known inference statement |
| `test_lazy_inference_depth_computed_before_storing` | Lazy-discovered inference with source at depth 2 is stored with `depth=3` |
| `test_lazy_inference_at_max_depth_is_stored` | Inference at exactly `MAX_INFERENCE_DEPTH` is stored |
| `test_lazy_inference_exceeding_depth_cap_not_stored` | Depth > max → no inference row in DB |
| `test_evaluator_called_with_inferences` | Second (evaluator) Ollama call's request includes inference statements from DB |

---

### Integration tests — `tests/integration/`

#### `test_inferences_repo.py` (additions to existing file)

| Test | Asserts |
|------|---------|
| `test_update_inference_status_to_stale` | `update_inference_status(..., "stale")` → DB row has `status="stale"` |
| `test_update_inference_status_to_invalidated` | `update_inference_status(..., "invalidated")` → DB row has `status="invalidated"` |
| `test_update_inference_status_to_active` | `update_inference_status(..., "active")` → restores a stale inference |
| `test_update_inference_status_returns_updated_inference` | Return value is an `Inference` with the new status |
| `test_update_inference_status_nonexistent_raises` | `NotFoundError` raised for unknown id |
| `test_get_inference_by_id_returns_correct_inference` | Returns the matching inference |
| `test_get_inference_by_id_returns_none_for_unknown` | `get_inference(9999)` returns `None` |
| `test_delete_inference_removes_row` | After `delete_inference`, row absent from `get_inferences(status="active")` |
| `test_delete_inference_nonexistent_raises` | `NotFoundError` raised |
| `test_get_inferences_stale_status_filter` | `get_inferences(status="stale")` returns only stale rows |
| `test_get_inferences_invalidated_status_filter` | `get_inferences(status="invalidated")` returns only invalidated rows |

---

#### `test_api_inference_generation.py` (new file)

Fixtures: a character with a set of facts and a pre-existing active inference. The Ollama mock returns a fixed JSON array for the eager pass, and a `{"holds": false, "reason": "..."}` JSON for revalidation calls.

**Eager pass endpoint**

| Test | Asserts |
|------|---------|
| `test_generate_inferences_returns_200` | POST returns 200 |
| `test_generate_inferences_returns_new_inferences_list` | Response JSON has `new_inferences` key with a list |
| `test_generate_inferences_stores_to_db` | After POST, new inference rows exist in DB with `status="active"` |
| `test_generate_inferences_unknown_character_returns_404` | POST for non-existent character_id → 404 |
| `test_generate_inferences_respects_depth_cap` | Ollama mock returns deep inference → not stored; not in response |
| `test_generate_inferences_applies_breadth_cap` | Ollama mock returns 7 inferences → only first 5 in response |
| `test_generate_inferences_empty_response_returns_empty_list` | Ollama returns `[]` → 200 with `new_inferences: []` |
| `test_generate_inferences_on_parse_error_returns_warning` | Ollama returns non-JSON → 200 with `new_inferences: []` and `warning` field |

**Revalidate endpoint**

| Test | Asserts |
|------|---------|
| `test_revalidate_returns_200` | POST returns 200 |
| `test_revalidate_returns_stale_inferences` | Response JSON has `stale_inferences` key |
| `test_revalidate_marks_stale_in_db` | After POST, affected inference has `status="stale"` in DB |
| `test_revalidate_does_not_affect_unrelated_inferences` | Inference with no dependency on changed fact still `active` in DB |
| `test_revalidate_unknown_character_returns_404` | POST for non-existent character_id → 404 |
| `test_revalidate_unknown_fact_returns_404` | POST with `changed_fact_id` not in DB → 404 |

**Inference management endpoints**

| Test | Asserts |
|------|---------|
| `test_delete_inference_returns_204` | DELETE → 204 |
| `test_delete_inference_removes_from_db` | After DELETE, row absent from `get_inferences` |
| `test_delete_inference_unknown_id_returns_404` | DELETE for non-existent inference_id → 404 |
| `test_patch_inference_status_to_active_returns_200` | PATCH `{status: "active"}` → 200 with updated `Inference` |
| `test_patch_inference_status_to_stale_returns_200` | PATCH `{status: "stale"}` → 200 |
| `test_patch_inference_status_updates_db` | After PATCH, row in DB has new status |
| `test_patch_inference_unknown_id_returns_404` | PATCH for non-existent inference_id → 404 |

---

#### `test_api_facts.py` (additions and updates to existing file)

| Test | Asserts |
|------|---------|
| `test_delete_fact_returns_200` | DELETE → 200 (updated from previous 204 assertion) |
| `test_delete_fact_response_has_invalidated_inferences_key` | Response JSON contains `invalidated_inferences` list |
| `test_delete_fact_cascade_marks_dependent_inference_invalidated` | Inference depending on deleted fact → appears in `invalidated_inferences` and has `status="invalidated"` in DB |
| `test_delete_fact_cascade_leaves_unrelated_inference_active` | Inference with no dependency on deleted fact stays `active` in DB |
| `test_delete_fact_no_dependents_returns_empty_list` | No inferences depend on deleted fact → `invalidated_inferences: []` |

---

#### `test_api_chat.py` (additions to existing file)

| Test | Asserts |
|------|---------|
| `test_send_message_system_message_includes_inferences` | When the character has active inferences, Ollama request system message contains inference statement text |
| `test_send_message_no_inferences_section_when_none_exist` | When no inferences exist, system message does not contain the `## Your Inferences` header |
| `test_send_message_lazy_logical_inference_stored_with_depth` | `new_inference_logical` evaluator mock with `source_inference_ids=[existing_id]` → stored inference has `depth=2` |
| `test_send_message_lazy_inference_at_max_depth_is_stored` | Lazy inference at exactly depth 5 → inference row written to DB |
| `test_send_message_lazy_inference_exceeding_depth_not_stored` | Lazy inference at depth 6 (when cap=5) → no new inference row in DB |

---

### Frontend tests — `tests/frontend/chat.test.js` (additions)

The stale/invalidated inference flows are surfaced via REST (not SSE), so `buildNotificationFromSidechannel` and `sseStateToLabel` are unchanged. The additions cover the four new API helper functions.

| Test | Asserts |
|------|---------|
| `test_apiGenerateInferences_posts_to_correct_url` | `fetch` called with `POST /api/characters/{id}/inferences/generate` |
| `test_apiGenerateInferences_uses_post_method` | Request method is `POST` |
| `test_apiRevalidateInferences_posts_to_correct_url` | `fetch` called with `POST /api/characters/{id}/inferences/revalidate` |
| `test_apiRevalidateInferences_sends_changed_fact_id_in_body` | Request body JSON contains `changed_fact_id` |
| `test_apiDeleteInference_sends_delete_to_correct_url` | `fetch` called with `DELETE /api/characters/{id}/inferences/{inference_id}` |
| `test_apiPatchInferenceStatus_sends_patch_to_correct_url` | `fetch` called with `PATCH /api/characters/{id}/inferences/{inference_id}` |
| `test_apiPatchInferenceStatus_sends_status_in_body` | Request body JSON contains the provided `status` string |

---

## Not in Scope for Phase 3

The following are intentionally deferred:

- **Experiences** — Phase 4. The `experiences` table exists but stays empty. `experience_update` verdict continues to be coerced to `pass`.
- **End-of-session evaluator pass (closing journal + Experience proposals)** — Phase 4.
- **Token counting (`prompt_eval_count` / `eval_count`)** — Phase 5a. Ollama response metadata is returned by the client but not persisted.
- **Context budget tracking and compression** — Phase 5a/5b.
- **Segment boundary logic** — Phase 5b. All messages remain in the single `session_start` segment.
- **Modelfile export** — Phase 6 stretch. Even if Facts + Inferences grow large, baking them into a Modelfile is deferred.
- **Phone-responsive layout** — Phase 6.
- **Playwright E2E tests** — deferred until UI is stable enough to warrant the setup cost.
- **Iterative / multi-hop eager pass** — the eager pass makes a single LLM call per trigger. Chains of depth > 1 emerge naturally across multiple fact edits; no iterative pass is implemented in Phase 3.
