# Phase 4 Implementation Plan — Fact Categories, Mutability Levels, and Inference Promotion

## Goals (from plan.md)

- Add `category` and `mutability` metadata to Facts so the evaluator can treat fluid things like mood differently from immutable things like height
- Section the character system prompt by category (`User`, `Character`, `Setting`) so the LLM has clear context about whose facts are whose
- Update the evaluator prompt to include category and mutability labels, and instruct it to surface high/low-mutability fact changes as `implication` rather than `contradiction`
- Add a `POST /api/characters/{id}/inferences/{inference_id}/promote` endpoint that elevates an Inference to a Fact and removes the source Inference
- Update the fact creation form and sidechannel fact list in the UI to expose category and mutability controls
- Add a "Promote to Fact" button to each inference row in the sidechannel

**Deliverable:** The evaluator stops flagging mood swings and clothing choices as hard contradictions. The fact model becomes expressive enough to handle the real diversity of what can be known. Characters can naturally evolve their emotional state and situational details without triggering regeneration loops.

---

## Task List

### T1 — Schema changes

The `facts` table gains two new columns and a revised uniqueness constraint. No migration code is needed — delete the existing `memories.db` file before starting the server for the first time on this branch. `init_db` creates a fresh database from `_DDL`.

**T1.1 — Update `_DDL` in `database.py`**

Replace the `facts` table definition with:

```sql
CREATE TABLE IF NOT EXISTS facts (
    id               INTEGER PRIMARY KEY,
    character_id     INTEGER REFERENCES characters(id),
    key              TEXT NOT NULL,
    value            TEXT NOT NULL,
    category         TEXT NOT NULL DEFAULT 'character',
    mutability       TEXT NOT NULL DEFAULT 'immutable',
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(character_id, category, key)
);
```

No changes to `init_db` beyond this DDL update.

---

### T2 — Model update

**T2.1 — `Fact` model in `models/__init__.py`**

Add `category` and `mutability` with `Literal` types for validation at the model boundary:

```python
from typing import Literal

class Fact(BaseModel):
    id: int
    character_id: int
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    mutability: Literal["immutable", "low", "high"] = "immutable"
    created_at: datetime
```

Using `Literal` here (rather than plain `str`) means Pydantic will reject invalid category or mutability values at the point data enters the system — both from the API and from DB rows. This gives a clear validation error rather than silent storage of an invalid string.

---

### T3 — Database layer changes

**T3.1 — Updated `create_fact`**

```python
async def create_fact(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    key: str,
    value: str,
    category: str = "character",
    mutability: str = "immutable",
) -> Fact:
```

The INSERT statement gains the two new columns:

```sql
INSERT INTO facts (character_id, key, value, category, mutability)
VALUES (?, ?, ?, ?, ?)
```

All existing callers omit `category` and `mutability`, so they continue to receive the defaults without modification. The `accept_implication` endpoint in `implication.py` calls `create_fact` for implied-but-not-yet-established keys; the defaults (`character`, `immutable`) are appropriate for user-accepted implications.

**T3.2 — Updated `update_fact` (ID-based)**

With `UNIQUE(character_id, category, key)`, the key alone no longer identifies a fact within a character — a user `name` and a character `name` are distinct rows. All mutation functions therefore address facts by their primary key `id`, which is always available to the frontend (returned in every create/list response).

The old signature `update_fact(db, *, character_id, key, value)` is replaced with:

```python
async def update_fact(
    db: aiosqlite.Connection,
    *,
    fact_id: int,
    value: str,
    category: str | None = None,
    mutability: str | None = None,
) -> Fact:
```

`value` stays required. `category` and `mutability` are optional: only non-`None` values are included in the `SET` clause. `character_id` is dropped from the signature — the ID is globally unique; ownership validation happens in the router before calling the DB function.

Dynamic SQL:

```python
updates = ["value = ?"]
params: list[Any] = [value]
if category is not None:
    updates.append("category = ?")
    params.append(category)
if mutability is not None:
    updates.append("mutability = ?")
    params.append(mutability)
params.append(fact_id)
cursor = await db.execute(
    f"UPDATE facts SET {', '.join(updates)} WHERE id = ?",
    tuple(params),
)
```

**T3.3 — New `patch_fact` for metadata-only updates (ID-based)**

```python
async def patch_fact(
    db: aiosqlite.Connection,
    *,
    fact_id: int,
    category: str | None = None,
    mutability: str | None = None,
) -> Fact:
```

Raises `ValueError` if neither `category` nor `mutability` is provided. Raises `NotFoundError` if `fact_id` does not exist.

**T3.4 — Updated `delete_fact` (ID-based)**

The old signature `delete_fact(db, *, character_id, key)` is replaced with:

```python
async def delete_fact(db: aiosqlite.Connection, *, fact_id: int) -> None:
```

**T3.5 — New `get_fact_by_category_key` for internal lookups**

The `accept_implication` endpoint in `implication.py` looks up a fact by `(character_id, category, key)` when handling a duplicate-key IntegrityError. A dedicated helper avoids raw SQL in the router:

```python
async def get_fact_by_category_key(
    db: aiosqlite.Connection,
    *,
    character_id: int,
    category: str,
    key: str,
) -> Fact | None:
```

Returns `None` if no matching row exists.

---

### T4 — API changes

**T4.1 — Updated `POST /api/characters/{character_id}/facts`**

New request body:

```python
class _CreateBody(BaseModel):
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    mutability: Literal["immutable", "low", "high"] = "immutable"
```

Passes `category` and `mutability` through to `create_fact`. Response is the full `Fact` model (which now includes these fields).

**T4.2 — Updated `PUT /api/characters/{character_id}/facts/{fact_id}`**

The URL parameter changes from `{key}` to `{fact_id}` (integer). Key is no longer unique per character; the fact's primary key is the stable address for mutations.

Extended request body:

```python
class _UpdateBody(BaseModel):
    value: str
    category: Literal["user", "character", "setting"] | None = None
    mutability: Literal["immutable", "low", "high"] | None = None
```

Router logic:
1. Load the fact by `fact_id`; verify it belongs to `character_id` (404 otherwise).
2. Call `update_fact(db, fact_id=fact_id, value=body.value, category=body.category, mutability=body.mutability)`.
3. Return 200 with the updated `Fact`.

`value` stays required. `category` and `mutability` are optional — if omitted, the existing values are preserved.

**Cascade note**: cascade revalidation is still triggered separately via `POST /inferences/revalidate` (unchanged). Changing only `category` or `mutability` does not require revalidation; the frontend only calls `/revalidate` when the value changes.

**T4.3 — New `PATCH /api/characters/{character_id}/facts/{fact_id}`**

For metadata-only updates (mutability icon click). URL uses `{fact_id}` for the same reason.

```python
class _PatchBody(BaseModel):
    category: Literal["user", "character", "setting"] | None = None
    mutability: Literal["immutable", "low", "high"] | None = None
```

Router logic:
1. Load the fact by `fact_id`; verify it belongs to `character_id` (404 otherwise).
2. If both fields are `None`, return 422.
3. Call `patch_fact(db, fact_id=fact_id, category=body.category, mutability=body.mutability)`.
4. Return 200 with the updated `Fact`.

**T4.4 — Updated `DELETE /api/characters/{character_id}/facts/{fact_id}`**

The URL parameter changes from `{key}` to `{fact_id}`. The existing endpoint fetches the fact's `id` before deletion anyway (to run the cascade); switching the path parameter to `fact_id` eliminates that extra SELECT.

Router logic:
1. Load the fact by `fact_id`; verify it belongs to `character_id` (404 otherwise).
2. Call `delete_fact(db, fact_id=fact_id)`.
3. Call `cascade_on_fact_delete(db, character_id, fact_id)`.
4. Return 200 with `{"invalidated_inferences": [...]}`.

**T4.5 — New `POST /api/characters/{character_id}/inferences/{inference_id}/promote`**

Request body:

```python
class _PromoteBody(BaseModel):
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    mutability: Literal["immutable", "low", "high"] = "immutable"
```

Endpoint logic:

1. Load and validate character (404 if not found).
2. Load all inferences for the character (`status="all"`); validate `inference_id` belongs to this character (404 if not).
3. Call `create_fact(db, character_id=..., key=..., value=..., category=..., mutability=...)`.
   - On `aiosqlite.IntegrityError` (same `(character_id, category, key)` tuple already exists): raise 409 with message `"A {category} Fact with key '{key}' already exists"`.
4. Check for downstream inferences whose `source_inference_ids` contains `inference_id`. Mark each one `stale` via `update_inference_status`. This is a conservative cascade: promoting an inference removes its derivation anchor, so downstream inferences can no longer trace back to original sources through it. The user can review and restore them if the promoted Fact provides equivalent grounding.
5. Call `delete_inference(db, inference_id)`.
6. Return 201 with the created `Fact`.

Response body:

```json
{
  "fact": { ...Fact fields... },
  "stale_inferences": [ ...Inference objects that were marked stale... ]
}
```

Returning `stale_inferences` in the promotion response allows the frontend to immediately surface any downstream inferences that need attention, without a separate request.

**Mounted in `main.py`**: this endpoint belongs in `inferences.py` (already mounted at `/api/characters`).

**T4.5 — No changes to `DELETE /api/characters/{character_id}/facts/{key}`**

The cascade on Fact delete already handles downstream inferences correctly. Promoting an inference does not delete any facts, so this endpoint is unaffected.

---

### T5 — Prompt builder update

**T5.1 — `build_system_prompt` groups facts by category**

Signature is unchanged. The implementation changes to group facts into category sections:

```
You are {name}. Stay in character at all times.

## Facts About The User
These are established truths about the person you are talking with.

name: Jon
location: Chicago

## Facts About You (Character)
These are established truths about you. Never contradict them and never invent
details that are not listed here.

occupation: surgeon
age: 33
mood: cheerful [fluid — may change within a session]
clothing: dark coat [low-mutability — changes infrequently and with context]

## Setting
These are established truths about the current environment.

time_of_day: evening
```

Rules for rendering:
- Omit any section for which no facts exist (no empty headers).
- Category order is always: `user` → `character` → `setting`.
- Within each section, facts appear in creation order (ascending by `id`).
- `immutable` facts show no mutability annotation (it is the default; annotating it adds noise).
- `low` facts show `[low-mutability — changes infrequently and with context]`.
- `high` facts show `[fluid — may change within a session]`.

The mutability annotations prime the character LLM to understand when it has some freedom to shift state (mood, clothing) vs. when it must not (birthdate, height).

**T5.2 — Updated `build_evaluator_prompt` includes category and mutability labels**

The fact listing in the evaluator prompt changes from:

```
[1] occupation: surgeon
```

to:

```
[1] occupation: surgeon  (category: character, mutability: immutable)
[2] mood: cheerful  (category: character, mutability: high)
[3] location: Chicago  (category: setting, mutability: low)
```

This gives the evaluator full context to judge whether a change is a contradiction or a plausible update.

**T5.3 — Updated evaluator verdict instructions for mutability**

The "Your Task" section of the evaluator prompt is updated with explicit mutability guidance:

```
## Mutability Rules
These rules govern how you classify violations against established Facts:

- IMMUTABLE facts: any response that contradicts an immutable Fact is a `contradiction`
  regardless of context. Do not surface these as implications — the value cannot change.
  Examples: height, birthdate, eye colour, bone structure.

- LOW-mutability facts: these change infrequently and only with clear narrative context
  (e.g., the character changed their clothes, moved to a new city). If the character's
  response implies a different value for a low-mutability Fact, return `implication` (not
  `contradiction`) — the change is plausible but needs user confirmation. Include a
  violation entry with the new implied value as `suggested_fact`.

- HIGH-mutability facts: these can change fluidly within a session (mood, emotional state,
  immediate desires). If the character's response reflects a different value for a
  high-mutability Fact, return `implication` — the change is expected and natural.
  Include a violation entry with the new implied value as `suggested_fact`. In the
  violation description, note that this is a high-mutability change: e.g.,
  "Mood appears to have shifted from 'cheerful' to 'anxious' (high-mutability fact)".

When building a `suggested_fact`, always include a `category` field that reflects whose
fact it is:
- `"character"` — something about the character themselves (their own clothing, mood, etc.)
- `"user"` — something about the person they are talking with (their clothing, appearance,
  things they have said about themselves)
- `"setting"` — something about the current environment or situation

Example: if the character says "I like your blue jacket", the suggested fact is about the
user, not the character:
  `{"key": "jacket_colour", "value": "blue", "category": "user"}`

If the category is unclear, default to `"character"`.

The verdict priority order is:
1. `contradiction` — ONLY for immutable Fact violations
2. `implication` — for low- or high-mutability Fact changes, or invented specific details
3. `new_inference_logical` — something derivable, not yet stored
4. `new_inference_probabilistic` — likely tendency, not strictly derivable
5. `pass` — only when every claim is grounded
```

This change has significant downstream effects. Responses that previously triggered the contradiction loop (e.g., character mentions being in a foul mood when the fact says "cheerful") will now return `implication` and be delivered with an ungrounded badge. The user can accept the mood shift (updating the Fact) or ignore it.

**T5.4 — No changes to `run_evaluator` signature**

The new fact fields (`category`, `mutability`) are properties of the `Fact` model, which is already passed through to the evaluator. No signature changes are needed — the updated prompt builder reads the new fields automatically from the `Fact` objects it receives.

---

### T6 — Accept-implication update: category forwarding

The evaluator now includes a `category` field in every `suggested_fact` (see T5.3). The `_AcceptImplicationBody` gains a `category` field so the frontend can forward it:

```python
class _AcceptImplicationBody(BaseModel):
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    regenerate: bool = True
```

The default of `"character"` preserves backward compatibility with any caller that omits it, and is the correct fallback since most implied facts are about the character. The evaluator will supply the correct category for user-facing details (e.g., clothing, appearance) and setting details.

The updated fact creation flow in `implication.py`:

```python
try:
    fact = await create_fact(
        db, character_id=session.character_id,
        key=body.key, value=body.value, category=body.category
        # mutability defaults to "immutable" — user can change it afterward
    )
except aiosqlite.IntegrityError:
    # A (character_id, category, key) conflict — look up by category+key, then update by ID.
    existing = await get_fact_by_category_key(
        db, character_id=session.character_id, category=body.category, key=body.key
    )
    if existing is None:
        raise  # shouldn't happen; re-raise if something else went wrong
    fact = await update_fact(db, fact_id=existing.id, value=body.value)
    # category and mutability are intentionally not passed — preserve the existing values
```

Category and mutability are never overwritten when updating an existing fact through implication acceptance. A `high`-mutability user fact (e.g., what the user is wearing today) keeps that classification across multiple value updates.

---

### T7 — Frontend updates

**T7.1 — Fact creation form**

Add two `<select>` elements below the existing key/value inputs:

| Control | Options | Default |
|---------|---------|---------|
| Category | User / Character / Setting | Character |
| Mutability | 🔒 Immutable / 📌 Settable / 💧 Fluid | 🔒 Immutable |

The option icons (🔒 📌 💧) appear in the dropdown and also on the row icons, giving visual consistency.

**T7.2 — Fact list grouped by category**

The sidechannel fact list renders three sections with headers:

```
USER
 [🔒] name          Jon
 [🔒] hometown      Chicago

CHARACTER
 [🔒] occupation    surgeon
 [🔒] age           33
 [💧] mood          cheerful
 [📌] clothing      dark coat

SETTING
 [📌] location      Chicago
```

Sections with no facts are omitted. The `[icon]` is a clickable button that opens an inline dropdown for the mutability choices. Clicking elsewhere closes the dropdown.

The icon legend:
- `🔒` immutable
- `📌` low (settable, changes with context)
- `💧` high (fluid, changes freely)

**T7.3 — Mutability inline dropdown**

When the user clicks a mutability icon:
1. A small `<div>` overlay appears below the icon with three rows: `🔒 Immutable`, `📌 Settable`, `💧 Fluid`.
2. The current value is highlighted.
3. Selecting a different value calls `apiPatchFactMutability(characterId, key, newMutability)`.
4. On success, the icon updates in place without a full fact list reload (optimistic update via Vue reactive state).

**T7.4 — Inference rows: collapsed by default with expandable derivation**

Inference rows are collapsed by default. Each row shows only the promote icon and the inference statement — no derivation text. This keeps the list scannable when many inferences are present.

```
⬆  Born in 1993
⬆  Likely works long hours
⬆  Was a teenager during 9/11
```

Clicking the statement text toggles an expansion panel that slides down beneath the row, revealing the full derivation in plain prose:

```
⬆  Born in 1993
     The character is 33 years old [fact #2] and the current year is 2026
     [fact #7], so logically the character must have been born in 1993.
     Type: logical
```

The derivation text is the `derivation` field stored in the DB (written by the LLM at inference creation time). Fact IDs referenced in `source_fact_ids` are surfaced as `[fact #N]` links; clicking one could highlight the corresponding fact in the fact list (nice-to-have, not required for Phase 4). The expansion state is per-row and tracked in Vue reactive state — navigating away and back resets all rows to collapsed.

The `⬆` icon sits in the leftmost column, fixed-width, so all statement text starts at the same indent regardless of expansion state. The expansion panel indents further right, visually subordinate to the statement.

**T7.5 — Inference Promote to Fact inline form**

Clicking `⬆` opens a small inline form that drops in below the row (not a modal, so the rest of the list stays visible):
- Key: editable text field (initially empty — the user names the fact)
- Value: editable text field (pre-filled with the inference `statement`)
- Category: same dropdown as the mutability icon panel — User / Character / Setting (default: `character`)
- Mutability: 🔒 Immutable / 📌 Settable / 💧 Fluid (default: 🔒 Immutable)
- Submit button labeled "Promote" and a Cancel link that collapses the form

On submit, calls `apiPromoteInference(characterId, inferenceId, key, value, category, mutability)`. On 201 success: collapses the form, removes the inference row, and adds the new fact into the appropriate category group in the fact list. On 409: shows an inline error directly below the Key field — "A fact with this key already exists".

**T7.6 — New and updated API helpers in `chat.js`**

```javascript
/**
 * @param {number} characterId
 * @param {string} key
 * @param {string} value
 * @param {string} [category='character']
 * @param {string} [mutability='immutable']
 * @returns {Promise<Response>}
 */
export function apiCreateFact(characterId, key, value, category = 'character', mutability = 'immutable') {
  return fetch(`/api/characters/${characterId}/facts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value, category, mutability }),
  });
}

/**
 * @param {number} characterId
 * @param {string} key
 * @param {string} mutability
 * @returns {Promise<Response>}
 */
export function apiPatchFactMutability(characterId, key, mutability) {
  return fetch(`/api/characters/${characterId}/facts/${encodeURIComponent(key)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mutability }),
  });
}

/**
 * @param {number} characterId
 * @param {string} key
 * @param {string} category
 * @returns {Promise<Response>}
 */
export function apiPatchFactCategory(characterId, key, category) {
  return fetch(`/api/characters/${characterId}/facts/${encodeURIComponent(key)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ category }),
  });
}

/**
 * @param {number} characterId
 * @param {number} inferenceId
 * @param {string} key
 * @param {string} value
 * @param {string} [category='character']
 * @param {string} [mutability='immutable']
 * @returns {Promise<Response>}
 */
export function apiPromoteInference(characterId, inferenceId, key, value, category = 'character', mutability = 'immutable') {
  return fetch(`/api/characters/${characterId}/inferences/${inferenceId}/promote`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ key, value, category, mutability }),
  });
}
```

No new SSE event types are needed for Phase 4. The evaluator returns `implication` for mutability-change events using the existing sidechannel mechanism. The existing `buildNotificationFromSidechannel` handler for `implication` already displays the violation description and suggested_fact value to the user, which is sufficient. The only distinction needed is in the violation `description` field (written by the evaluator, e.g., "Mood appears to have shifted from 'cheerful' to 'anxious' (high-mutability fact)").

---

## Architecture Decisions

### D1 — PATCH endpoint for metadata-only fact updates

Adding a `PATCH /facts/{key}` endpoint is cleaner than making `value` optional in the PUT body. The UI use case for PATCH is purely the mutability icon dropdown — the user is changing metadata, not the fact's established value. Making `value` optional in PUT would require all existing callers to handle a nullable value, and would blur the semantic distinction between "update the established truth" and "update the classification of the established truth." PATCH's partial-update semantics match the intent precisely.

The downside is a second endpoint to maintain. This is acceptable given the distinct semantics and the clarity it provides in tests.

### D2 — Cascade on inference promotion: mark downstream stale (not invalidated)

When an inference is promoted to a Fact, its downstream inferences (those with this inference ID in `source_inference_ids`) lose their derivation anchor. Two options:

- **Invalidate**: downstream inferences are broken — the source inference no longer exists.
- **Stale**: downstream inferences may still be valid, but should be reviewed.

`stale` is correct here. If inference A ("born in 1993") is promoted to a Fact and inference B ("was a teenager during 9/11") derives from A, inference B is still valid — the underlying information is now a stronger Fact rather than a weaker Inference. Marking it `stale` prompts the user to review it and restore it as `active` (or delete it if they disagree), without automatically discarding valid reasoning.

This is different from Fact deletion (`cascade_on_fact_delete`), which marks inferences `invalidated` because the source information is gone entirely.

### D3 — Default mutability for accepted implications is `immutable`

When the user accepts an implication for a key that does not yet exist as a Fact, the new Fact is created with `mutability="immutable"`. This is the conservative default. If the implied Fact should be high-mutability (e.g., mood), the user can patch the mutability afterward via the icon dropdown. 

An alternative is to ask the user for mutability at the moment of acceptance, but that adds friction to the common case (most implied facts — names, relationships, biographical details — should be immutable). The accept/edit/ignore notification is already a two-step interaction; adding a third choice (mutability) would make it more cumbersome.

### D4 — No change to `experience_update` coercion

The `experience_update` verdict is still coerced to `pass` in `run_evaluator`. Phase 4 does not implement Experiences. The coercion remains.

### D5 — Inference promotion is in `inferences.py`, not a new file

The promote endpoint fits naturally alongside the existing inference management endpoints (`DELETE`, `PATCH`). Adding it to `inferences.py` avoids an extra router file and keeps all inference lifecycle operations in one place.

### D6 — The evaluator's `suggested_fact` carries the NEW value for mutability changes

For a `high`-mutability fact change (e.g., mood: "cheerful" → "anxious"), the evaluator returns `implication` with:

```json
{
  "type": "implication",
  "description": "Mood appears to have shifted from 'cheerful' to 'anxious' (high-mutability fact)",
  "suggested_fact": { "key": "mood", "value": "anxious" }
}
```

The `suggested_fact.value` is the NEW value (what the character expressed), not the old one. When the user accepts, `accept_implication` calls `update_fact(..., value="anxious")`, which updates the existing Fact in place while preserving its `mutability="high"` and `category="character"`. This is the correct behavior — the high-mutability Fact's value updates, but its classification is unchanged.

### D7 — `build_system_prompt` renders mutability annotations only for non-immutable facts

`immutable` is the default and the most common case. Annotating every immutable fact with `[immutable]` would clutter the prompt. Only `low` and `high` facts get annotations. The character LLM does not need to know that the absence of an annotation means immutable — it just needs to know when something CAN change.

### D8 — Fact key URL-encoding

Fact keys appear in URL paths (`/facts/{key}`). The new `apiPatchFactMutability` and `apiPatchFactCategory` helpers use `encodeURIComponent(key)` to handle keys that contain spaces or special characters. Existing helpers that do the same thing (e.g., `apiRevalidateInferences`) should be audited and updated if they do not already URL-encode path segments.

---

## Potential Gotchas

**G1 — Evaluator prompt engineering is critical for mutability correctness**

The mutability logic lives entirely in the evaluator prompt. If the LLM misclassifies a `high`-mutability fact change as `contradiction`, the response will be regenerated unnecessarily. If it misclassifies an `immutable` fact change as `implication`, the response will be delivered without correction. The prompt in T5.3 is the primary safeguard. If misclassification is observed in practice, the fix is prompt engineering, not code changes.

**G2 — Same-key facts in different categories: why the schema change, not a key prefix**

Both the character and the user will almost always have a `name` fact. The old `UNIQUE(character_id, key)` constraint would have forced clunky workarounds (`character_name` / `user_name`) that the user would need to discover themselves, since the UI cleanly separates fact categories and gives no hint that the keys must be globally distinct.

Two approaches were considered:

1. **Implicit backend prefix** (`user:name` stored, `name` displayed): enforces convention system-wide and could help the LLM by making keys self-annotating. Rejected because it creates a leaky abstraction — the API surface, cascade queries, implication acceptance, and evaluator prompt all have to strip or apply the prefix, and the convention is invisible to API callers. It also offers no benefit over the section-header approach: `## Facts About The User` / `## Facts About You (Character)` already gives the LLM unambiguous context for distinguishing same-named facts.

2. **Schema change** `UNIQUE(character_id, category, key)`: the right semantic model. A key is unique within its category, which matches the user's mental model (they create facts in named sections). All mutations already use the fact's `id` at the DB layer, so no compound-key lookups are needed.

The schema change approach is implemented (see T1.1, T3.2–T3.5, T4.2–T4.4).

**G3 — Downstream inference cascade on promotion may be aggressive**

If an inference with many descendants is promoted, all descendants are marked `stale`. In a deep inference chain (A → B → C → D), promoting A marks B, C, and D all stale simultaneously — even though B, C, and D may all still be valid (because the promoted Fact provides equivalent grounding). The user review burden can be significant. For Phase 4, this is acceptable. A future improvement could revalidate downstream inferences using the LLM (similar to `cascade_on_fact_edit`) and only mark those that the LLM says are broken.

**G4 — Implications can be about the user, not only the character**

The character can invent or observe details about the person they are talking with — commenting on the user's jacket, shoes, or something the user mentioned earlier. These are `user`-category facts, not `character`-category ones. The evaluator is instructed to include a `category` field in every `suggested_fact` (see T5.3), and `_AcceptImplicationBody` forwards it to `create_fact` and `get_fact_by_category_key` (see T6).

The default category in `_AcceptImplicationBody` is `"character"` for backward compatibility. If the evaluator omits `category` from a `suggested_fact` (old prompt, LLM non-compliance), the frontend falls back to `"character"` — which may be wrong for user-detail implications but is safe (no data is lost; the user can re-categorise the fact afterward via the PATCH endpoint). The evaluator prompt must be considered the primary safeguard here.

**G5 — Inference promotion responds with `fact` + `stale_inferences`**

The response body is a combined object, not just a `Fact`. FastAPI's `response_model` parameter cannot use a plain `Fact` model for this endpoint. Use `response_model=None` and return a raw `dict`, or define a `_PromoteResponse` Pydantic model. The latter is preferred for documentation and validation.

**G6 — `inference_id` must belong to the character**

The promote endpoint receives `character_id` and `inference_id` as path parameters. The inference must be validated to belong to this character (not just that it exists). The existing delete and patch inference endpoints use `get_inferences(db, character_id, status="all")` and check `any(inf.id == inference_id for inf in inferences)`. Use the same pattern.

**G7 — `Literal` validation in `Fact` model rejects unexpected DB values**

If any test inserts a `Fact` row directly into SQLite with a `category` or `mutability` value that does not match the `Literal` (e.g., a typo in a fixture), `Fact.model_validate(_row(row))` will raise a `ValidationError`. The in-memory test DB is always freshly created from `_DDL`, so this only surfaces as a test authoring error — it is caught immediately rather than silently storing bad data.

---

## Test Plan

Tests are written first. The implementation is complete when all Phase 4 tests pass alongside all existing Phase 1–3 tests, and overall coverage stays at or above 80%.

---

### Unit tests — `tests/unit/`

#### `test_prompt_builder.py` — additions

These tests exercise `build_system_prompt` with the new category-grouped layout.

| # | Test name | Asserts |
|---|-----------|---------|
| 1 | `test_character_facts_appear_under_character_section` | Facts with `category="character"` appear under a section header containing "Character" |
| 2 | `test_user_facts_appear_under_user_section` | Facts with `category="user"` appear under a section header containing "User" |
| 3 | `test_setting_facts_appear_under_setting_section` | Facts with `category="setting"` appear under a section header containing "Setting" |
| 4 | `test_section_omitted_when_no_facts_in_category` | If no `user` facts exist, no User section header appears in the output |
| 5 | `test_all_three_category_sections_rendered` | A mix of user, character, and setting facts produces three distinct section headers |
| 6 | `test_section_order_is_user_then_character_then_setting` | User section precedes Character section; Character precedes Setting |
| 7 | `test_facts_within_category_in_id_order` | Two facts in the same category appear in ascending `id` order |
| 8 | `test_immutable_fact_has_no_mutability_annotation` | A fact with `mutability="immutable"` does not have any annotation appended to its value line |
| 9 | `test_low_mutability_fact_has_annotation` | A fact with `mutability="low"` has a `[low-mutability` annotation (or equivalent) on its line |
| 10 | `test_high_mutability_fact_has_annotation` | A fact with `mutability="high"` has a `[fluid` annotation (or equivalent) on its line |
| 11 | `test_no_facts_message_when_all_categories_empty` | Empty facts list → some fallback text; no section headers |
| 12 | `test_inferences_section_follows_all_fact_sections` | The `## Your Inferences` header appears after all fact section headers |
| 13 | `test_inferences_absent_when_none_provided` | `inferences=None` → no Inferences section header |
| 14 | `test_single_fact_per_category_renders_correctly` | One character fact, one user fact → both appear under their respective headers |

#### `test_evaluator_service.py` — additions

These tests exercise `build_evaluator_prompt` with the new category+mutability labels and updated verdict instructions.

| # | Test name | Asserts |
|---|-----------|---------|
| 15 | `test_evaluator_prompt_includes_category_for_each_fact` | Each fact line includes its category label (e.g., `"category: character"`) |
| 16 | `test_evaluator_prompt_includes_mutability_for_each_fact` | Each fact line includes its mutability label (e.g., `"mutability: immutable"`) |
| 17 | `test_evaluator_prompt_labels_user_category_facts` | A fact with `category="user"` is labeled accordingly in the evaluator listing |
| 18 | `test_evaluator_prompt_labels_setting_category_facts` | A fact with `category="setting"` is labeled accordingly |
| 19 | `test_evaluator_prompt_contains_immutable_contradiction_instruction` | The prompt contains text explaining that immutable fact violations are `contradiction` |
| 20 | `test_evaluator_prompt_contains_high_mutability_implication_instruction` | The prompt contains text explaining that high-mutability changes should return `implication` |
| 21 | `test_evaluator_prompt_contains_low_mutability_implication_instruction` | The prompt contains text explaining that low-mutability changes should return `implication` |
| 22 | `test_evaluator_prompt_format_for_fact_with_all_fields` | A fact with all fields renders as `[id] key: value  (category: X, mutability: Y)` or equivalent |

---

### Integration tests — `tests/integration/`

#### `tests/integration/test_db_init.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 23 | `test_facts_table_has_category_column` | After `init_db`, `PRAGMA table_info(facts)` includes a column named `category` |
| 24 | `test_facts_table_has_mutability_column` | After `init_db`, `PRAGMA table_info(facts)` includes a column named `mutability` |
| 25 | `test_category_column_default_is_character` | `PRAGMA table_info(facts)` shows `dflt_value` for `category` is `'character'` |
| 26 | `test_mutability_column_default_is_immutable` | `PRAGMA table_info(facts)` shows `dflt_value` for `mutability` is `'immutable'` |
| 27 | `test_facts_uniqueness_constraint_is_category_scoped` | `PRAGMA index_list(facts)` / `sqlite_master` shows the unique constraint covers `(character_id, category, key)` |

#### `tests/integration/test_facts_repo.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 31 | `test_create_fact_default_category_is_character` | `create_fact(key="x", value="y")` without category → returned Fact has `category="character"` |
| 32 | `test_create_fact_default_mutability_is_immutable` | `create_fact(key="x", value="y")` without mutability → returned Fact has `mutability="immutable"` |
| 33 | `test_create_fact_with_user_category` | `create_fact(..., category="user")` → `Fact.category == "user"` |
| 34 | `test_create_fact_with_setting_category` | `create_fact(..., category="setting")` → `Fact.category == "setting"` |
| 35 | `test_create_fact_with_low_mutability` | `create_fact(..., mutability="low")` → `Fact.mutability == "low"` |
| 36 | `test_create_fact_with_high_mutability` | `create_fact(..., mutability="high")` → `Fact.mutability == "high"` |
| 37 | `test_update_fact_value_does_not_change_category` | Create fact with `category="user"`; call `update_fact(fact_id=..., value="v2")` → category still `"user"` |
| 38 | `test_update_fact_value_does_not_change_mutability` | Create fact with `mutability="high"`; call `update_fact(fact_id=..., value="v2")` → mutability still `"high"` |
| 39 | `test_update_fact_category_changes_correctly` | `update_fact(fact_id=..., value="v", category="setting")` → reloaded Fact has `category="setting"` |
| 40 | `test_update_fact_mutability_changes_correctly` | `update_fact(fact_id=..., value="v", mutability="low")` → reloaded Fact has `mutability="low"` |
| 41 | `test_update_fact_all_three_fields_simultaneously` | `update_fact(fact_id=..., value="v2", category="user", mutability="high")` → all three updated |
| 42 | `test_patch_fact_mutability_only` | `patch_fact(fact_id=..., mutability="high")` → `Fact.mutability == "high"`, value and category unchanged |
| 43 | `test_patch_fact_category_only` | `patch_fact(fact_id=..., category="user")` → `Fact.category == "user"`, value and mutability unchanged |
| 44 | `test_patch_fact_raises_not_found_for_unknown_id` | `patch_fact(fact_id=99999, mutability="high")` → `NotFoundError` raised |
| 44b | `test_same_key_allowed_in_different_categories` | `create_fact(key="name", category="user")` then `create_fact(key="name", category="character")` → both succeed; `get_facts` returns two rows with key="name" |
| 44c | `test_same_key_same_category_raises_integrity_error` | `create_fact(key="name", category="user")` twice → `aiosqlite.IntegrityError` on the second call |
| 44d | `test_get_fact_by_category_key_returns_matching_fact` | `get_fact_by_category_key(character_id=..., category="user", key="name")` → returns the user-category fact, not the character-category one |
| 44e | `test_get_fact_by_category_key_returns_none_for_missing` | `get_fact_by_category_key(character_id=..., category="setting", key="name")` when no such fact exists → returns `None` |

#### `tests/integration/test_api_facts.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 45 | `test_create_fact_with_category_returns_201_with_category` | `POST /facts` with `category="user"` → 201 response body has `category: "user"` |
| 46 | `test_create_fact_with_mutability_returns_201_with_mutability` | `POST /facts` with `mutability="high"` → 201 response body has `mutability: "high"` |
| 47 | `test_create_fact_default_category_in_response` | `POST /facts` without category → response body has `category: "character"` |
| 48 | `test_create_fact_default_mutability_in_response` | `POST /facts` without mutability → response body has `mutability: "immutable"` |
| 49 | `test_create_fact_invalid_category_returns_422` | `POST /facts` with `category="invalid"` → 422 |
| 50 | `test_create_fact_invalid_mutability_returns_422` | `POST /facts` with `mutability="invalid"` → 422 |
| 51 | `test_list_facts_includes_category_field` | `GET /facts` → each item in the list has a `category` key |
| 52 | `test_list_facts_includes_mutability_field` | `GET /facts` → each item in the list has a `mutability` key |
| 53 | `test_update_fact_value_preserves_category` | Create with `category="user"` (returns fact with `id`); `PUT /facts/{id}` with only new value → `GET /facts` shows `category="user"` |
| 54 | `test_update_fact_value_preserves_mutability` | Create with `mutability="high"` (returns fact with `id`); `PUT /facts/{id}` with only new value → `GET /facts` shows `mutability="high"` |
| 55 | `test_update_fact_with_new_category_via_put` | `PUT /facts/{id}` with `category="setting"` → response has `category: "setting"` |
| 56 | `test_update_fact_with_new_mutability_via_put` | `PUT /facts/{id}` with `mutability="low"` → response has `mutability: "low"` |
| 57 | `test_patch_fact_mutability_returns_200` | `PATCH /facts/{id}` with `{mutability: "high"}` → 200 with updated Fact |
| 58 | `test_patch_fact_mutability_updates_db` | After `PATCH /facts/{id}`, `GET /facts` shows the new mutability |
| 59 | `test_patch_fact_category_returns_200` | `PATCH /facts/{id}` with `{category: "setting"}` → 200 with updated Fact |
| 60 | `test_patch_fact_category_updates_db` | After `PATCH /facts/{id}`, `GET /facts` shows the new category |
| 61 | `test_patch_fact_value_not_changed_by_patch` | `PATCH /facts/{id}` with mutability only → value in DB unchanged |
| 62 | `test_patch_fact_empty_body_returns_422` | `PATCH /facts/{id}` with `{}` → 422 (nothing to update) |
| 63 | `test_patch_fact_unknown_id_returns_404` | `PATCH /facts/99999` → 404 |
| 64 | `test_patch_fact_invalid_mutability_returns_422` | `PATCH /facts/{id}` with `{mutability: "invalid"}` → 422 |
| 65 | `test_patch_fact_invalid_category_returns_422` | `PATCH /facts/{id}` with `{category: "invalid"}` → 422 |
| 65b | `test_create_two_facts_same_key_different_categories` | `POST /facts` with `{key: "name", category: "user"}` then `POST /facts` with `{key: "name", category: "character"}` → both return 201; `GET /facts` returns two rows with `key="name"` |
| 65c | `test_create_two_facts_same_key_same_category_returns_409` | Two `POST /facts` calls with same key and same category → second returns 409 |
| 65d | `test_put_fact_wrong_character_returns_404` | `PUT /facts/{id}` where the fact belongs to a different character → 404 |
| 65e | `test_delete_fact_by_id_returns_200` | `DELETE /facts/{id}` → 200 (note: existing tests that referenced `DELETE /facts/{key}` are updated to use the fact `id` returned by the create call) |

#### New `tests/integration/test_api_inference_promotion.py`

Fixtures: a character with at least one active inference. The inference may optionally have downstream inferences.

| # | Test name | Asserts |
|---|-----------|---------|
| 66 | `test_promote_inference_returns_201` | `POST /inferences/{id}/promote` with valid body → 201 |
| 67 | `test_promote_inference_response_contains_fact` | Response body has a `fact` key with the created Fact |
| 68 | `test_promote_inference_fact_has_correct_key_and_value` | Created Fact has the key and value from the request body |
| 69 | `test_promote_inference_default_category_is_character` | Promote without category → `fact.category == "character"` |
| 70 | `test_promote_inference_default_mutability_is_immutable` | Promote without mutability → `fact.mutability == "immutable"` |
| 71 | `test_promote_inference_with_custom_category` | Promote with `category="setting"` → `fact.category == "setting"` |
| 72 | `test_promote_inference_with_custom_mutability` | Promote with `mutability="high"` → `fact.mutability == "high"` |
| 73 | `test_promote_inference_fact_stored_in_db` | After promotion, `GET /facts` includes the new fact key |
| 74 | `test_promote_inference_deletes_source_inference` | After promotion, `GET /inferences` (status=all) does not include the promoted inference |
| 75 | `test_promote_inference_response_contains_stale_inferences_list` | Response body has a `stale_inferences` key (may be empty) |
| 76 | `test_promote_inference_marks_downstream_inference_stale` | A downstream inference (with `source_inference_ids` containing the promoted id) is marked `stale` in DB and appears in `stale_inferences` in the response |
| 77 | `test_promote_inference_non_downstream_inference_stays_active` | An unrelated active inference is not stale after promotion |
| 78 | `test_promote_inference_unknown_character_returns_404` | POST for non-existent `character_id` → 404 |
| 79 | `test_promote_inference_unknown_inference_returns_404` | POST for non-existent `inference_id` → 404 |
| 80 | `test_promote_inference_from_different_character_returns_404` | POST with an `inference_id` belonging to a different character → 404 |
| 81 | `test_promote_inference_key_already_exists_in_same_category_returns_409` | Promote with `category="character"` and a key that already exists as a `character` Fact → 409 |
| 81b | `test_promote_inference_key_exists_in_different_category_succeeds` | Promote with `category="character"` and a key that only exists as a `user` Fact → 201 (different category, no conflict) |
| 82 | `test_promote_inference_stale_inference_can_be_promoted` | An inference with `status="stale"` can still be promoted (no status restriction) |

#### `tests/integration/test_api_chat.py` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 83 | `test_chat_system_prompt_groups_user_and_character_facts` | Create a `user` fact and a `character` fact; send a message; inspect the first Ollama call's `messages[0].content` → contains both category section headers |
| 84 | `test_chat_system_prompt_omits_empty_setting_section` | No `setting` facts exist; inspect system prompt → no Setting section header |
| 85 | `test_chat_system_prompt_annotates_high_mutability_fact` | Create a fact with `mutability="high"`; send a message; system prompt contains the `[fluid` annotation |
| 86 | `test_chat_system_prompt_annotates_low_mutability_fact` | Create a fact with `mutability="low"`; send a message; system prompt contains the `[low-mutability` annotation |
| 87 | `test_accept_implication_on_high_mutability_fact_preserves_mutability` | Create a fact with `mutability="high"`; run a turn where the evaluator returns `implication` for the same key with a new value; accept the implication; reload the fact → `mutability` is still `"high"` |

---

### Frontend tests — `tests/frontend/chat.test.js` — additions

| # | Test name | Asserts |
|---|-----------|---------|
| 88 | `apiCreateFact_posts_to_correct_url` | `fetch` called with `POST /api/characters/{id}/facts` |
| 89 | `apiCreateFact_sends_key_value_category_mutability_in_body` | Request body JSON contains `key`, `value`, `category`, and `mutability` |
| 90 | `apiCreateFact_uses_default_category_character` | `apiCreateFact(7, "mood", "cheerful")` → body has `category: "character"` |
| 91 | `apiCreateFact_uses_default_mutability_immutable` | `apiCreateFact(7, "mood", "cheerful")` → body has `mutability: "immutable"` |
| 92 | `apiCreateFact_accepts_custom_category` | `apiCreateFact(7, "mood", "cheerful", "user", "high")` → body has `category: "user"` |
| 93 | `apiCreateFact_accepts_custom_mutability` | `apiCreateFact(7, "mood", "cheerful", "user", "high")` → body has `mutability: "high"` |
| 94 | `apiPatchFactMutability_sends_patch_to_correct_url` | `fetch` called with `PATCH /api/characters/7/facts/mood` |
| 95 | `apiPatchFactMutability_sends_mutability_in_body` | Request body JSON contains `{ mutability: "high" }` |
| 96 | `apiPatchFactMutability_does_not_send_category_field` | Request body does NOT contain a `category` key |
| 97 | `apiPatchFactCategory_sends_patch_to_correct_url` | `fetch` called with `PATCH /api/characters/7/facts/mood` |
| 98 | `apiPatchFactCategory_sends_category_in_body` | Request body JSON contains `{ category: "setting" }` |
| 99 | `apiPatchFactCategory_does_not_send_mutability_field` | Request body does NOT contain a `mutability` key |
| 100 | `apiPatchFactMutability_url_encodes_key_with_spaces` | Key `"home city"` → URL contains `home%20city` |
| 101 | `apiPromoteInference_posts_to_correct_url` | `fetch` called with `POST /api/characters/7/inferences/42/promote` |
| 102 | `apiPromoteInference_sends_key_value_category_mutability_in_body` | Request body JSON contains `key`, `value`, `category`, and `mutability` |
| 103 | `apiPromoteInference_uses_default_category_character` | `apiPromoteInference(7, 42, "birth_year", "1993")` → body has `category: "character"` |
| 104 | `apiPromoteInference_uses_default_mutability_immutable` | `apiPromoteInference(7, 42, "birth_year", "1993")` → body has `mutability: "immutable"` |
| 105 | `apiPromoteInference_accepts_custom_category` | `apiPromoteInference(7, 42, "location", "Chicago", "setting", "low")` → body has `category: "setting"` |
| 106 | `apiPromoteInference_accepts_custom_mutability` | `apiPromoteInference(7, 42, "location", "Chicago", "setting", "low")` → body has `mutability: "low"` |
| 107 | `buildNotificationFromSidechannel_still_handles_implication_for_mutability_change` | Payload with `type: "implication"` where violation description mentions "high-mutability fact" → returns valid notification object (existing implication handler covers this; no new case needed) |

---

## Not in Scope for Phase 4

- **Experiences** — Phase 5. The `experiences` table exists but stays empty. `experience_update` verdict continues to be coerced to `pass`.
- **Inter-fact constraint logic** — Proposed Future Work. E.g., "clothing would not change while character is at work." Phase 4 adds mutability levels but does not encode inter-fact dependencies. All `low`-mutability changes are surfaced as `implication` uniformly.
- **Evaluator LLM revalidation on promotion** — the promote endpoint marks downstream inferences stale without an LLM call. Actual revalidation of downstream inferences against the new Fact is the user's responsibility (they can PATCH status back to `active` if the inference still holds). A future improvement could run `revalidate_single_inference` for each downstream inference and only mark those that fail.
- **User prompt for mutability on implication acceptance** — when the user accepts an implied fact, the mutability defaults to `immutable`. A future UX improvement could ask for mutability at acceptance time (particularly useful for mood/location implications). Deferred to Phase 7 polish.
- **Numerical range facts** — Proposed Future Work. Big 5 traits, D&D alignment, etc. require a `fact_type_definitions` table and slider UI; out of scope.
- **Experiences, context budget, and compression** — Phases 5 and 6.
