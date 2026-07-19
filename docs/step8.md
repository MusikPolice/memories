# Step 8 — UI: Schema-Driven Fact Tree, Approval Cards, and Legacy Removal

## Overview

Step 8 rebuilds the entire in-session UI around the Facts v2 model. Until now the
frontend has been running against the **old flat fact store** — it fetches
`GET /api/characters/{id}/facts` expecting a list of `{id, key, value, category,
mutability}` rows, renders three category columns, offers a create-fact form, per-fact
mutability/category dropdowns, and an inference "promote to fact" flow. Every one of
those affordances belongs to a fact model that no longer exists server-side (Steps 2–7
moved the source of truth into the `character_facts` JSON blob and the tool-call
evaluator). The fact panel today is effectively disconnected from the world state the
character actually sees. This step reconnects it.

The new fact panel is a **collapsible tree that mirrors `fact_schema.json`**. The frontend
fetches the schema once at startup (`GET /api/schema`) and the current values per
character (`GET /api/characters/{id}/facts`, rewritten in this step to return the masked
blob). Each schema leaf renders a type-aware inline editor — a dropdown for `Enum`, a
numeric input for `Integer`, free text for `String`. Fact **creation** is gone (paths are
defined by the schema, not the user); only leaf **values** are editable. The three
`fact_update_*` sidechannel cards emitted since Steps 5–7 are wired in, along with the
`require_fact` blocking card (emitted since Step 6) and the `inference_proposed` quiet
notification (emitted since Step 5). All the dead legacy notification cards, their
handlers, and their tests are deleted.

Because Steps 9 and 10 add **no** new HTTP endpoints (Step 9 migrates inference source
columns; Step 10 is documentation only), the small backend needed to serve and mutate the
fact blob is included here: `routers/facts.py` is rewritten to expose a blob-read endpoint
and a by-path value-write endpoint over the existing `get_facts()` / `patch_fact()`
repository functions. The legacy row-based CRUD endpoints and their integration tests are
removed — per the plan, none of the old fact machinery needs to survive.

**Success criterion:** all tests in `tests/frontend/chat.test.js`,
`tests/frontend/chat-component.test.js`, and `tests/integration/test_api_facts.py` pass
(`npm run test:coverage` holds the 80% threshold on `chat.js`; `uv run pytest
tests/integration/test_api_facts.py` is green), along with every other existing test in
both suites. The dev server renders the fact tree, allows inline value edits, and shows
the mutable / immutable-unset / require_fact blocking cards with working accept/edit/
reject/dismiss buttons.

---

## What Steps 1–7 Delivered

Concrete foundations this step consumes (all verified present in the current tree):

- **`GET /api/schema`** (`src/memories/routers/schema.py`) — returns `fact_schema.json`
  verbatim as a recursive JSON tree; a node with a `Type` key is a leaf, a node without is
  a grouping. Mounted at `/api` in `main.py` (so the URL is `/api/schema`).
- **`src/memories/schema_loader.py`** — `load_schema()`, `check_write_permitted(path,
  schema) -> str` (returns `"Immutable"`/`"Mutable"`/`"Fluid"`; raises `ValueError` for an
  unknown path or a grouping path), `_collect_leaves(node, prefix="") -> list[tuple[str,
  dict]]` (flattens the schema to `(dot.path, leaf_dict)` pairs), `apply_mask(blob)`.
- **`src/memories/database.py`** — `get_facts(db, character_id) -> dict` (returns the
  schema-masked blob, `{}` if no row), `set_facts(db, character_id, blob)`,
  `patch_fact(db, character_id, path_tuple: tuple[str, ...], value)` (writes one leaf's
  `{"Value": value}`, creating intermediate groupings, non-destructive to siblings).
- **Sidechannel event shapes the server already emits** (payloads observed in
  `evaluator.py` and `chat_service.py`):
  - `fact_update_fluid` → `{"type", "turn_id", "path", "value"}` (applied already; quiet)
  - `fact_update_mutable` → `{"type", "turn_id", "path", "proposed"}` (blocking)
  - `fact_update_immutable_unset` → `{"type", "turn_id", "path", "proposed"}` (blocking)
  - `inference_proposed` → `{"type", "turn_id", "inference": <Inference model dump>}`
    (already written to DB; quiet)
  - `require_fact` → `{"type", "turn_id", "path", "reason", "suggested_value"}` (blocking)
  - `contradiction` → `{"type", "iteration", "description"}` (already handled in the UI)
- **Blocking-card respond endpoints:**
  - `POST /api/sessions/{sid}/turns/{tid}/require-fact/respond`, body `{"value": str |
    null}` (`routers/require_fact.py`, Step 6). 404 if no gate, 409 on double-resolve.
  - `POST /api/sessions/{sid}/turns/{tid}/set-fact/respond`, body `{"action":
    "accept"|"edit"|"reject"|"dismiss", "value": str | null}` (`routers/fact_approval.py`,
    Step 7). 404 if no gate, 409 on double-resolve. `accept` uses the model's proposed
    value; `edit` uses the supplied `value` and regenerates; `reject` (mutable) discards
    and regenerates; `dismiss` (immutable-unset) delivers the response as-is.
- **`message` event** no longer carries an `ungrounded` key (Step 7 Part H removed it); it
  carries `content`, `turn_id`, `contradiction_exhausted`, and the Phase-5
  `active_experience_ids` / `experience_scores`.
- **Frontend as it stands** — `src/memories/frontend/{chat.js, chat-component.js,
  index.html}` plus `tests/frontend/{chat.test.js, chat-component.test.js}`. The
  `parseSSEBlock`, `parseSSEBlocks`, `sseStateToLabel`, `buildScoreMap`, `sortExperiences`,
  `buildProposalList` pure helpers and the experience/session-end panel are retained
  unchanged except where noted.

---

## What This Step Does NOT Change

- **The inferences table and `source_fact_ids`.** Inferences still use integer
  `source_fact_ids` until Step 9. `GET /api/characters/{id}/inferences` (currently defined
  in `routers/facts.py`) keeps working and keeps its shape. The side-panel inference list
  keeps rendering inferences; only the **promote-to-fact** affordance is removed here (the
  `/inferences/{id}/promote` **endpoint** removal is Step 9).
- **The evaluator, World Builder, chat service, and tool-gate.** No server logic under
  `services/` is touched. This step only rewrites `routers/facts.py` and edits the three
  frontend files.
- **`routers/require_fact.py` and `routers/fact_approval.py`.** The respond endpoints are
  consumed as-is; no backend change.
- **The contradiction notification card.** `scType === 'contradiction'` handling in
  `chat.js` / `index.html` / `chat-component.js` stays exactly as it is.
- **The experiences / session-end panel.** `sortExperiences`, `buildScoreMap`,
  `buildProposalList`, `endSession`, `acceptProposal`, `confirmEditProposal`,
  `discardProposal`, `deleteExperience`, and the active-experience tracking in the
  `message` event handler are untouched. Only the `experience_update` **contradiction
  delete** path (`removeContradictedExperiences` + its sidechannel handler) is removed,
  because the `experience_update` verdict was deleted server-side in the Facts v2 rework.
- **Inference cascade on fact edit.** Editing a leaf value in the new fact panel writes the
  blob but does **not** revalidate downstream inferences in this step. Path-based cascade
  belongs with Step 9's `source_fact_paths` migration. Accepted transitional gap (see Edge
  Cases).
- **`CLAUDE.md` / `README.md`.** Step 10.

---

## Detailed Design

### Part A — Backend: rewrite `src/memories/routers/facts.py` to the blob model

Replace the row-based fact endpoints with two blob endpoints. The `GET
/{character_id}/inferences` endpoint stays exactly as it is.

**Imports — before:**

```python
from memories.database import (
    create_fact,
    delete_fact,
    get_character,
    get_fact_rows,
    get_inferences,
    patch_fact_row,
    update_fact,
)
from memories.deps import get_db
from memories.exceptions import NotFoundError
from memories.models import Fact, Inference
```

**Imports — after:**

```python
from memories.database import (
    get_character,
    get_facts,
    get_inferences,
    patch_fact,
)
from memories.deps import get_db
from memories.models import Inference
from memories.schema_loader import _collect_leaves, check_write_permitted, load_schema
```

(`sqlite3`, `Fact`, `NotFoundError`, `Query`-based create/update/patch/delete imports are
dropped. `Literal` is no longer needed.)

**Remove entirely:** `_CreateBody`, `_UpdateBody`, `_PatchBody`, `create_fact_endpoint`
(POST), `update_fact_endpoint` (PUT `/{fact_id}`), `patch_fact_endpoint` (PATCH
`/{fact_id}`), `delete_fact_endpoint` (DELETE `/{fact_id}`).

**Rewrite `list_facts_endpoint`** to return the masked blob:

```python
@router.get("/{character_id}/facts")
async def get_facts_endpoint(character_id: int, db: _DB) -> dict[str, Any]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return await get_facts(db, character_id)
```

**Add a by-path value-write endpoint.** The body carries a dot-notation `path` and a
string `value`; the handler validates the path exists in the schema, coerces the value to
the leaf's declared `Type`, writes it via `patch_fact`, and returns the updated blob so the
client can re-render without a second round trip:

```python
class _SetFactBody(BaseModel):
    path: str
    value: str


@router.put("/{character_id}/facts")
async def set_fact_value_endpoint(
    character_id: int, body: _SetFactBody, db: _DB
) -> dict[str, Any]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    schema = load_schema()
    try:
        check_write_permitted(body.path, schema)  # raises ValueError on unknown/grouping path
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    leaf = dict(_collect_leaves(schema))[body.path]
    coerced: str | int = body.value
    if leaf["Type"] == "Enum":
        match = next(
            (c for c in leaf["Constraint"] if c.lower() == body.value.lower()), None
        )
        if match is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{body.value!r} is not a valid value for {body.path}. "
                    f"Valid values: {', '.join(leaf['Constraint'])}"
                ),
            )
        coerced = match
    elif leaf["Type"] == "Integer":
        try:
            coerced = int(body.value)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"{body.value!r} is not a valid integer"
            ) from exc

    await patch_fact(db, character_id, tuple(body.path.split(".")), coerced)
    return await get_facts(db, character_id)
```

Notes:
- No mutability check on this endpoint — the fact panel is the user's own settings surface;
  the user is the author and may set any leaf, exactly as the World Builder can.
- `Any` is imported from `typing` (already used elsewhere in routers; add
  `from typing import Annotated, Any`).
- Clearing a value (removing a leaf) is **not** exposed in Step 8; unset-by-UI is deferred.
  Editing a value to a new value is the only mutation. (A user can still "blank" a `String`
  leaf by saving an empty string, which is harmless.)

### Part B — `src/memories/frontend/chat.js`

#### B1. `buildNotificationFromSidechannel` — remove dead cases, add live cases

**Remove** the `implication`, `new_inference_probabilistic`, `extraction_applied`,
`implicit_fact_proposed`, and `experience_update` branches. None of those types are emitted
by the Facts v2 backend.

**Keep** the `contradiction` branch unchanged.

**Add** five branches (four new + `contradiction` retained). New card builders:

```javascript
if (payload.type === 'fact_update_fluid') {
  return {
    role: 'notification', scType: 'fact_update_fluid',
    turn_id: payload.turn_id, path: payload.path, value: payload.value,
  };
}

if (payload.type === 'fact_update_mutable') {
  return {
    role: 'notification', scType: 'fact_update_mutable',
    turn_id: payload.turn_id, path: payload.path, proposed: payload.proposed,
    _editValue: payload.proposed ?? '', _loading: false,
  };
}

if (payload.type === 'fact_update_immutable_unset') {
  return {
    role: 'notification', scType: 'fact_update_immutable_unset',
    turn_id: payload.turn_id, path: payload.path, proposed: payload.proposed,
    _editValue: payload.proposed ?? '', _loading: false,
  };
}

if (payload.type === 'require_fact') {
  return {
    role: 'notification', scType: 'require_fact',
    turn_id: payload.turn_id, path: payload.path, reason: payload.reason,
    _editValue: payload.suggested_value ?? '', _loading: false,
  };
}

if (payload.type === 'inference_proposed') {
  return {
    role: 'notification', scType: 'inference_proposed',
    turn_id: payload.turn_id, inference: payload.inference,
  };
}
```

(Keep the `contradiction` branch above/below these; keep the final `return null`.)

#### B2. Remove dead helpers

Delete these exports (all tied to removed flows) and drop their tests in B/Test Plan:
`removeViolation`, `apiAcceptImplication`, `apiIgnoreImplication`, `apiAcceptInference`,
`apiIgnoreInference`, `apiCreateFact`, `apiPatchFactMutability`, `apiPatchFactCategory`,
`apiPromoteInference`, `removeContradictedExperiences`, `apiUndoUserFact`,
`apiAcceptImplicitFact`, `apiIgnoreImplicitFact`, `apiDeleteFact`. Retain
`apiGenerateInferences`, `apiRevalidateInferences`, `apiDeleteInference`,
`apiPatchInferenceStatus`, `apiEndSession`, `apiCreateExperience`, `apiListExperiences`,
`apiDeleteExperience`, and all SSE/experience pure helpers.

#### B3. New schema-tree pure helpers

```javascript
/**
 * Flatten a fact schema tree into an ordered, depth-first row list.
 * A node with a `Type` key is a leaf; any other node is a grouping.
 * @param {object} schema
 * @param {string} [prefix='']
 * @param {number} [depth=0]
 * @returns {Array<{path,name,depth,isLeaf,type?,mutability?,constraint?,description?}>}
 */
export function flattenSchema(schema, prefix = '', depth = 0) {
  const rows = [];
  for (const [name, node] of Object.entries(schema || {})) {
    const path = prefix ? `${prefix}.${name}` : name;
    if (node && typeof node === 'object' && 'Type' in node) {
      rows.push({
        path, name, depth, isLeaf: true,
        type: node.Type, mutability: node.Mutability,
        constraint: node.Constraint ?? null, description: node.Description ?? '',
      });
    } else {
      rows.push({ path, name, depth, isLeaf: false });
      rows.push(...flattenSchema(node, path, depth + 1));
    }
  }
  return rows;
}

/**
 * Read a leaf value out of a fact blob by dot-notation path.
 * @returns the stored Value, or undefined if the path is unset.
 */
export function lookupBlobValue(blob, path) {
  let node = blob;
  for (const part of path.split('.')) {
    if (!node || typeof node !== 'object' || !(part in node)) return undefined;
    node = node[part];
  }
  return (node && typeof node === 'object') ? node.Value : undefined;
}

/**
 * Produce the visible fact-tree rows: flatten the schema, drop rows whose ancestor
 * grouping is collapsed, and attach the current value to each leaf.
 * @param {object} schema
 * @param {object} blob
 * @param {Set<string>} collapsedPaths  grouping paths whose children are hidden
 * @returns {Array<object>}  rows with `value` set on leaves (undefined if unset)
 */
export function buildVisibleFactRows(schema, blob, collapsedPaths) {
  return flattenSchema(schema)
    .filter(row => {
      for (const c of collapsedPaths) {
        if (row.path !== c && row.path.startsWith(c + '.')) return false;
      }
      return true;
    })
    .map(row => (row.isLeaf ? { ...row, value: lookupBlobValue(blob, row.path) } : row));
}

/** Look up a single leaf's schema definition by path (for type-aware card inputs). */
export function schemaLeaf(schema, path) {
  return flattenSchema(schema).find(r => r.isLeaf && r.path === path) ?? null;
}
```

#### B4. New API helpers

```javascript
export function apiGetSchema() {
  return fetch('/api/schema');
}

export function apiGetFactBlob(characterId) {
  return fetch(`/api/characters/${characterId}/facts`);
}

export function apiSetFactValue(characterId, path, value) {
  return fetch(`/api/characters/${characterId}/facts`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path, value }),
  });
}

export function apiRespondRequireFact(sessionId, turnId, value) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/require-fact/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value }),
  });
}

export function apiRespondSetFact(sessionId, turnId, action, value = null) {
  return fetch(`/api/sessions/${sessionId}/turns/${turnId}/set-fact/respond`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, value }),
  });
}
```

### Part C — `src/memories/frontend/index.html`

#### C1. Message-card block

No change to the user/assistant/journal/thinking blocks. The `contradictionExhausted`
warning stays. (There is no `ungrounded` badge in the current template — confirmed by grep
— so nothing to remove there.)

#### C2. Notification cards

**Remove** the `implication`, `new_inference_probabilistic`, `experience_update`,
`extraction_applied`, and `implicit_fact_proposed` `v-else-if` card blocks. **Keep** the
`contradiction` card. **Add** five cards.

- **`fact_update_fluid`** — quiet, dismissible:

```html
<div v-else-if="msg.role === 'notification' && msg.scType === 'fact_update_fluid'"
     class="notification-card fact-fluid">
  <div class="notification-title">💧 {{ msg.path }} updated: {{ msg.value }}</div>
  <div class="notification-body">
    <div class="notification-actions">
      <button class="btn-ignore" @click="dismissNotification(msg)">Dismiss</button>
    </div>
  </div>
</div>
```

- **`fact_update_mutable`** — blocking; accept / edit / reject. A type-aware input pre-filled
  with the proposed value; the input is rendered from `schemaLeaf(msg.path)`:

```html
<div v-else-if="msg.role === 'notification' && msg.scType === 'fact_update_mutable'"
     class="notification-card fact-mutable">
  <div class="notification-title">📌 Character implied a change — {{ msg.path }}</div>
  <div class="notification-body">
    <p class="violation-desc">Proposed value: <strong>{{ msg.proposed }}</strong></p>
    <div class="violation-fact">
      <label>Value</label>
      <select v-if="leafType(msg.path) === 'Enum'" v-model="msg._editValue">
        <option v-for="c in leafConstraint(msg.path)" :key="c" :value="c">{{ c }}</option>
      </select>
      <input v-else :type="leafType(msg.path) === 'Integer' ? 'number' : 'text'"
             v-model="msg._editValue" />
    </div>
    <div class="notification-actions">
      <button class="btn-accept" :disabled="msg._loading" @click="resolveMutable(msg, 'accept')">
        {{ msg._loading ? 'Working…' : 'Accept' }}
      </button>
      <button class="btn-accept" :disabled="msg._loading" @click="resolveMutable(msg, 'edit')">
        Save edit
      </button>
      <button class="btn-ignore" :disabled="msg._loading" @click="resolveMutable(msg, 'reject')">
        Reject
      </button>
    </div>
  </div>
</div>
```

- **`fact_update_immutable_unset`** — blocking; accept / edit / dismiss (same input pattern,
  different verbs and handler):

```html
<div v-else-if="msg.role === 'notification' && msg.scType === 'fact_update_immutable_unset'"
     class="notification-card fact-immutable">
  <div class="notification-title">🔒 Set a permanent value — {{ msg.path }}</div>
  <div class="notification-body">
    <p class="violation-desc">Character used: <strong>{{ msg.proposed }}</strong>
      (locks permanently once set)</p>
    <div class="violation-fact">
      <label>Value</label>
      <select v-if="leafType(msg.path) === 'Enum'" v-model="msg._editValue">
        <option v-for="c in leafConstraint(msg.path)" :key="c" :value="c">{{ c }}</option>
      </select>
      <input v-else :type="leafType(msg.path) === 'Integer' ? 'number' : 'text'"
             v-model="msg._editValue" />
    </div>
    <div class="notification-actions">
      <button class="btn-accept" :disabled="msg._loading" @click="resolveImmutable(msg, 'accept')">
        {{ msg._loading ? 'Working…' : 'Accept' }}
      </button>
      <button class="btn-accept" :disabled="msg._loading" @click="resolveImmutable(msg, 'edit')">
        Save edit
      </button>
      <button class="btn-ignore" :disabled="msg._loading" @click="resolveImmutable(msg, 'dismiss')">
        Dismiss
      </button>
    </div>
  </div>
</div>
```

- **`require_fact`** — blocking; confirm / dismiss. Same type-aware input, seeded from
  `suggested_value`:

```html
<div v-else-if="msg.role === 'notification' && msg.scType === 'require_fact'"
     class="notification-card fact-immutable">
  <div class="notification-title">❓ The character needs a value — {{ msg.path }}</div>
  <div class="notification-body">
    <p class="violation-desc">{{ msg.reason }}</p>
    <div class="violation-fact">
      <label>Value</label>
      <select v-if="leafType(msg.path) === 'Enum'" v-model="msg._editValue">
        <option v-for="c in leafConstraint(msg.path)" :key="c" :value="c">{{ c }}</option>
      </select>
      <input v-else :type="leafType(msg.path) === 'Integer' ? 'number' : 'text'"
             v-model="msg._editValue" />
    </div>
    <div class="notification-actions">
      <button class="btn-accept" :disabled="msg._loading" @click="resolveRequireFact(msg, true)">
        {{ msg._loading ? 'Working…' : 'Confirm' }}
      </button>
      <button class="btn-ignore" :disabled="msg._loading" @click="resolveRequireFact(msg, false)">
        Not now
      </button>
    </div>
  </div>
</div>
```

- **`inference_proposed`** — quiet, dismissible:

```html
<div v-else-if="msg.role === 'notification' && msg.scType === 'inference_proposed'"
     class="notification-card inference">
  <div class="notification-title">💡 New inference</div>
  <div class="notification-body">
    <p class="inference-statement">{{ msg.inference.statement }}</p>
    <p class="inference-derivation">{{ msg.inference.derivation }}</p>
    <div class="notification-actions">
      <button class="btn-ignore" @click="dismissNotification(msg)">Dismiss</button>
    </div>
  </div>
</div>
```

Add matching CSS classes (`.notification-card.fact-fluid`, `.fact-mutable`,
`.fact-immutable`) mirroring the existing colour-band pattern; reuse `.inference` for
`inference_proposed`.

#### C3. Facts panel — replace the three category columns with the schema tree

Replace the entire `.facts-list` body (the three `factsByCategory.*` `template` blocks) with
a single flat, indented, collapsible render driven by `visibleFactRows`:

```html
<div class="facts-list">
  <template v-for="row in visibleFactRows" :key="row.path">
    <div v-if="!row.isLeaf" class="fact-group"
         :style="{ paddingLeft: (row.depth * 12) + 'px' }"
         @click="toggleGroup(row.path)">
      <span>{{ collapsedGroups.has(row.path) ? '▸' : '▾' }}</span> {{ row.name }}
    </div>
    <div v-else class="fact-leaf" :style="{ paddingLeft: (row.depth * 12) + 'px' }">
      <span class="fact-leaf-name" :title="row.description">{{ row.name }}</span>
      <select v-if="row.type === 'Enum'"
              v-model="leafEdits[row.path]"
              @change="saveLeaf(row)">
        <option value="">(not set)</option>
        <option v-for="c in row.constraint" :key="c" :value="c">{{ c }}</option>
      </select>
      <input v-else
             :type="row.type === 'Integer' ? 'number' : 'text'"
             v-model="leafEdits[row.path]"
             @keyup.enter="saveLeaf(row)"
             @blur="saveLeaf(row)"
             placeholder="(not set)" />
    </div>
  </template>
  <p v-if="visibleFactRows.length === 0" style="color:#aaa;font-size:13px">
    Loading schema…
  </p>
</div>
```

**Remove** the entire `.add-fact-form` block (create form). **Keep** the `.facts-footer`
End-Session / New-Session block. Add minimal CSS for `.fact-group` (bold, pointer cursor)
and `.fact-leaf` (flex row, small name label + input).

#### C4. Inference side-panel — drop promote, add delete

In the inferences pane, **remove** the `.promote-btn` button and the entire
`.promote-form` block. **Add** a delete button per row calling `deleteInference(inf)`.
Keep the expand-on-click derivation and the type badge:

```html
<div v-for="inf in inferences" :key="inf.id" class="inference-row">
  <div class="inference-row-top">
    <span class="inference-row-statement" @click="inf._expanded = !inf._expanded">
      <span class="inference-type-badge" :class="inf.inference_type">{{ inf.inference_type }}</span>{{ inf.statement }}
    </span>
    <button class="exp-delete" @click.stop="deleteInference(inf)" title="Delete">✕</button>
  </div>
  <div v-if="inf._expanded" class="inference-row-derivation">{{ inf.derivation }}</div>
</div>
```

### Part D — `src/memories/frontend/chat-component.js`

#### D1. Imports

Import the new/retained helpers from `chat.js`; drop all removed ones:

```javascript
import {
  parseSSEBlock, sseStateToLabel, buildNotificationFromSidechannel,
  apiEndSession, apiCreateExperience, apiDeleteExperience,
  buildScoreMap, buildProposalList, sortExperiences,
  apiDeleteInference,
  buildVisibleFactRows, schemaLeaf,
  apiGetSchema, apiGetFactBlob, apiSetFactValue,
  apiRespondRequireFact, apiRespondSetFact,
} from './chat.js';
```

#### D2. State — remove old fact machinery, add schema/blob/tree state

**Remove** refs and helpers: `newKey`, `newValue`, `newCategory`, `newMutability`,
`factError`, `factsByCategory` (computed), `mutabilityIcon`, `closeAllMutDropdowns`,
`closeAllCatDropdowns`, `toggleMutability`, `toggleCategory`, `patchCategory`,
`patchMutability`, `addFact`, `saveFact`, `deleteFact`, the `onMounted`/`onUnmounted`
document-click listeners, and all `_promote*`/`_mut*`/`_cat*` fields in `loadInferences`.
Remove `acceptImplication`, `ignoreImplication`, `acceptInference`, `ignoreInference`,
`undoUserFact`, `deleteExtractedFact`, `acceptImplicitFact`, `ignoreImplicitFact`, and
`promoteInference` / `togglePromote`.

`facts` was a `ref([])` of row objects; replace it with a blob object plus schema:

```javascript
const schema = ref({});                 // fetched once from GET /api/schema
const factsBlob = ref({});              // fetched per character from GET /facts
const collapsedGroups = ref(new Set()); // grouping paths currently collapsed
const leafEdits = ref({});              // path -> current input value (string)
```

#### D3. Schema fetch, blob load, tree computed

```javascript
async function loadSchema() {
  const r = await apiGetSchema();
  if (r.ok) schema.value = await r.json();
}

async function loadFacts() {
  if (!currentCharacter.value) return;
  const r = await apiGetFactBlob(currentCharacter.value.id);
  factsBlob.value = r.ok ? await r.json() : {};
  syncLeafEdits();
}

// Seed the editable inputs from the blob so v-model has a starting value per leaf.
function syncLeafEdits() {
  const edits = {};
  for (const row of buildVisibleFactRows(schema.value, factsBlob.value, new Set())) {
    if (row.isLeaf) edits[row.path] = row.value ?? '';
  }
  leafEdits.value = edits;
}

const visibleFactRows = computed(() =>
  buildVisibleFactRows(schema.value, factsBlob.value, collapsedGroups.value)
);

function toggleGroup(path) {
  const next = new Set(collapsedGroups.value);
  next.has(path) ? next.delete(path) : next.add(path);
  collapsedGroups.value = next;
}

function leafType(path) { return schemaLeaf(schema.value, path)?.type ?? 'String'; }
function leafConstraint(path) { return schemaLeaf(schema.value, path)?.constraint ?? []; }

async function saveLeaf(row) {
  const value = leafEdits.value[row.path] ?? '';
  await apiSetFactValue(currentCharacter.value.id, row.path, value);
  await loadFacts();
}
```

`loadSchema()` is called once, alongside `loadCharacters()`, at the end of `setup()`.
`loadFacts()` keeps its old name and call sites (`pickCharacter`, the `done` SSE branch,
etc.) but now loads the blob.

#### D4. Inference delete + reload on turn end

```javascript
async function deleteInference(inf) {
  await apiDeleteInference(currentCharacter.value.id, inf.id);
  await loadInferences();
}
```

`loadInferences()` loses its `_promote*` fields (keep only `_expanded: false`). In the SSE
`done` branch, reload inferences too so `inference_proposed` results appear:

```javascript
} else if (eventName === 'done') {
  if (currentCharacter.value) {
    await loadFacts();
    await loadInferences();
  }
}
```

#### D5. Blocking-card resolve handlers

```javascript
async function resolveRequireFact(notif, confirmed) {
  notif._loading = true;
  try {
    const value = confirmed ? (notif._editValue ?? '') : null;
    await apiRespondRequireFact(sessionId.value, notif.turn_id, value);
    dismissNotification(notif);
  } finally { notif._loading = false; }
}

async function resolveMutable(notif, action) {
  notif._loading = true;
  try {
    const value = action === 'edit' ? (notif._editValue ?? '') : null;
    await apiRespondSetFact(sessionId.value, notif.turn_id, action, value);
    dismissNotification(notif);
  } finally { notif._loading = false; }
}

async function resolveImmutable(notif, action) {
  notif._loading = true;
  try {
    const value = action === 'edit' ? (notif._editValue ?? '') : null;
    await apiRespondSetFact(sessionId.value, notif.turn_id, action, value);
    dismissNotification(notif);
  } finally { notif._loading = false; }
}
```

Each handler dismisses its own card and returns immediately; the still-open SSE reader
delivers the resumed `message`/`done` events. `sending`/`generating` remain managed by the
`sendMessage` `finally` block.

#### D6. `sendMessage` SSE loop

In the `sidechannel` branch, delete the `experience_update` special-case (the
`removeContradictedExperiences` call). Blocking cards should stop the "Thinking…" spinner
so the card is visible while the stream waits on the user:

```javascript
} else if (eventName === 'sidechannel' && dataStr) {
  const notif = buildNotificationFromSidechannel(JSON.parse(dataStr));
  if (notif) {
    const blocking = ['fact_update_mutable', 'fact_update_immutable_unset', 'require_fact'];
    if (blocking.includes(notif.scType)) {
      generating.value = false;
      statusText.value = '';
    }
    messages.value.push(notif);
    await scrollToBottom();
  }
}
```

#### D7. `newSession` reset

Reset the new state alongside the existing resets: `factsBlob.value = {}`,
`collapsedGroups.value = new Set()`, `leafEdits.value = {}`. (Do **not** clear
`schema.value` — the schema is process-global and does not change between sessions.)

#### D8. `return` object

Remove every deleted symbol; add `schema`, `factsBlob`, `visibleFactRows`,
`collapsedGroups`, `leafEdits`, `toggleGroup`, `leafType`, `leafConstraint`, `saveLeaf`,
`deleteInference`, `resolveRequireFact`, `resolveMutable`, `resolveImmutable`, `loadFacts`,
`loadInferences`. Keep `dismissNotification`, the experience/session-end symbols, and the
chat symbols.

---

## Transitional State After Step 8

- The fact panel is a live schema tree with type-aware inline editing, backed by the real
  `character_facts` blob. Fact creation, per-fact mutability/category dropdowns, and the
  three-category layout are gone.
- All five Facts v2 sidechannel cards render: `fact_update_fluid` and `inference_proposed`
  as quiet dismissible notes; `fact_update_mutable`, `fact_update_immutable_unset`, and
  `require_fact` as blocking cards whose buttons POST to the respective respond endpoints
  and let the SSE stream resume.
- The inference side panel shows inferences with an expand-derivation click and a delete
  button; the promote-to-fact affordance is gone (the endpoint is removed in Step 9).
- Editing a fact value in the panel writes the blob but does **not** cascade to inferences
  (deferred to Step 9). Inferences still key on integer `source_fact_ids`.
- `GET /api/characters/{id}/facts` returns the masked blob; `PUT
  /api/characters/{id}/facts` writes one leaf by path. The legacy row CRUD endpoints and
  their tests are gone; `test_api_facts.py` covers the new endpoints.
- The legacy `implication.py` router endpoints remain mounted but are now unreferenced by
  the UI (full removal is Step 9+). No frontend code calls them.

---

## Test Plan

### `tests/integration/test_api_facts.py` — rewrite

**Delete** every test targeting the removed row CRUD endpoints (create/put-by-id/
patch-mutability/patch-category/delete-by-id and their 404/409 cases).

**Add** (each creates a character via the integration `character` fixture):

- `test_get_facts_returns_empty_blob_for_new_character` — GET returns `{}` (200).
- `test_get_facts_returns_masked_blob` — seed via `set_facts(db, char.id, {...})`; GET
  returns the same masked structure.
- `test_get_facts_unknown_character_returns_404`.
- `test_put_fact_writes_string_leaf_and_returns_blob` — PUT
  `{"path": "Character.Identity.Name", "value": "Sarah"}` → 200; response blob has
  `Character.Identity.Name.Value == "Sarah"`.
- `test_put_fact_coerces_integer_leaf` — PUT `{"path": "Character.Identity.Age", "value":
  "34"}` → stored `Value` is the int `34`.
- `test_put_fact_coerces_enum_case_insensitively` — PUT a `State-Of-Mind.Mood` value
  `"anxious"` → stored `"Anxious"`.
- `test_put_fact_invalid_enum_returns_422` — PUT an out-of-constraint mood → 422.
- `test_put_fact_invalid_integer_returns_422` — PUT `"abc"` to an Integer leaf → 422.
- `test_put_fact_unknown_path_returns_422` — PUT `{"path": "Nope.Not.Here", ...}` → 422.
- `test_put_fact_unknown_character_returns_404`.

(Retain the `GET /{id}/inferences` tests if this file has any; otherwise they live in the
inference test files and are untouched.)

### `tests/frontend/chat.test.js` — changes

**Delete these describe blocks in full** (all cover removed exports/cases):
- `buildNotificationFromSidechannel` sub-tests for `implication` and
  `new_inference_probabilistic`.
- `removeViolation`.
- The `API helpers` block's `apiAcceptImplication` / `apiIgnoreImplication` /
  `apiAcceptInference` / `apiIgnoreInference` tests.
- `Phase 4 fact API helpers`, `Phase 4 inference promotion API helper`,
  `Phase 4 buildNotificationFromSidechannel — mutability implication`.
- `Phase 5 buildNotificationFromSidechannel — experience_update`.
- `removeContradictedExperiences`.
- `Phase 6 buildNotificationFromSidechannel — extraction_applied`,
  `Phase 6 buildNotificationFromSidechannel — implicit_fact_proposed`,
  `Phase 6 API helpers`.

Update the top-of-file import list to drop every deleted symbol and add the new ones.

**Keep** `parseSSEBlock`, `parseSSEBlocks`, `sseStateToLabel`, the `contradiction`
notification test, `sortExperiences`, `buildScoreMap`, `buildProposalList`, and the
Phase-5 experience API helper tests.

**Add** `buildNotificationFromSidechannel` tests:
- `test_buildNotification_fact_update_fluid_shape` — payload `{type:'fact_update_fluid',
  turn_id, path, value}` → `scType==='fact_update_fluid'`, `path`, `value` preserved.
- `test_buildNotification_fact_update_mutable_seeds_editValue_from_proposed`.
- `test_buildNotification_fact_update_immutable_unset_seeds_editValue_from_proposed`.
- `test_buildNotification_require_fact_seeds_editValue_from_suggested_value`.
- `test_buildNotification_require_fact_editValue_empty_when_suggested_value_absent`.
- `test_buildNotification_inference_proposed_carries_inference_object`.
- `test_buildNotification_returns_null_for_removed_implication_type` — payload
  `{type:'implication'}` now returns `null`.

**Add** `flattenSchema` tests:
- `test_flattenSchema_marks_leaves_and_groups` — a 2-level schema; assert grouping rows
  have `isLeaf===false` and leaf rows carry `type`/`mutability`.
- `test_flattenSchema_depth_increases_per_level`.
- `test_flattenSchema_leaf_carries_constraint_for_enum`.
- `test_flattenSchema_preserves_key_order`.

**Add** `lookupBlobValue` tests:
- `test_lookupBlobValue_returns_value_for_set_leaf`.
- `test_lookupBlobValue_returns_undefined_for_unset_path`.
- `test_lookupBlobValue_returns_undefined_for_partial_path`.

**Add** `buildVisibleFactRows` tests:
- `test_buildVisibleFactRows_attaches_value_to_leaves`.
- `test_buildVisibleFactRows_hides_descendants_of_collapsed_group`.
- `test_buildVisibleFactRows_shows_collapsed_group_row_itself`.
- `test_buildVisibleFactRows_leaf_value_undefined_when_unset`.

**Add** `schemaLeaf` tests:
- `test_schemaLeaf_returns_leaf_for_valid_path`.
- `test_schemaLeaf_returns_null_for_unknown_path`.

**Add** API-helper tests (mock `global.fetch`):
- `test_apiGetSchema_gets_slash_api_schema`.
- `test_apiGetFactBlob_gets_character_facts_url`.
- `test_apiSetFactValue_puts_path_and_value`.
- `test_apiRespondRequireFact_posts_value`.
- `test_apiRespondSetFact_posts_action_and_value`.
- `test_apiRespondSetFact_defaults_value_null`.

### `tests/frontend/chat-component.test.js` — changes

**Delete these describe blocks in full** (cover removed handlers/state):
- `factsByCategory`.
- `saveFact`.
- `acceptInference`, `ignoreInference`, `acceptImplication`, `ignoreImplication`.
- `Phase 6 SSE extraction events`, `undoUserFact`, `deleteExtractedFact`,
  `acceptImplicitFact`, `ignoreImplicitFact`.
- `sendMessage SSE sidechannel — experience_update`.
- `mutabilityIcon`.

**Keep** `endSession`, `sendMessage SSE message event — active experience tracking`,
`newSession`, `proposal lifecycle`, `dismissNotification`.

**Update** `newSession` test(s) if they assert on `facts` being reset — assert
`factsBlob.value` resets to `{}` and `collapsedGroups.value` is empty instead.

**Add** (using the `setupComponent(routes)` + `makeStreamResponse` helpers already in the
file):

- `test_loadSchema_populates_schema_ref` — `setupComponent({ '/api/schema': {Character:{...}} })`;
  after `await vm.loadFacts`-adjacent tick, `vm.schema.value` has the fetched tree. (Call
  `loadSchema` directly if exposed.)
- `test_loadFacts_populates_factsBlob` — route `/facts` to a blob object; call
  `vm.loadFacts()`; assert `vm.factsBlob.value` equals it.
- `test_visibleFactRows_reflects_schema_and_blob` — set `schema.value` and `factsBlob.value`;
  assert the computed rows include a leaf with the expected `value`.
- `test_toggleGroup_collapses_and_expands` — call `vm.toggleGroup('Character')` twice;
  assert membership in `collapsedGroups` flips.
- `test_saveLeaf_puts_value_and_reloads` — stub fetch; `vm.saveLeaf({path:'Character.Identity.Name'})`
  with `leafEdits['Character.Identity.Name']='Sarah'`; assert a PUT to `/facts` and a
  follow-up GET.
- `test_leafType_and_leafConstraint_read_from_schema`.
- `test_deleteInference_deletes_and_reloads`.
- `test_sendMessage_fact_update_fluid_pushes_quiet_notification` — stream a
  `fact_update_fluid` sidechannel; assert a `scType==='fact_update_fluid'` message is
  pushed and `generating` is not forced (quiet).
- `test_sendMessage_fact_update_mutable_pushes_blocking_card_and_stops_spinner` — assert
  card pushed and `generating.value === false`.
- `test_sendMessage_require_fact_pushes_blocking_card`.
- `test_sendMessage_inference_proposed_pushes_notification`.
- `test_resolveMutable_accept_posts_action_and_dismisses` — pre-push a mutable card into
  `messages`; `vm.resolveMutable(notif, 'accept')`; assert POST to `/set-fact/respond` with
  `action:'accept'` and the card removed.
- `test_resolveMutable_edit_posts_editValue`.
- `test_resolveImmutable_dismiss_posts_dismiss`.
- `test_resolveRequireFact_confirm_posts_editValue`.
- `test_resolveRequireFact_notnow_posts_null`.
- `test_sendMessage_done_reloads_facts_and_inferences` — assert both `/facts` and
  `/inferences` GETs fire after a `done` event.

---

## Edge Cases

- **Schema fetched before the blob renders.** `visibleFactRows` depends on both
  `schema.value` and `factsBlob.value`. If the schema request is slow, the panel shows
  "Loading schema…" until `loadSchema()` resolves. `buildVisibleFactRows({}, blob, …)`
  returns `[]`, so an empty schema degrades gracefully rather than throwing.
- **`leafEdits` and stale inputs.** `syncLeafEdits()` reseeds every leaf input from the blob
  after each `loadFacts()`. A `fact_update_fluid` event during a turn updates the blob only
  after the `done`→`loadFacts()` refresh, so a fluid change shows in the tree once the turn
  completes, not mid-stream. This is acceptable; the quiet card communicates the change
  immediately.
- **Blocking card while the SSE stream is open (ASGITransport limitation).** The blocking
  cards rely on the SSE stream staying open while the user acts, then resuming after the
  respond POST. The **frontend** tests exercise this only at the unit level (pushing a card
  object and calling the resolver), never against a live buffered transport — so no
  ASGITransport hazard exists in the JS suite. True end-to-end suspend/resume coverage lives
  in the Python `test_set_fact_approval_live.py` / `test_require_fact_live.py` files
  (Steps 6–7, unchanged here).
- **Enum leaf with a value not in `Constraint`.** If the blob holds a legacy or
  out-of-constraint enum value, the `<select>` will have no matching `<option>` and render
  blank. Saving then coerces (or 422s) server-side. Because the whole DB is wiped on first
  run of Facts v2, this only arises from manual tinkering; no special handling is added.
- **Integer input yields a string via `v-model`.** `leafEdits[path]` is always a string
  (HTML input value). `apiSetFactValue` sends it as a string; the backend `PUT` coerces
  `Integer` leaves with `int(value)` and 422s on non-numeric input. The `type="number"`
  input reduces (does not eliminate) bad input.
- **`reject`/`dismiss` action mismatch.** The mutable card sends `reject`; the
  immutable-unset card sends `dismiss`. Sending the wrong verb is prevented structurally by
  per-card handlers (`resolveMutable` vs `resolveImmutable`), matching Step 7's note that
  the server trusts the frontend to send the right action per card type.
- **Double-resolve of a blocking card.** If a user clicks twice quickly, the second POST
  hits the respond endpoint after `resolve_gate` already fired and receives a 409. The
  handler sets `_loading` during the first call and dismisses the card on success, so the
  button is disabled/gone before a second click normally lands; a stray 409 is swallowed
  (no user-visible error is required).
- **Editing a fact does not revalidate inferences (transitional).** Because inferences
  still key on integer `source_fact_ids` until Step 9, `saveLeaf` cannot cascade a
  path-based revalidation. A leaf edit that logically invalidates an inference leaves that
  inference in place until Step 9 wires path cascade. Documented and accepted.
- **`inference_proposed` appears twice.** The quiet card is pushed into the chat stream and
  the inference also lands in the side panel after the `done`→`loadInferences()` refresh.
  This is intentional: the card is the transient "just happened" signal; the panel is the
  durable list.

---

## Post-Implementation Cleanup Tasks

_(Populated by `/review-step` after implementation.)_
