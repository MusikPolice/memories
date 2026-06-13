# Facts v2 — Schema-Constrained Fact Hierarchy

## Problem Statement

The current fact model is a flat key-value store within three top-level categories
(`character`, `user`, `setting`). In practice the evaluator and pre-turn extractor are
free to invent any key they like, and they do: `emotional_state`, `current_mood`,
`mood_right_now`, `feeling`, `character_mood` have all appeared in real sessions for the
same concept. There is no taxonomy enforcement, no signal about what kinds of facts *can*
exist, and no constraint on the evaluator's creative naming. The result is an increasingly
disorganised fact store whose inconsistency compounds over time — keys that should merge
stay separate, the sidechannel becomes an unreadable list, and the evaluator loses its
ability to detect contradictions reliably (it misses `emotional_state=anxious`
contradicting `current_mood=calm` because they look like different facts).

The current system also allows the evaluator to *create* new facts (via the `implication`
verdict) and allows the user to *promote* inferences to facts. Both flows compound the
problem: the evaluator invents whatever path it feels like, and the schema drifts further
with each session.

The fix is not better prompting. Prompting is a negotiation the evaluator wins eventually.
The fix is a **deterministic schema** that defines every fact path that can possibly exist.
The schema lives in a JSON file in the repository. The evaluator can only *update values*
for paths already defined in the schema — it cannot create new paths. New fact types are
added by editing the JSON file between sessions. This makes schema evolution a deliberate,
versioned act rather than an emergent side effect of roleplay.

This document supersedes the two deferred-work items in `docs/plan.md`:
- "Hierarchical fact organisation and auto-discovered clusters"
- "Numerical ranges for facts"

---

## Core Proposal

The fact store is a **closed recursive tree** defined entirely by `fact_schema.json`.
Every valid fact path — from top-level category down to individual leaf key — is
pre-declared in that file. No pass (World Builder, Character Evaluator, or otherwise)
can create a path that does not already exist in the schema.

A leaf node is identified by the presence of a `Type` field. Any node without `Type` is
a grouping that can contain further groupings or leaves. The tree has no fixed depth —
groupings nest to whatever level makes semantic sense for the world being modelled. In
practice this will be two to four levels, but no application logic encodes that
expectation.

The evaluator's job is to write **values** to existing leaf paths. Schema evolution —
adding new groupings or leaves — is a deliberate act performed by editing the JSON file
between sessions. A future `suggest_fact` tool may allow LLM passes to propose new paths
for user approval at runtime, but that is deferred work (see Open Questions).

---

## Mutability Tiers

Every leaf in the schema is assigned one of three mutability levels. These are enforced
deterministically by the server — the evaluator does not decide whether a mutation is
permitted; the schema does.

### `immutable`

A `set_fact` call targeting an immutable path is **immediately rejected** by the server
with no user-facing side effect — the LLM receives a tool error and the turn continues
as if the call was never made. The character cannot sneak an immutable change through by
calling `set_fact`; the tool simply refuses.

If the character's response *implies* a value that contradicts an already-set immutable
fact, that is detected by the Character Evaluator as a contradiction and triggers
regeneration exactly as today — that flow does not go through `set_fact` at all.

If an immutable path has no value yet, the correct tool is `require_fact(path, reason,
suggested_value?)` rather than `set_fact`. `require_fact` pauses the turn, surfaces a
blocking card to the user with the path's `Description` as guidance and the LLM's
suggestion (if provided) pre-filled as an editable default. The value is written and
locked only once the user explicitly confirms.

Note: the World Builder has author-level authority to overwrite any fact including
immutable ones. It does this via a separate tool, `author_set_fact(path, value)`, which
bypasses all mutability checks. The Character Evaluator and Character LLM use
`set_fact(path, value)`, which enforces mutability. Tool-list scoping is the primary
enforcement mechanism — each pass is given only the tools listed in its system prompt,
so the Character LLM never sees `author_set_fact` and cannot call it.

Examples: `Character.Identity.Name`, `Character.Identity.Pronouns`,
`Character.Appearance.Body.Height`.

### `mutable`

A `set_fact` call on a mutable path does not apply immediately. The character's response
is **withheld** until the user acts. A blocking sidechannel card shows the proposed
change: accept, edit, or reject.

- **Accept**: the value is written to the blob; the withheld response is delivered.
- **Edit**: the user's amended value is written instead; the Character LLM is re-invoked
  with the updated fact in context (skipping the World Builder — the user's original
  message was already processed); the new response replaces the withheld one.
- **Reject**: the value is not written; the Character LLM is re-invoked without the
  proposed change in context (again skipping the World Builder); the new response
  replaces the withheld one.

Regeneration is required for edit and reject because the withheld response was built on
a world state the user has just overruled — delivering it would produce an immediate
inconsistency between what the character said and what the world model reflects.

Examples: `Character.Appearance.Outfit.Top`, `Setting.Location.Name`.

### `fluid`

A `set_fact` call on a fluid path is accepted immediately with no approval gate. The
value is written to the blob and a non-blocking info notification is shown in the
sidechannel so the user can see what changed, but no action is required of them.

Examples: `Character.State-Of-Mind.Mood`, `Character.State-Of-Mind.Energy`,
`setting.environment.*`.

---

## Schema as a JSON File

The schema lives in the source repository at `src/memories/fact_schema.json`. The backend
reads it at startup and uses it to:
- Build the `## Fact Schema` section injected into evaluator and extractor prompts
- Validate evaluator fact proposals before writing to the DB (path must exist; mutability
  must permit the operation)
- Apply as a mask when reading facts from the DB (unknown paths silently dropped)
- Drive the UI fact-grouping and type-aware controls

Keeping the schema in source control means changes are reviewed, versioned, and deployed
deliberately. There is no `fact_schema` database table. When the schema needs to evolve,
the JSON file is edited; the mask behaviour handles any in-DB facts that no longer match.

**Intentional limitation:** a single schema applies to all characters. Paths that do not
make sense for a given character (e.g. `Character.Appearance.Outfit.*` for a disembodied
narrator) simply remain unpopulated in the blob — blank paths are harmless. Future work
may support multiple named schemas with a per-character selection at creation time, but
that is out of scope here.

### Schema file structure

The schema is a recursive tree. Any node that contains a `Type` field is a **leaf** — a
fact definition. Any node without `Type` is a **grouping** that can contain further
groupings or leaves. Depth is not fixed; groupings can nest to whatever depth makes
semantic sense.

Each leaf defines exactly three fields:

| Field | Purpose |
|---|---|
| `Type` | `String`, `Integer`, or `Enum` |
| `Mutability` | `Immutable`, `Mutable`, or `Fluid` (see Mutability Tiers) |
| `Description` | Plain-English guidance for the LLM on how to interpret and populate this fact |

For `Enum` leaves an additional `Constraint` field lists the allowed values. The
`Constraint` list is included verbatim in the prompt so the LLM sees the valid options
at the point it decides what value to pass.

**Enum validation policy:** the `set_fact` / `author_set_fact` handler applies
coercion before validation — a case-insensitive exact match against the `Constraint`
list is attempted first (e.g. `"calm"` → `"Calm"`). If coercion succeeds, the
normalised value is written. If no match is found (e.g. `"Apprehensive"`,
`"Anxious/Guarded"`), the tool call returns a hard error and the LLM is expected to
retry with a valid value. This policy may need to be softened to a silent drop if hard
failures prove disruptive in practice.

In the tool-calling model, a failed `set_fact` call returns an error message to the LLM
within the same invocation (e.g. *"'Apprehensive' is not a valid value. Valid values are:
Calm, Anxious, ..."*). The LLM sees the error and can immediately retry with a corrected
value — it does not re-evaluate the character's prose from scratch. The risk of repeated
identical failures is therefore low. The Step 0 PoC should include a case where a tool
call returns a validation error, to confirm the model uses the error message to self-
correct rather than retrying with the same invalid value.

**Tool-call retry cap.** The server controls the tool-call loop, not the LLM. To prevent
an LLM from retrying a failing tool call in perpetuity, the server enforces a maximum
number of tool-call round-trips per pass (suggested default: **10**, configurable via
`MAX_TOOL_CALL_ROUNDS`). If the cap is reached, the server stops re-invoking the LLM
and resolves the pass by one of two fallback strategies:

- **Non-terminal failure** (e.g. a `set_fact` that cannot be coerced): drop the failed
  call as a no-op, inject a system message instructing the LLM to call a terminal tool
  (`report_pass` or `report_contradiction`) and do not count this as another round-trip.
- **Terminal failure** (cap reached with no terminal tool called): treat the pass as
  `report_pass` and log a warning. This is the same defensive posture as the existing
  `EvaluatorParseError` path — deliver the response unverified rather than hanging.

The cap applies independently to each pass (World Builder, Character LLM, Character
Evaluator). The Step 0 PoC must verify the cap fires correctly and the fallback resolves
cleanly.

```json
{
  "Character": {
    "Identity": {
      "Name": {
        "Type": "String",
        "Mutability": "Immutable",
        "Description": "The character's given name, typically in the format 'First Last'"
      },
      "Age": {
        "Type": "Integer",
        "Mutability": "Immutable",
        "Description": "The character's age in years. Subtracting this from Setting.Temporal.Current-Year yields the character's birth year"
      }
    },
    "Appearance": {
      "Body": {
        "Height": {
          "Type": "String",
          "Mutability": "Immutable",
          "Description": "The character's height, expressed in the user's preferred unit"
        },
        "Hair-Colour": {
          "Type": "String",
          "Mutability": "Mutable",
          "Description": "The character's natural or current hair colour. Descriptive modifiers are encouraged (e.g. 'Chestnut Brown', 'Strawberry Blonde')"
        }
      },
      "Outfit": {
        "Top": {
          "Type": "String",
          "Mutability": "Mutable",
          "Description": "What the character is wearing on their upper body"
        },
        "Bottom": {
          "Type": "String",
          "Mutability": "Mutable",
          "Description": "What the character is wearing on their lower body"
        },
        "Shoes": {
          "Type": "String",
          "Mutability": "Mutable",
          "Description": "The character's footwear"
        }
      }
    },
    "State-Of-Mind": {
      "Mood": {
        "Type": "Enum",
        "Constraint": ["Calm", "Anxious", "Angry", "Sad", "Joyful", "Neutral", "Guarded", "Excited"],
        "Mutability": "Fluid",
        "Description": "The character's dominant emotional state at this moment in the narrative"
      },
      "Energy": {
        "Type": "Enum",
        "Constraint": ["Exhausted", "Tired", "Neutral", "Alert", "Energised"],
        "Mutability": "Fluid",
        "Description": "The character's physical energy level"
      }
    }
  },
  "User": {
    "Identity": {
      "Name": {
        "Type": "String",
        "Mutability": "Immutable",
        "Description": "The user's name as they prefer to be addressed by the character"
      }
    }
  },
  "Setting": {
    "Temporal": {
      "Current-Year": {
        "Type": "Integer",
        "Mutability": "Mutable",
        "Description": "The year in which the narrative is currently set"
      }
    },
    "Location": {
      "Name": {
        "Type": "String",
        "Mutability": "Mutable",
        "Description": "The name of the current location (e.g. 'Chicago Memorial Hospital', 'The Crown pub')"
      },
      "Space": {
        "Type": "Enum",
        "Constraint": ["Interior", "Exterior"],
        "Mutability": "Mutable",
        "Description": "Whether the scene takes place inside a building or outdoors"
      },
      "Description": {
        "Type": "String",
        "Mutability": "Fluid",
        "Description": "A brief description of the immediate surroundings and atmosphere"
      }
    }
  }
}
```

This is an illustrative starting point, not the final schema. The actual `fact_schema.json`
will be fleshed out fully before implementation begins.

Key types:
- `String` — free-form text
- `Integer` — whole number
- `Enum` — value from the `Constraint` list; evaluator must choose from the listed options

---

## Per-Turn Architecture: Three Components

Every turn involves two LLM evaluator passes with fundamentally different authority over
the fact store. Their roles diverge under this design in a way the current system does not
distinguish.

### The framing: author vs. participant

> The **user is the author** of the story. The **character is a participant** in it.

This single distinction drives both passes. The author can assert anything about the world
and it becomes true. The participant can only act within the world as it exists, subject
to what they are physically and narratively capable of changing.

---

### Pass 1 — World Builder (pre-turn, runs on the user's message)

The World Builder is a refactor of the existing `extraction_service.py` from Phase 6,
not a build-from-scratch service. Significant parts of the existing extractor — the
prompt structure, the `run_fact_extractor` entry point, the `ExtractionResult` models,
and the `run_turn` integration — can be retained and adapted. The primary changes are:
switching from a structured JSON verdict to tool calls (`author_set_fact`), replacing
the four-tier explicit/implicit model with the simpler schema-constrained authority
model described below, and renaming the service accordingly.

Runs before the character LLM is invoked. Its job is to extract any facts — explicit or
implicit — asserted by the user's message and update the world state accordingly.

**Authority: unrestricted.** The World Builder may write to any schema path regardless of
mutability. The user is the author; their assertions are ground truth. A user saying
*"I brushed a lock of her auburn red hair back behind her ear"* has just established
`Character.Appearance.Body.Hair-Colour = "Auburn Red"` and possibly
`Character.Appearance.Body.Hair-Length = "long"` — even if those facts were previously
set to different values, even if they were marked `Immutable`. The author can retcon.

The World Builder does not perform contradiction detection. If the user's message
contradicts an existing fact, the fact is updated to match the user's assertion. A quiet,
non-blocking sidechannel notification informs the user of what was extracted and changed,
giving visibility without interrupting the flow.

**Tool use.** Rather than producing a structured JSON verdict, the World Builder expresses
its fact writes as calls to `author_set_fact`. Tool calling is the firm approach here because:

- The LLM can call `author_set_fact` multiple times per turn — one call per extracted
  fact — rather than producing a single monolithic JSON blob
- Each `author_set_fact` call passes through a deterministic server-side handler that can:
  - Validate the path exists in the schema
  - Validate the value matches the declared `Type` (and `Constraint` for Enum)
  - Trigger recalculation of any `Derived` facts that depend on the updated path
  - Cascade invalidation to any Inferences that sourced from the changed fact
- No mutability check is applied — the user is the author; their prose is ground truth
- The tool call log becomes a natural audit record of what the World Builder extracted

`author_set_fact` is only listed in the World Builder's system prompt. The Character
Evaluator and Character LLM never see it and cannot call it.

Example tool calls the World Builder might make from *"I crossed the room and kissed her"*:

```
author_set_fact("Setting.Location.Space", "Interior")
author_set_fact("Character.State-Of-Mind.Mood", "Surprised")
author_set_fact("User.State.Proximity-To-Character", "Close")
```

The schema defines what facts *can* be extracted; the World Builder decides which ones
*are* implied by the prose.

**Scope.** The World Builder only writes facts. It does not generate character responses,
run contradiction logic, propose inferences, or surface approval prompts to the user.
It runs, updates the world state, and passes the updated context to the character LLM.

---

### Pass 2 — Character LLM (generates the character's response)

Receives the full world state (Facts, Inferences, active Experiences) updated by the
World Builder and generates the character's in-character prose response.

**Tool list (proposed):**

- **`require_fact(path, reason, suggested_value?)`** — called when the character
  realises mid-generation that it needs a value for an unset `Immutable` path and cannot
  produce a coherent response without it. Pauses generation, surfaces a blocking card to
  the user, and resumes once the user confirms a value. This is the primary caller of
  `require_fact`; the Character Evaluator may catch the same gap as a fallback, but the
  Character LLM is better positioned to detect it early.
- **`propose_inference(statement, derivation)`** *(if Option B is adopted)* — called when
  the character reasons that something follows from known Facts. See Inference Generation
  open question.

The Character LLM does **not** have access to `set_fact` or `author_set_fact`. Fact
writes from the character's perspective are handled by the Character Evaluator after the
response is complete.

#### Requesting an unset immutable value from the user

A special case arises when the character needs to reference an `Immutable` fact that has
never been given a value. The character cannot invent one — `set_fact` will reject the
call deterministically — but it also cannot produce a coherent response without it.

The canonical example: a brand-new character with no name set. The user opens with *"Hi,
my name is Steve, what's your name?"* The character has to respond, but
`Character.Identity.Name` is unset and immutable. The character must not make up a name.

The solution is `require_fact(path, reason, suggested_value?)` (name TBD; see Open
Questions), a tool in the Character LLM's tool list. The Character LLM calls it
mid-generation when it realises it cannot produce a coherent response without a value
for an unset immutable path. The Character Evaluator may catch the same gap as a
fallback — if the character's finished response references an unset immutable path
without having called `require_fact` — but the Character LLM is the primary caller.

The optional `suggested_value` argument lets the character offer a plausible starting
point without asserting it. The blocking card pre-fills that suggestion so the user can
confirm it with one action, edit it, or clear it entirely. The value is never written
until the user explicitly confirms — the character's suggestion is a convenience, not
an act of authorship.

**Flow:**
1. Character LLM is invoked. Before generating any prose, it recognises that
   `Character.Identity.Name` is unset and calls `require_fact("Character.Identity.Name",
   "I need my name to introduce myself", suggested_value="Elena")` instead of generating
   a response. Tool calls and prose responses are mutually exclusive per invocation — the
   LLM calls the tool *instead of* generating text, not after a partial response.
2. The server receives the tool call and does not respond immediately. It emits a blocking
   sidechannel card via SSE: *"The character needs a name. Suggested: Elena — confirm,
   edit, or replace."* The SSE stream stays open; the LLM is paused awaiting the tool
   result.
3. The user confirms, edits, or replaces the suggestion.
4. The server writes the value via `author_set_fact`, locking it permanently (immutable),
   and returns the confirmed value as the tool result.
5. The LLM resumes with the tool result in its message history and now generates its
   prose response with the fact populated.
6. The turn continues normally through the Character Evaluator pass.

#### Inference generation

Two complementary mechanisms operate in parallel:

**Option A — Evaluator-observed (baseline, always active).** The Character Evaluator
reads the character's response and identifies assertions derivable from Facts but not yet
stored, proposing them via `new_inference_logical` / `new_inference_probabilistic` tool
calls. This is the proven current approach and runs regardless of what the Character LLM
does.

**Option B — Character tool (additive).** The Character LLM is given `propose_inference`
in its tool list. When it reasons that something follows from what it knows — *"She
mentioned her occupation is surgery and she looks exhausted — I infer she just came off
a long shift"* — it can call the tool directly. The Inference IS the character's thought
rather than an external observation of it. If the LLM calls the tool, the inference is
captured early; if it does not, the Evaluator covers it via Option A.

The two options are not mutually exclusive and both run by default. A future options menu
may allow toggling each independently, since the user's choice of LLM will materially
affect how reliably Option B fires in practice.

### Pass 3 — Character Evaluator (post-character, runs on the character's response)

Runs after the character LLM generates a response, before the user sees it. "Conflict
Detector" undersells this pass — it checks for conflicts, yes, but it is also the home
of the character's capacity for reasoning: deriving Inferences from the current world
state and the character's own response.

#### Facts vs. Inferences: an epistemological distinction

> **Facts** are *what is known* — objective world state, established by the user or
> extracted from user prose by the World Builder. Authoritative and schema-bound.
>
> **Inferences** are *what is believed to be true* — the character's subjective
> conclusions drawn from Facts. Traceable but not authoritative. Only the character
> generates them, because only the character has beliefs.

This framing matters: the World Builder has no business generating Inferences (it deals
in facts, not beliefs), and the user does not generate Inferences (they are the author,
not a reasoner within the fiction). Inferences are the shared responsibility of the
Character LLM and the Character Evaluator. The Character LLM is the preferred source —
it reasons in first-person and can call `propose_inference` directly when it draws a
conclusion from what it knows. The Character Evaluator observes the finished response
and proposes any inferences the character did not surface itself. The hope is that the
character does most of the heavy lifting; the evaluator is there to pick up the slack.

#### Conflict detection (fact writes)

**Authority: mutability-constrained.** The character is in the story, not above it.

| Situation | Server action |
|---|---|
| Character's response conflicts with an `Immutable` path already set | Contradiction — regenerate |
| Character implies a value for an `Immutable` path that is **unset** | Surface blocking card with character's suggested value; branch on user action (see below) |
| Character implies a value for a `Mutable` path | Call `set_fact`; surface mutable-update card to user (accept / edit / reject) |
| Character implies a value for a `Fluid` path | Call `set_fact`; apply silently; quiet sidechannel notification |
| Character invents a detail with no matching schema path | Call `propose_inference`; surface as a new inference for user review |
| Character's response is consistent with all facts | Pass — deliver response |

**Immutable-unset branching:** when the evaluator detects a character response that uses
an invented value for an unset `Immutable` path (either because the Character LLM called
`require_fact` mid-generation, or because the evaluator catches it post-generation as a
fallback), a blocking card surfaces the character's suggested value. The outcome branches
on the user's action:

- **Accept** (user confirms the character's suggested value) — the value is written and
  locked immutably; the existing response is delivered as-is. No regeneration is needed
  because the character already used the correct value.
- **Edit** (user supplies a different value) — the value is written and locked; the
  Character LLM is re-invoked with the corrected value in context; the new response
  replaces the original. Same regeneration flow as the current `implication` accept path.
- **Dismiss** — no value is written; the response is delivered with the invented value
  treated as an ungrounded detail (same as the current `implication` ignore path).

The Character Evaluator uses `set_fact` for schema-path writes. For details that cannot
be captured as Facts — because no schema path exists — the evaluator calls
`propose_inference` instead. This reframes off-schema inventions as character beliefs
(Inferences) rather than world facts, which is semantically appropriate: the detail
exists in the character's mind, but it has not been established as ground truth by the
user. The user can accept it as an Inference or, if they want it as a Fact, add the path
to the schema and set it explicitly. `propose_inference` is the direct replacement for
the old `implication` verdict for the off-schema case.

**What the Character Evaluator cannot do:**
- Write to a path not in the schema (uses `propose_inference` instead)
- Unilaterally change an `Immutable` fact (triggers contradiction loop instead)
- Create new schema paths

#### Cascade re-evaluation when Facts change

When the World Builder (or the user via the sidechannel) updates a fact, all active
Inferences that sourced from that fact path need re-evaluation. This already works today
via `cascade_on_fact_edit()` in `inference_service.py`, which BFS-walks downstream
Inferences and re-invokes the evaluator on each. With Facts now stored as paths rather
than integer IDs, `source_fact_ids` becomes `source_fact_paths` (see Implementation
Sketch, Step 6).

One new concern: the World Builder may write **multiple facts in a single turn**. Firing
a cascade after each individual `set_fact` call would be wasteful and could produce
intermediate states mid-cascade that interfere with each other. The cascade should be
batched: collect all fact paths written by the World Builder for the turn, then run a
single cascade pass over all affected Inferences once the World Builder is done.

**Cascade performance.** `cascade_on_fact_edit` already operates in-memory: it loads
the full inference set for the character and BFS-walks it in Python, never issuing
per-inference SQL queries. Replacing integer `source_fact_ids` with path-string
`source_fact_paths` changes the matching logic from `int in list` to `str in list` —
both are trivially fast for the inference counts expected in this application. No SQL
index tuning or query restructuring is needed; `source_fact_paths` is stored as a JSON
array string in the DB and deserialized at load time.

---

## What Is Removed

These features are removed or significantly curtailed in this design:

- **`implication` verdict** — replaced by three mechanisms: the World Builder pass (for user-asserted facts), the Character Evaluator's mutability-gated `set_fact` tool call (for character-implied facts that match a schema path), and the `propose_inference` tool call (for character-invented details that have no schema path, which are stored as Inferences rather than Facts)
- **Inference promotion to Fact** — the `/inferences/{id}/promote` endpoint and its UI
  button are removed. Inferences remain as derived conclusions; they cannot be elevated
  to facts. If a user wants a derived conclusion to become a ground truth, they add it
  to the schema JSON and set it as a Fact manually.
- **Evaluator-invented fact paths** — the evaluator's output schema no longer includes a
  free-form `suggested_fact.key` field; all proposals must reference a schema path.
- **Experience deletion on contradiction** — the `experience_update` verdict and its
  immediate-delete behaviour are removed. Experiences are now an immutable append-only
  log; superseded Experiences are never deleted, only joined by newer ones. The
  session-end evaluator adds new Experiences naturally; retrieval tie-breaking (most
  recent wins) handles cases where multiple similar Experiences match the same query.

---

## Fact Storage — JSON Blob per Character

Facts are stored as a single JSON blob per character rather than individual DB rows. The
current `facts` table (one row per fact) is replaced by a `character_facts` table with
one row per character and a single `facts_json` TEXT column.

### DB schema

```sql
CREATE TABLE character_facts (
    character_id INTEGER PRIMARY KEY REFERENCES characters(id),
    facts_json   TEXT NOT NULL DEFAULT '{}',
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Blob structure

The blob mirrors the schema tree exactly. Grouping nodes are preserved; leaf nodes gain
a `Value` field when they have been populated. Leaves with no value yet are absent from
the blob entirely — the blob is sparse.

```json
{
  "Character": {
    "Identity": {
      "Name": { "Value": "Sarah" },
      "Age":  { "Value": 33 }
    },
    "Appearance": {
      "Body": {
        "Hair-Colour": { "Value": "Chestnut Brown" }
      },
      "Outfit": {
        "Top":   { "Value": "Blue surgical scrubs" },
        "Shoes": { "Value": "White clogs" }
      }
    },
    "State-Of-Mind": {
      "Mood":   { "Value": "Anxious" },
      "Energy": { "Value": "Tired" }
    }
  },
  "User": {
    "Identity": {
      "Name": { "Value": "Jon" }
    }
  },
  "Setting": {
    "Temporal": {
      "Current-Year": { "Value": 2026 }
    },
    "Location": {
      "Name":  { "Value": "Chicago Memorial Hospital" },
      "Space": { "Value": "Interior" }
    }
  }
}
```

The blob stores only `Value` — it does not duplicate `Type`, `Mutability`, or
`Description` from the schema file. Those are always read from `fact_schema.json` at
runtime and merged with the blob values for prompting and display.

### Why a blob is sufficient

Facts are always read and written as a complete set for a character — there is no use
case that requires fetching a single fact in isolation. The evaluator and character LLM
receive all facts at once; the UI displays all facts at once; updates patch a single
`Value` field in the blob and write it back.

Individual fact identity (formerly an integer `fact_id`) is replaced by **path strings**
(e.g. `"Character.Identity.Name"`). The `source_fact_ids` column in the `inferences`
table becomes `source_fact_paths` storing a JSON array of path strings.

### Clearing a fact value

"Deleting" a fact means removing its `Value` entry from the blob — the schema path
remains defined; only the stored value is erased. The repository exposes this as
`clear_fact(character_id, path_tuple)`, which removes the leaf's `Value` key from the
blob and writes it back. The existing `DELETE /api/characters/{id}/facts/{fact_id}`
endpoint is replaced by `DELETE /api/characters/{id}/facts/{path}` where `path` is the
dot-notation path string (e.g. `Character.Identity.Name`), URL-encoded. Cascade
behaviour on clear is the same as on delete: all inferences with this path in
`source_fact_paths` are marked `invalidated`.

---

## Schema Masking at Read Time

When `get_facts(character_id)` reads the blob from the DB, it applies the schema as a
mask before returning:

1. Walk every node in the blob tree
2. If a node path exists in the schema as a leaf (has `Type`) → keep
3. If a node path exists in the schema as a grouping → keep (recurse into children)
4. If a node path does not exist in the schema at all → silently drop

This handles schema evolution without migrations:
- A grouping renamed or removed → blob entries under that path silently disappear on next read
- Molly (character id=2, created before this change) has flat-key facts; none match any
  schema path and all are silently dropped on read
- Facts for a leaf removed from the schema disappear automatically

The dropped facts are not deleted from the blob immediately — the blob is only written
on explicit fact updates, so stale data ages out naturally. A background cleanup pass
is possible but not required.

---

## Prompt Changes

### Evaluator and extractor prompts

Both prompts receive a `## Fact Schema` section generated at build time from
`fact_schema.json`. The section enumerates every valid leaf path grouped by mutability,
with descriptions and constraints. The evaluator is told explicitly: *"You may update
the value of any path listed below, subject to its mutability. You may NOT create new
paths."*

The rendered schema section is derived entirely from the JSON file — there is no
hand-written prompt text describing the fact taxonomy. If the schema changes, the prompt
updates automatically. A sketch of what the rendered output looks like (using the
illustrative schema from above):

```
## Fact Schema
You may UPDATE values for paths listed below.
You may NOT create new paths.

IMMUTABLE (cannot be changed once set):
  Character.Identity.Name — The character's given name
  Character.Identity.Age — The character's age in years
  Character.Appearance.Body.Height — The character's height
  Character.Appearance.Body.Hair-Colour — The character's hair colour
  User.Identity.Name — The user's name as they prefer to be addressed

MUTABLE (contextually appropriate; surfaced to user for approval):
  Character.Appearance.Outfit.Top — Upper-body clothing
  Character.Appearance.Outfit.Bottom — Lower-body clothing
  Character.Appearance.Outfit.Shoes — Footwear
  Setting.Location.Name — Current location name
  Setting.Location.Space — Interior | Exterior
  Setting.Temporal.Current-Year — Narrative year

FLUID (applied silently; no approval needed):
  Character.State-Of-Mind.Mood — Calm | Anxious | Angry | Sad | Joyful | Neutral | Guarded | Excited
  Character.State-Of-Mind.Energy — Exhausted | Tired | Neutral | Alert | Energised
  Setting.Location.Description — Immediate surroundings and atmosphere
```

### Evaluator output — tool calls only

The Character Evaluator no longer returns a structured JSON verdict. Instead, every
decision is expressed as a tool call. The server acts on whatever tools the evaluator
calls and infers the overall verdict from them.

**How tool calling works:** the LLM generates a tool call and stops. The server executes
the tool, appends the result as a `tool` role message, and re-invokes the LLM. The LLM
resumes. This repeats until the LLM generates a final response or calls a terminal tool.
All tool execution happens server-side; the LLM simply declares what it wants called.

**Character Evaluator tool list:**

| Tool | When called |
|---|---|
| `set_fact(path, value)` | Character implied a value for an existing schema path |
| `propose_inference(statement, derivation, source_paths)` | Character invented a detail with no schema path |
| `report_contradiction(description)` | Character response conflicts with a set `Immutable` fact |
| `report_pass()` | Response is clean — no violations, no updates |

The evaluator may call `set_fact` and/or `propose_inference` zero or more times, followed
by exactly one terminal tool (`report_contradiction` or `report_pass`). The server
re-derives mutability from the schema on each `set_fact` call — the evaluator's choice
of tool is not trusted for access control.

**Experiences are immutable.** Once written, an Experience is a permanent episodic
record — something the character *lived through*, not a snapshot of current world state.
Current state belongs in Facts (e.g. `Setting.Location.Name`); Experiences record what
happened (e.g. "Jon and the character spent an evening talking in a Chicago bar during
session 2", "Jon and the character met again in New York in session 5"). These are
parallel memories, not contradictions — both remain true. The evaluator does not delete
or overwrite Experiences; `report_experience_update` is removed. New episodic memories
are proposed and appended at session end via the normal session-end evaluator pass.

If multiple similar Experiences match the current retrieval query, the naive tie-breaking
rule is to surface the most recent one. More sophisticated tie-breaking (e.g. recency
weighted by similarity score) is deferred future work.

This is a deliberate departure from the current codebase, which deletes contradicted
Experiences immediately on an `experience_update` verdict. That behaviour is removed as
part of this work.

### Character system prompt

`build_system_prompt()` renders facts from the masked blob, walking the tree and
rendering each populated leaf under its grouping path as a heading. Each leaf renders
its current value, its `Description` (so the character understands the intended meaning
of the fact), and for `Enum` leaves the full `Constraint` list (so the character knows
the range of values the fact can take and can reason about them). Unpopulated leaves are
omitted from the character prompt — the character only sees what is known.

---

## UI Changes

### Sidechannel fact list

- Facts displayed as a tree mirroring the schema structure; each grouping node is a
  collapsible section; leaves show their current value (or a placeholder if unset)
- Fact creation is removed from the sidechannel for regular sessions — the fact list is
  read-only during roleplay. The user may still edit existing leaf values inline.
- A future settings panel (out of scope here) will expose the full schema with editable
  values and allow extending the taxonomy

### Removed UI elements

- "Promote to Fact" button on inferences in the sidechannel — removed
- Fact creation form — removed from the in-session UI (facts are defined by the schema;
  values are set either before the session in the settings panel or by the evaluator
  during roleplay subject to mutability rules)

### New notification cards

- **`fact_update` (fluid):** a quiet, non-blocking indicator in the sidechannel:
  *"Mood updated: anxious"*. Visible but requires no user action.
- **`fact_update` (mutable):** a blocking sidechannel card shown in place of the
  character's response: *"Character's outfit may have changed — top: blue scrubs →
  green scrubs. Accept / Edit / Reject."* Accept writes the value and reveals the
  response. Edit writes the user's amended value and regenerates. Reject discards the
  proposed change and regenerates. Both edit and reject skip the World Builder on
  regeneration.
- **`fact_update` (immutable, unset):** a blocking sidechannel card: *"Character implied
  their name is Sarah. Record character.identity.name = Sarah? Accept / Edit / Dismiss."*
  Accept locks the value and delivers the existing response unchanged. Edit locks the
  user's value and regenerates the response. Dismiss delivers the response with the
  invented value treated as ungrounded.

All three require the standard four-part commit rule (chat.js case, index.html card,
chat-component.js handler, chat-component.test.js test).

---

## Migration of Existing Facts

No migration script is needed. The schema masking at read time handles legacy data.
On first startup after deployment:

1. Create `character_facts` with an empty blob `{}` for each existing character
2. Legacy `facts` table rows are left in place but never read again
3. Molly's facts effectively reset to a clean slate; the user re-establishes them through
   normal roleplay (the evaluator will surface unset immutable paths as it encounters them)

The legacy `facts` table can be dropped in a future cleanup.

---

## Open Questions

1. ~~**World Builder authority vs. `set_fact` immutable rejection.**~~ **Resolved.**
   The World Builder uses `author_set_fact(path, value)`, which applies no mutability
   check. The Character Evaluator and Character LLM use `set_fact(path, value)`, which
   enforces mutability. The tools are scoped per invocation — each LLM pass is given only
   the tools listed in its system prompt, so the Character LLM cannot call
   `author_set_fact` because it never sees the tool described to it. The server validates
   caller identity as defence in depth, but tool-list scoping is the primary enforcement.

2. ~~**Mutable rejection feedback loop.**~~ **Resolved.** When the user rejects a
   `mutable` fact update, the rejected value is never written to the blob. The full fact
   schema with current values is injected into the system prompt on every turn, so the
   character LLM naturally picks up the correct (unchanged) value on the next turn without
   any explicit correction note. An explicit context note (*"your proposed change was
   rejected"*) may be added later if self-correction proves insufficient in practice, but
   is not required for initial implementation.

3. **Naming the immutable-value request tool.** `request_fact_value(path, reason)` is
   accurate but awkward. Candidates: `require_fact`, `prompt_user_for_fact`,
   `request_fact_from_user`. Should communicate "I cannot proceed without this and I am
   not allowed to invent it" — `require_fact` comes closest. Settle on a name before
   implementation.

4. ~~**Inference generation: evaluator-observed vs. character tool.**~~ **Resolved.**
   Option A (evaluator-observed) is the baseline and is always active — the Character
   Evaluator always observes the character's response and proposes logical/probabilistic
   inferences from it. Option B (character tool) is additive: the Character LLM is given
   `propose_inference` in its tool list and may call it when it reasons something follows
   from Facts. If the LLM uses the tool, great; if it does not, the evaluator covers it.
   The two approaches are not mutually exclusive. A future options menu may allow
   toggling each independently, since the user's choice of LLM will materially affect
   how reliably Option B fires.

5. ~~**Mutable updates: user approval flow.**~~ **Resolved.** The character's response is
   withheld until the user acts on the blocking card. Accept writes the value and delivers
   the existing response. Edit writes the user's value and regenerates (skipping the World
   Builder). Reject discards the proposed value and regenerates (skipping the World
   Builder). Regeneration on edit/reject is necessary because the withheld response was
   built on a world state the user has overruled.

6. **`fact_update` for unset fluid paths.** If `Character.State-Of-Mind.Mood` has no
   value yet and the evaluator first sets it (rather than changes it), should that be
   treated differently from subsequent updates? Probably not — the first write is still a
   fluid operation and should apply silently.

7. **Inference source paths.** Replacing `source_fact_ids` (integer array) with
   `source_fact_paths` (path string array) means existing inference rows become orphaned.
   Simplest fix: mark all existing inferences `invalidated` at migration time and let the
   next eager pass regenerate them against the new schema.

8. **Settings panel for schema editing.** The plan defers a UI for editing
   `fact_schema.json` to future work. The JSON file is the interim solution. This is
   acceptable for a local toy but needs to be designed before any multi-user or non-
   technical-user scenario.

9. **`suggest_fact` tool for live taxonomy growth.** Alongside `set_fact`, either pass
   could be given a `suggest_fact(path, value, description)` tool to call when it
   encounters a concept worth recording that has no schema path for it. Rather than
   silently discarding the observation or inventing a path, the tool surfaces a "new fact
   type" proposal to the user — the suggested path, an initial value, and the LLM's
   rationale — which the user can approve, edit, or dismiss. Approval would add the new
   path to the live schema and set the value in the same action. This requires the schema
   editing UI (itself deferred — see question 4) to exist first, and raises the question
   of whether live-inserted schema paths are written back to `fact_schema.json` or held
   only in memory for the session. Speculative; defer until the schema editing UI and the
   tool-use investigation are both underway.

10. **Derived / computed facts.** Some facts have deterministic mathematical relationships
   to others — the clearest example is `Character.Identity.Birth-Year = Setting.Temporal.Current-Year - Character.Identity.Age`. A `Derived` fact type would signal that the backend always recalculates this value from its dependencies rather than reading it from the blob, guaranteeing it never drifts. The real-world inventory of such facts is thin (birth year, user birth year, age gap between character and user, possibly an age-category label). Whether this justifies a formal expression mechanism or can be handled more simply (e.g. a hardcoded resolver for the small set of cases, with the formula expressed in prose in the `Description` field) is an open question. Deferred until the schema is fully built out and the full list of candidate derived facts is known.

---

## Implementation Sketch

Steps in dependency order; each independently reviewable.

**Step 0 — Tool-calling proof of concept (gate on all subsequent steps)**

Tool calling is the load-bearing mechanism for the World Builder (`author_set_fact`), the
Character Evaluator (`set_fact`), and the Character LLM (`require_fact`). The current
`ollama_client.py` has no tool-calling support. Before any schema, storage, or prompt
work begins, this PoC must validate:

- `OllamaClient` extended to accept a `tools` list and handle `tool_calls` responses,
  append tool results as `tool` role messages, and re-invoke until the model returns
  plain content
- `qwen3:7b` (and any other target model) reliably calls a simple tool (e.g., a single
  `set_fact(path, value)` stub) rather than hallucinating the call or embedding it in
  prose
- Multi-call behaviour: does the model call the tool multiple times per turn (one call
  per fact) or attempt to batch calls? How does it behave when a tool call is rejected
  with an error response?
- Latency: a tool-calling World Builder may involve 3–6 round-trips per turn; measure
  wall-clock cost on the target hardware

If the PoC reveals that `qwen3:7b` does not handle tool calling reliably, the
World Builder and Character Evaluator must fall back to structured JSON verdicts
(matching the current extractor/evaluator pattern), and this plan requires revision
before implementation proceeds.

**Step 0b — `require_fact` suspension PoC (gate on Character LLM tool work)**

Because tool calls are synchronous from the LLM's perspective — the model stops
generating and waits for the server to respond before continuing — `require_fact` does
not require task cancellation or coroutine suspension. When the Character LLM calls
`require_fact`, the model is already paused waiting for the tool result. The server
simply delays returning the tool result while it surfaces the blocking card via SSE and
waits for the user's HTTP response. Once the user provides the value, the server writes
it and returns it as the tool result. The LLM resumes with the value populated.

This is simpler than originally anticipated, but the PoC must still verify:

- The SSE stream can stay open for an indefinite user-interaction period without timing
  out or being dropped by the browser or any intermediate proxy
- The partial character response generated before the `require_fact` call is cleanly
  discarded — the LLM should regenerate from the start once the value is known, not
  append to whatever it had already said
- `run_turn` can coordinate the SSE stream, the tool-call loop, and the user-input
  endpoint without leaking DB state (the user message has already been stored)

If the SSE keep-alive proves unworkable, the fallback is to handle `require_fact` purely
via the Character Evaluator post-generation: the evaluator detects the unset immutable
path, surfaces the blocking card, and the Character LLM is re-invoked fresh once the
user provides the value. This avoids mid-generation suspension entirely.

**Step 1 — Schema JSON file and loader**
- Write `src/memories/fact_schema.json` with the default schema above
- Write `src/memories/schema_loader.py`: `load_schema()`, `render_schema_for_prompt()`,
  `apply_mask(blob)` (drops invalid paths per schema), `check_write_permitted(path, schema)`
  (returns mutability or raises if path not in schema)
- Tests: mask drops paths not in schema; known leaf paths pass through; unknown grouping
  paths dropped; immutable path blocked on re-write; mutability returned correctly for
  all path types

**Step 2 — DB: `character_facts` table and repository**
- Add `character_facts` table in `init_db()`
- Repository functions: `get_facts(character_id) → dict` (applies mask), `set_facts(character_id, blob)`,
  `patch_fact(character_id, path_tuple, value)`
- `get_facts()` returns `{}` if no row exists yet
- Tests: round-trip read/write; mask applied on read; patch updates a nested key; missing
  character returns empty dict

**Step 3 — Prompt changes**
- Update `build_system_prompt()` to render from blob by walking the schema tree; enum
  values as strings
- Update evaluator prompt to include schema section and `fact_update` verdict instructions;
  remove `implication` verdict from the prompt vocabulary
- Update World Builder prompt to match
- Update `EvaluatorResult` to replace `implication` violations with `fact_update` shape
- Tests: prompt contains schema section; facts render grouped; range rendering correct;
  evaluator output parsed with new shape

**Step 4 — Server-side mutability enforcement**
- On `fact_update` verdict: look up mutability from schema (not from evaluator output)
- `fluid`: apply immediately, emit quiet sidechannel notification
- `mutable`: queue as blocking notification; apply only if user accepts
- `immutable`, path already set: treat as contradiction, trigger regeneration
- `immutable`, path unset: emit unset-fact blocking notification
- Tests: fluid update applies and emits notification; mutable update blocks; immutable
  change triggers contradiction loop; immutable unset surfaces card

**Step 5 — UI: tree display, new notification cards, removed create/promote paths**
- Sidechannel fact list rendered as collapsible schema tree; inline value editing retained
- Remove "Promote to Fact" button from inference cards
- Remove in-session fact creation form
- Add `fact_update_fluid`, `fact_update_mutable`, `fact_update_immutable_unset` sidechannel
  cards; all three follow the four-part commit rule

**Step 6 — Inference path migration**
- Rename `source_fact_ids` → `source_fact_paths` in `inferences` table
- Mark all existing inferences `invalidated` at migration (regenerated by next eager pass)
- Update inference service to write/read path strings
- Remove `POST .../inferences/{id}/promote` endpoint and its tests
