# Phase 6 Implementation Plan — Fact Extraction from User Messages

## Goals (from plan.md)

- Add a pre-turn extractor LLM pass that analyses the user's message for explicit or implicit facts before the character LLM is invoked — so the world model grows passively from natural conversation rather than requiring the user to manually add Facts through the sidechannel
- Apply a four-tier extraction model: explicit new facts and explicit fact updates are auto-applied before the character call (so the character always responds with current user-stated values); implicit new facts and implicit fact updates are surfaced post-response as proposals for user confirmation
- The user is notified of every extraction: explicit changes via an informational `extraction_applied` sidechannel event with undo/delete options; implicit proposals via a confirmatory `implicit_fact_proposed` sidechannel event with Accept/Ignore actions
- A new `status(extracting)` SSE status event is emitted before `status(generating)` to communicate the new pre-turn step to the client
- After each turn, the fact list in the sidechannel refreshes to show any newly auto-extracted or auto-updated facts

**Deliverable:** The user can say "We're meeting in Chicago" mid-conversation and the character immediately knows it. If they say something that more weakly implies a fact, the system surfaces it as a proposal without acting on it unilaterally. When the user contradicts an established Fact outright, the character's response reflects the update, and the user can undo it if the extractor was wrong.

**Post-implementation note:** Once Phase 6 is shipped, the extraction tier model and its rationale should be backported to `docs/plan.md`, `README.md`, and `CLAUDE.md` to keep the architecture documentation current.

---

## Extraction Tier Model

The extractor assigns every candidate fact to one of four tiers based on two axes: **certainty** (explicit vs. implicit) and **novelty** (new vs. conflicting with an existing Fact). Each tier has a distinct treatment that balances the user's authorship of the world against the risk of polluting the world model with incorrect data.

### The two axes

**Certainty: explicit vs. implicit**

An *explicit* fact is one the user has directly and unambiguously stated. "We're meeting in Chicago" is an explicit setting fact. "My name is Sarah" is an explicit user fact. The user has committed to these values; acting on them immediately is the right call.

An *implicit* fact is one the system infers from what the user said, without the user having directly asserted it. "I just got home from work" implies the user's current location is home — but the user did not say so. "I've been feeling off all week" implies something about the user's emotional state — but is vague enough that a specific fact value would be a guess. Implicit extractions are inherently lower-confidence: the extractor's inference might be wrong, and the user may not have intended to assert anything persistent.

**Novelty: new vs. conflicting**

A *new* fact is one for which no existing Fact shares the same `(character_id, category, key)` triple. Adding it carries no risk of overwriting established information.

A *conflicting* fact is one where the user's stated value differs from an established Fact's stored value. Writing it means overwriting something the user (or the system, via implication acceptance) previously established. The old value is lost unless the write is undone.

### The four tiers

| Tier | Certainty | Novelty | Action | User sees |
|------|-----------|---------|--------|-----------|
| 1 | Explicit | New | Auto-create before character call | Informational note, delete button |
| 2 | Explicit | Conflicting | Auto-update before character call | Informational note with old value, undo button |
| 3 | Implicit | New | Surface post-response for confirmation | Proposal card: Accept / Ignore |
| 4 | Implicit | Conflicting | Surface post-response for confirmation | Proposal card with old value: Accept / Ignore |

### Why this tiering is the right model

**Tiers 1 and 2 are auto-applied because the user is the author of the world.** The plan's foundational premise is that the user establishes ground truth. When they state something explicitly, acting on it immediately — and letting the character respond with that information in context — produces a more coherent conversation than waiting for confirmation. For Tier 2, the auto-apply-with-undo pattern (rather than a blocking confirmation) avoids interrupting the conversation flow while still giving the user a recovery path if the extractor misread the message.

**Tiers 3 and 4 are not auto-applied because certainty matters more than speed.** If the extractor's inference about an implicit fact is wrong and that fact is auto-applied, the character will act on incorrect information going forward. For Tier 3 (no existing fact to overwrite) the cost of being wrong is low — a spurious fact can be deleted — but the UX signal sent by auto-applying uncertain data is bad: the user sees facts appearing in the sidechannel that they never said, which erodes trust in the system. For Tier 4, the stakes are higher: an implicit inference overwrites an established value, which is a loss of information the user previously committed to. Surfacing both tiers as proposals keeps the user in control without blocking the conversation.

**This mirrors the evaluator's existing certainty tiers for character-generated content.** The evaluator already distinguishes between logical inferences (high certainty, auto-promoted) and probabilistic inferences (lower certainty, surfaced for confirmation). Applying the same principle to extraction gives the system a coherent and learnable mental model: high-confidence signals are acted on silently; uncertain signals are surfaced for review. A user who understands the implication flow will intuitively understand the implicit extraction proposal flow.

**Implicit proposals are surfaced post-response, not pre-response.** Blocking the character call to wait for implicit proposal resolution would add friction and break the conversational rhythm. The character responds without the implicit facts in context; the user confirms the proposal; the next turn benefits. This is the same trade-off made for probabilistic inferences — the character misses one turn of context in exchange for keeping the world model trustworthy.

---

## Task List

### Backend Tasks

---

#### T1 — Extraction service (new file: `extraction_service.py`)

This module is the counterpart to `evaluator.py` but operates on the user's message rather than the character's response. It owns the extractor prompt, the LLM call, and the result parsing.

The extractor prompt should include: the current user message, the full list of existing Facts (with ID, category, mutability, and value), and a brief listing of active Inferences. Giving the extractor visibility into existing Facts serves two purposes — it enables conflict detection, and it prevents the LLM from proposing facts already established at the same value.

The extractor is instructed to produce a structured JSON result with three top-level lists:

`new_facts` — explicitly stated facts that do not overlap with any existing Fact. Each entry includes key, value, category (user/character/setting), mutability (immutable/low/high), and the source quote from the user's message.

`fact_updates` — explicitly stated values for keys that already exist as established Facts, where the stated value differs from the stored one. Each entry includes the existing Fact's ID, the existing key, the old value, the new value, and the source quote.

`implicit_proposals` — facts the user implied but did not directly state. Each entry includes key, proposed value, category, mutability, source quote, and an optional `existing_fact_id` field: absent if this would be a new fact, set to the existing Fact's ID if the implied value conflicts with an established one. The `existing_fact_id` field is what distinguishes an implicit new fact (Tier 3) from an implicit conflicting fact (Tier 4) in the backend handling.

The extractor must be conservative. It should only populate `new_facts` and `fact_updates` for things the user has stated clearly and directly. Narration, questions, vague comments, and things already captured at the same value should produce empty lists. `implicit_proposals` should be populated for strong inferences only — things any reasonable person would conclude from the user's phrasing, not speculative readings. When in doubt, the extractor should prefer an empty result over a speculative one.

The extraction should focus on user-category facts (about the person speaking) and setting-category facts (about the current environment). It should generally avoid proposing character-category facts, which are established through the user-managed sidechannel and the implication flow.

The module defines Pydantic models for the extractor's structured output: `ExtractedFact` (for `new_facts`), `FactUpdate` (for `fact_updates`, including `old_value`), `ImplicitProposal` (for `implicit_proposals`, with optional `existing_fact_id` and `old_value`), and `ExtractionResult` wrapping all three. An `ExtractionParseError` exception class signals JSON decode and validation failures. These are ephemeral LLM output types, not DB entities.

The `run_fact_extractor` function takes the user message, character, current facts list, active inferences list, and an `OllamaClient`. It issues a non-streaming Ollama chat call with `think=False` and `format="json"`, parses the response, and returns an `ExtractionResult`. Parse failures raise `ExtractionParseError`, caught by `run_turn`, which logs a warning and continues with an empty result.

---

#### T2 — Chat service: pre-turn extraction hook

After loading facts and inferences (and after Phase 5 experience retrieval), `run_turn` calls `run_fact_extractor`. Only Tier 1 and Tier 2 results are written to the DB at this point. Tier 3 and 4 proposals are carried forward in the return value but do not touch the DB before the character call.

For `new_facts` (Tier 1): each is written via `create_fact`. An IntegrityError on a given entry (duplicate key/category) is caught per-insert and skipped silently.

For `fact_updates` (Tier 2): each is written via `update_fact(db, fact_id=..., value=new_value)`, preserving existing category and mutability. No cascade is run at this point — cascade runs only when the user explicitly undoes an update via the undo endpoint (T4), keeping pre-character latency bounded.

After all Tier 1 and Tier 2 writes, the facts list is reloaded from the DB. The character LLM receives the current, user-authoritative world model.

Tier 3 and Tier 4 proposals (`implicit_proposals`) are not written to the DB. They are passed through in `run_turn`'s return value so `chat.py` can emit sidechannel events after the character responds.

`run_turn`'s return type expands to include the full `ExtractionResult`. On extraction failure, the result defaults to an empty `ExtractionResult`.

---

#### T3 — Chat router: SSE event sequence update

Two changes to the SSE event sequence in `chat.py`:

First, emit `status(extracting)` before awaiting `run_turn`. The full updated sequence is: `status(extracting)` → `status(generating)` → `status(reviewing)` → (contradiction loop if needed) → `thinking` → `message` → (sidechannel events) → `done`.

Second, after the `message` event, emit up to two sidechannel events from the extraction result:

`extraction_applied` — emitted when `new_facts` or `fact_updates` is non-empty. The payload carries two lists: `added` (Tier 1: key, value, category, fact ID) and `updated` (Tier 2: fact ID, key, old value, new value). This is an informational notification — the facts are already in the DB.

`implicit_fact_proposed` — emitted when `implicit_proposals` is non-empty. The payload carries two lists: `new_proposals` (Tier 3: key, proposed value, category, mutability, source quote) and `update_proposals` (Tier 4: existing fact ID, key, old value, proposed value, source quote). This is a confirmatory notification — the user must accept or ignore each proposal.

Both events are emitted after the `message` event, consistent with how implication and experience_update sidechannel events are positioned.

---

#### T4 — Extraction resolution endpoints (new endpoints in `implication.py`)

Three new turn-scoped endpoints, added to `implication.py` alongside the existing resolution endpoints.

`POST /api/sessions/{session_id}/turns/{turn_id}/undo-user-fact` — reverts a Tier 2 auto-update. Body: `{ fact_id, restore_value }`. The restore value is trusted from the client (it was supplied by the server in the `extraction_applied` payload). Calls `update_fact` with the restore value, then runs `cascade_on_fact_edit` to revalidate downstream Inferences. Response: `{ fact, stale_inferences }`. Validates session not ended (409), fact exists and belongs to the session's character (404).

`POST /api/sessions/{session_id}/turns/{turn_id}/accept-implicit-fact` — accepts a Tier 3 or Tier 4 proposal. Body: `{ key, value, category, mutability, existing_fact_id? }`. If `existing_fact_id` is absent (Tier 3), calls `create_fact`. If `existing_fact_id` is present (Tier 4), calls `update_fact` and then runs `cascade_on_fact_edit`. Response: `{ fact, stale_inferences }` (stale_inferences is empty for Tier 3). Validates session not ended (409), character exists (404), and for Tier 4 that the referenced fact belongs to the session's character (404).

`POST /api/sessions/{session_id}/turns/{turn_id}/ignore-implicit-fact` — no-op for Tier 3 or Tier 4 proposals. Body: `{ key }` (for logging). Returns 200. The client dismisses the proposal card.

For newly auto-added Tier 1 facts the user wants to remove, the existing `DELETE /api/characters/{character_id}/facts/{fact_id}` endpoint is sufficient — no new endpoint needed. The `extraction_applied` notification card supplies the fact ID for this call.

---

### Frontend Tasks

---

#### T5 — `chat.js`: new SSE state, sidechannel handlers, API helpers

The `sseStateToLabel` mapping gains a new entry for `'extracting'` — a label communicating that the system is reading the user's message for facts.

`buildNotificationFromSidechannel` gains two new cases:

For `type === 'extraction_applied'`: returns a notification with `scType: 'extraction_applied'`, carrying the `added` list (Tier 1 changes) and `updated` list (Tier 2 changes with old values and fact IDs).

For `type === 'implicit_fact_proposed'`: returns a notification with `scType: 'implicit_fact_proposed'`, carrying the `new_proposals` list (Tier 3) and `update_proposals` list (Tier 4, with existing fact IDs and old values).

New API helper functions:

`apiUndoUserFact(sessionId, turnId, factId, restoreValue)` — posts to `undo-user-fact` with `{ fact_id, restore_value }`.

`apiAcceptImplicitFact(sessionId, turnId, key, value, category, mutability, existingFactId)` — posts to `accept-implicit-fact` with the proposal data. `existingFactId` is optional (null for Tier 3, set for Tier 4).

`apiIgnoreImplicitFact(sessionId, turnId, key)` — posts to `ignore-implicit-fact` with `{ key }`.

`apiDeleteFact(characterId, factId)` — delete helper for Tier 1 auto-added facts the user wants to remove.

---

#### T6 — `index.html`: notification cards and fact list refresh

**`extraction_applied` notification card** (`v-else-if="msg.scType === 'extraction_applied'"`):

Renders two sections when non-empty. For `added` entries (Tier 1): "Added: [key] = [value]" with a delete link calling `apiDeleteFact`. For `updated` entries (Tier 2): "Updated: [key] changed from [old value] → [new value]" with an Undo button calling `apiUndoUserFact`. On successful delete or undo, the row collapses to a brief confirmation and the fact list is refreshed. If an undo response includes stale inferences, show a brief "N inferences may need review" note linking to the inferences pane.

**`implicit_fact_proposed` notification card** (`v-else-if="msg.scType === 'implicit_fact_proposed'"`):

Renders two sections. For `new_proposals` (Tier 3): "Implied: [key] = [value] ([source quote])" with Accept and Ignore buttons. For `update_proposals` (Tier 4): "Implied update: [key] from [old value] → [proposed value] ([source quote])" with Accept and Ignore buttons. Accept calls `apiAcceptImplicitFact`; Ignore calls `apiIgnoreImplicitFact`. On acceptance, the row collapses to a confirmation and the fact list is refreshed. If a Tier 4 acceptance response includes stale inferences, surface the same brief note used in the `extraction_applied` card.

The fact list (Column 2) refreshes after every `done` SSE event, picking up any Tier 1/2 auto-applied facts.

---

#### T7 — `chat-component.js`: handlers for both new sidechannel types

The `sendMessage` SSE loop gains handler branches for both `extraction_applied` and `implicit_fact_proposed` sidechannel events. Each handler calls `buildNotificationFromSidechannel` and pushes the notification onto the messages array.

The `done` event handler triggers a fact list refresh.

---

## Architecture Decisions

### D1 — Extractor runs sequentially before the character LLM

The character LLM must receive Tier 1 and Tier 2 updates in its system prompt. These writes must complete before the character call begins. Parallelisation is not possible without splitting context-building. Keeping the extractor prompt lean (user message + facts only, no conversation history) mitigates the latency impact.

### D2 — Tiers 1 and 2 are applied before the character call; Tiers 3 and 4 are not

Only explicit extractions are applied pre-character. The character always responds with correct, user-authoritative values for things the user has explicitly stated. Implicit proposals are withheld: the extractor's inference might be wrong, and building the character's response on an incorrect implicit assumption produces worse output than building it on the existing established facts. The character misses the implicit context for one turn; the world model stays clean. This matches the treatment of probabilistic inferences from the evaluator, which are also surfaced post-response rather than auto-promoted.

### D3 — Explicit conflicts (Tier 2) are auto-applied; the user is informed and can undo

The plan's foundational premise is that the user is the author of the world. An explicit conflict is not ambiguous — the user said something directly, and it differs from the stored value. Blocking the turn to ask for confirmation would add friction for a case where the user is almost certainly right. Auto-applying with an informational notification and undo provides recovery without interruption. The undo button carries the old value (supplied in the sidechannel payload) so no server-side history is needed.

### D4 — Implicit conflicts (Tier 4) are surfaced rather than auto-applied

An implicit conflict combines two sources of uncertainty: the extraction inference might be wrong, and even if it is right, overwriting an established Fact with an inferred value destroys information the user previously committed to. The cost of a Tier 4 false positive is higher than any other tier — a known-good value is replaced with a guess. Surfacing it as a proposal is the conservative and correct default. If the inference is right, the user accepts it with one click; if not, they ignore it.

### D5 — No response regeneration for any extraction path

Tier 1 and 2 facts are applied before the character call, so the character's response is already correct — no regeneration needed. Tier 3 and 4 proposals are not applied before the character call, so the character's response does not incorporate them — again, no regeneration needed. This is a direct consequence of the tier model cleanly separating pre-call and post-call actions.

### D6 — Cascade deferred to undo and accept-implicit-update, not run at extraction time

`cascade_on_fact_edit` involves an LLM call per downstream Inference. Running it during extraction for every Tier 2 update would add unbounded pre-character latency for characters with deep inference chains. Cascade runs only when the user explicitly undoes a Tier 2 update or accepts a Tier 4 proposal — cases where the user has consciously engaged with the change and can tolerate the extra latency.

### D7 — Undo and accept-implicit-update carry the restore/old value from the client

Rather than maintaining a server-side history of previous fact values, the undo endpoint accepts `restore_value` from the client, which received it in the sidechannel payload. The `accept-implicit-fact` endpoint similarly receives the full proposal details from the client. Both values originated from the server and are trusted. The worst-case consequence of a malicious or malformed client call is an incorrect fact update, no different from any other fact edit via the PUT endpoint.

### D8 — Extraction scope: user and setting facts only

The extractor focuses on user-category and setting-category facts. Character facts are established through the user-managed sidechannel and the evaluator's implication flow. Allowing the extractor to also propose character-category facts would create a confusing dual-path and risk hallucinating character attributes from ambiguous phrasing.

### D9 — Extraction failures are non-fatal

If the extractor returns invalid JSON or the Ollama connection drops, `run_turn` continues with an empty `ExtractionResult`. A warning is logged. The character call proceeds with the existing facts. Extraction enhances the conversation; it is not load-bearing.

### D10 — Fact list refresh on every `done` event

Rather than selectively refreshing only when the server signals that facts were changed, the frontend refreshes the fact list after every `done` event. This keeps the list authoritative without client-side merging logic and handles edge cases like server restarts mid-session.

---

## Potential Gotchas

**G1 — Extractor may misclassify explicit vs. implicit**

The explicit/implicit boundary is a judgment call by the extractor LLM. A fact the user clearly stated might be classified as implicit; an inference the user did not intend might be classified as explicit. The prompt should provide concrete examples of each to calibrate the model. If misclassification is common in practice, the fix is prompt engineering. The consequences are bounded: a misclassified explicit-as-implicit goes to a proposal card rather than auto-applying (minor friction), and a misclassified implicit-as-explicit gets auto-applied with an undo button (user recovers with one click).

**G2 — Extractor may be too aggressive for roleplay narration**

If the user narrates an action ("I walk into the tavern and order an ale"), the extractor might propose a setting fact (location = tavern). The prompt must clearly distinguish user speech-as-themselves from roleplay narration. This is a prompt engineering concern; the tier model does not change the risk, but Tier 3/4 proposals (likely where narration-derived facts land) require explicit user acceptance before entering the world model.

**G3 — Extraction adds latency before every character response**

The pre-turn extractor call adds 1–3 seconds on a local machine. The `status(extracting)` indicator communicates this. If latency becomes unacceptable, the extractor prompt should be trimmed. Skipping extraction for very short messages is a possible future optimisation.

**G4 — IntegrityError on Tier 1 inserts must be caught per-insert**

A blanket try/except around the Tier 1 batch would suppress real errors. The error must be caught precisely per-insert so other new facts in the same batch are still written.

**G5 — Tier 2 cascade deferral leaves downstream Inferences temporarily stale**

After a Tier 2 auto-update, downstream Inferences that referenced the old value remain active until the user undoes the update or the next eager pass runs. For Phase 6 this is acceptable. If it proves to be a correctness problem in practice, cascade can be added to the Tier 2 write path at the cost of pre-character latency.

**G6 — Source quotes must be trimmed server-side**

The `source_quote` field in both sidechannel payloads quotes the relevant portion of the user's message. Trim to a maximum length (e.g., 200 characters) before JSON serialisation to avoid oversized payloads.

**G7 — Tier 4 acceptance may produce stale inferences**

Accepting a Tier 4 proposal (implicit conflict) runs `cascade_on_fact_edit`, which may mark downstream Inferences stale. The accept-implicit-fact response includes a `stale_inferences` list; the notification card must surface a brief note when it is non-empty. This mirrors the undo endpoint's handling of the same situation.

**G8 — Both sidechannel events may be emitted in the same turn**

A single user message might contain one explicit fact update (Tier 2) and one implicit new fact proposal (Tier 3), producing both an `extraction_applied` and an `implicit_fact_proposed` sidechannel event in the same SSE stream. The client must handle both in the same turn gracefully — two separate notification cards will appear below the character's response. The `sendMessage` SSE loop already handles multiple sidechannel events per turn (implication + experience_update can co-occur); this extends that pattern.

---

## Test Plan

Tests are written first. Phase 6 is complete when all Phase 6 tests pass alongside all existing Phase 1–5 tests, and overall coverage stays at or above 80%.

Extractor Ollama calls are mocked with `respx` via a `make_extractor_ndjson()` helper in the unit `conftest.py`, mirroring `make_evaluator_ndjson()`. The default mock returns a valid JSON extraction result with all three lists populated as needed per test.

---

### Backend unit tests — new file `tests/unit/test_extraction_service.py`

| # | Test name | Asserts |
|---|-----------|---------|
| 1 | `test_extractor_prompt_includes_user_message` | User message appears verbatim in the prompt |
| 2 | `test_extractor_prompt_includes_all_existing_facts` | Every fact appears with its key and value |
| 3 | `test_extractor_prompt_includes_fact_ids` | Each fact line includes its DB ID (needed for `fact_updates` references) |
| 4 | `test_extractor_prompt_includes_fact_categories` | Each fact line includes its category label |
| 5 | `test_extractor_prompt_includes_fact_mutability` | Each fact line includes its mutability label |
| 6 | `test_extractor_prompt_includes_inferences` | Active inferences are listed in the prompt |
| 7 | `test_extractor_prompt_omits_inferences_section_when_none` | With no inferences, the inferences section is absent |
| 8 | `test_extractor_prompt_no_facts_shows_placeholder` | Empty fact list renders a placeholder, not an empty section |
| 9 | `test_parse_extraction_result_new_facts_only` | JSON with only `new_facts` → correct `ExtractionResult` |
| 10 | `test_parse_extraction_result_fact_updates_only` | JSON with only `fact_updates` (with old_value) → correct `ExtractionResult` |
| 11 | `test_parse_extraction_result_implicit_proposals_only` | JSON with only `implicit_proposals` → correct `ExtractionResult` |
| 12 | `test_parse_extraction_result_implicit_proposal_new_has_no_existing_fact_id` | Tier 3 proposal entry has no `existing_fact_id` |
| 13 | `test_parse_extraction_result_implicit_proposal_update_has_existing_fact_id` | Tier 4 proposal entry has `existing_fact_id` and `old_value` |
| 14 | `test_parse_extraction_result_all_three_lists_populated` | JSON with all three lists non-empty → all parsed correctly |
| 15 | `test_parse_extraction_result_all_empty` | All three lists empty → `ExtractionResult` with empty lists |
| 16 | `test_parse_extraction_result_invalid_json_raises_error` | Non-JSON content → `ExtractionParseError` |
| 17 | `test_parse_extraction_result_missing_required_fields_raises_error` | JSON missing a required field on a `new_fact` entry → `ExtractionParseError` |
| 18 | `test_run_fact_extractor_returns_extraction_result` | Mocked Ollama returns valid JSON → `ExtractionResult` returned |
| 19 | `test_run_fact_extractor_on_parse_error_raises_extraction_parse_error` | Invalid JSON from Ollama → `ExtractionParseError` |

### Backend unit tests — `tests/unit/test_chat_service.py` additions

| # | Test name | Asserts |
|---|-----------|---------|
| 20 | `test_run_turn_calls_extractor_before_character_llm` | Extractor Ollama call precedes character Ollama call (respx call order) |
| 21 | `test_run_turn_auto_adds_tier1_facts` | `new_facts` in result → `create_fact` called for each before character call |
| 22 | `test_run_turn_auto_updates_tier2_facts` | `fact_updates` in result → `update_fact` called for each before character call |
| 23 | `test_run_turn_does_not_write_implicit_proposals_to_db` | `implicit_proposals` in result → no `create_fact` or `update_fact` called for them |
| 24 | `test_run_turn_passes_tier1_and_tier2_facts_to_character` | Character system prompt includes extracted/updated fact values |
| 25 | `test_run_turn_character_prompt_does_not_include_implicit_proposals` | System prompt does not contain proposed-but-unconfirmed values from implicit_proposals |
| 26 | `test_run_turn_returns_extraction_result` | `run_turn` return value includes the full `ExtractionResult` |
| 27 | `test_run_turn_on_extraction_failure_continues_with_empty_result` | `ExtractionParseError` → turn completes; extraction result is empty; warning logged |
| 28 | `test_run_turn_on_ollama_connection_error_during_extraction_continues` | Ollama connection failure → turn continues; warning logged |
| 29 | `test_run_turn_deduplicates_tier1_facts_that_already_exist` | Extractor proposes a new fact with existing key/category → IntegrityError caught silently |
| 30 | `test_run_turn_empty_extraction_result_does_not_change_facts` | All three lists empty → no fact writes; turn proceeds normally |

### Backend integration tests — `tests/integration/test_api_chat.py` additions

| # | Test name | Asserts |
|---|-----------|---------|
| 31 | `test_turn_tier1_fact_added_to_db` | Extractor returns a new_fact; after turn, GET /facts includes it |
| 32 | `test_turn_tier1_fact_in_character_prompt` | Character system prompt includes the Tier 1 fact |
| 33 | `test_turn_tier2_update_overwrites_fact_in_db` | Extractor returns a fact_update; after turn, GET /facts shows updated value |
| 34 | `test_turn_tier2_character_prompt_uses_new_value` | Character system prompt includes the Tier 2 updated value, not the old one |
| 35 | `test_turn_tier3_proposal_not_written_to_db` | Extractor returns implicit new proposal; after turn, GET /facts does not include it |
| 36 | `test_turn_tier4_proposal_does_not_overwrite_existing_fact` | Extractor returns implicit update proposal; after turn, GET /facts still shows old value |
| 37 | `test_turn_emits_extraction_applied_when_tier1_or_tier2_present` | SSE stream includes `sidechannel` event with `type: extraction_applied` |
| 38 | `test_turn_extraction_applied_added_list_has_key_and_fact_id` | `extraction_applied` payload `added` list contains key and fact ID |
| 39 | `test_turn_extraction_applied_updated_list_has_old_and_new_values` | `extraction_applied` payload `updated` list has old_value and new_value |
| 40 | `test_turn_emits_implicit_fact_proposed_when_implicit_proposals_present` | SSE stream includes `sidechannel` event with `type: implicit_fact_proposed` |
| 41 | `test_turn_implicit_fact_proposed_new_proposals_list_populated` | `implicit_fact_proposed` payload `new_proposals` list is populated for Tier 3 |
| 42 | `test_turn_implicit_fact_proposed_update_proposals_has_old_value` | `implicit_fact_proposed` payload `update_proposals` has old_value for Tier 4 |
| 43 | `test_turn_both_sidechannel_events_emitted_in_same_turn` | A turn producing both Tier 2 and Tier 3 results emits both sidechannel events |
| 44 | `test_turn_no_sidechannel_when_extraction_empty` | Extractor returns empty result; no extraction sidechannel events |
| 45 | `test_turn_emits_status_extracting_before_status_generating` | SSE stream contains `status(extracting)` and it precedes `status(generating)` |
| 46 | `test_turn_on_extractor_failure_still_delivers_response` | Extractor mock raises error; SSE stream still delivers character `message` event |

### Backend integration tests — new file `tests/integration/test_api_extraction_resolution.py`

Fixtures: a character with at least one established Fact and an active session.

| # | Test name | Asserts |
|---|-----------|---------|
| 47 | `test_undo_user_fact_returns_200` | `POST .../undo-user-fact` with valid fact_id and restore_value → 200 |
| 48 | `test_undo_user_fact_restores_value_in_db` | After undo, GET /facts shows restore_value |
| 49 | `test_undo_user_fact_preserves_category_and_mutability` | Category and mutability unchanged after undo |
| 50 | `test_undo_user_fact_response_includes_fact_and_stale_inferences` | Response has `fact` key and `stale_inferences` list |
| 51 | `test_undo_user_fact_triggers_cascade` | Fact with downstream inferences → `stale_inferences` non-empty in response |
| 52 | `test_undo_user_fact_unknown_session_returns_404` | Non-existent session_id → 404 |
| 53 | `test_undo_user_fact_unknown_fact_returns_404` | Non-existent fact_id → 404 |
| 54 | `test_undo_user_fact_wrong_character_returns_404` | Fact belongs to different character → 404 |
| 55 | `test_undo_user_fact_ended_session_returns_409` | Ended session → 409 |
| 56 | `test_accept_implicit_fact_tier3_creates_new_fact` | `POST .../accept-implicit-fact` without existing_fact_id → 201; GET /facts includes new fact |
| 57 | `test_accept_implicit_fact_tier3_response_includes_fact` | Response body has `fact` key with created Fact |
| 58 | `test_accept_implicit_fact_tier3_stale_inferences_is_empty` | Tier 3 acceptance (new fact) → `stale_inferences` is empty |
| 59 | `test_accept_implicit_fact_tier4_updates_existing_fact` | `POST .../accept-implicit-fact` with existing_fact_id → GET /facts shows new value |
| 60 | `test_accept_implicit_fact_tier4_triggers_cascade` | Tier 4 acceptance on fact with downstream inferences → `stale_inferences` non-empty |
| 61 | `test_accept_implicit_fact_tier4_wrong_character_returns_404` | existing_fact_id belongs to different character → 404 |
| 62 | `test_accept_implicit_fact_ended_session_returns_409` | Ended session → 409 |
| 63 | `test_ignore_implicit_fact_returns_200` | `POST .../ignore-implicit-fact` → 200 |
| 64 | `test_ignore_implicit_fact_does_not_modify_db` | After ignore, GET /facts unchanged |
| 65 | `test_ignore_implicit_fact_ended_session_returns_409` | Ended session → 409 |

---

### Frontend tests — `tests/frontend/chat.test.js` additions

| # | Test name | Asserts |
|---|-----------|---------|
| 66 | `sseStateToLabel_extracting_returns_non_empty_string` | `sseStateToLabel('extracting')` returns a non-empty string |
| 67 | `sseStateToLabel_extracting_differs_from_generating` | `sseStateToLabel('extracting')` differs from `sseStateToLabel('generating')` |
| 68 | `buildNotificationFromSidechannel_handles_extraction_applied` | Payload `type: 'extraction_applied'` → notification object returned |
| 69 | `buildNotificationFromSidechannel_extraction_applied_scType` | Returned notification has `scType === 'extraction_applied'` |
| 70 | `buildNotificationFromSidechannel_extraction_applied_includes_added` | Returned notification includes `added` array |
| 71 | `buildNotificationFromSidechannel_extraction_applied_includes_updated` | Returned notification includes `updated` array |
| 72 | `buildNotificationFromSidechannel_extraction_applied_updated_entry_has_old_and_new_value` | `updated` entry has `old_value` and `new_value` |
| 73 | `buildNotificationFromSidechannel_extraction_applied_updated_entry_has_fact_id` | `updated` entry has `fact_id` for undo call |
| 74 | `buildNotificationFromSidechannel_handles_implicit_fact_proposed` | Payload `type: 'implicit_fact_proposed'` → notification object returned |
| 75 | `buildNotificationFromSidechannel_implicit_fact_proposed_scType` | Returned notification has `scType === 'implicit_fact_proposed'` |
| 76 | `buildNotificationFromSidechannel_implicit_fact_proposed_includes_new_proposals` | Returned notification includes `new_proposals` array (Tier 3) |
| 77 | `buildNotificationFromSidechannel_implicit_fact_proposed_includes_update_proposals` | Returned notification includes `update_proposals` array (Tier 4) |
| 78 | `buildNotificationFromSidechannel_implicit_update_proposal_has_old_value` | `update_proposals` entry has `old_value` |
| 79 | `buildNotificationFromSidechannel_implicit_update_proposal_has_existing_fact_id` | `update_proposals` entry has `existing_fact_id` |
| 80 | `apiUndoUserFact_posts_to_correct_url` | `fetch` called with correct `POST` URL including session_id and turn_id |
| 81 | `apiUndoUserFact_sends_fact_id_and_restore_value` | Request body contains `fact_id` and `restore_value` |
| 82 | `apiAcceptImplicitFact_posts_to_correct_url` | `fetch` called with correct `POST` URL |
| 83 | `apiAcceptImplicitFact_sends_proposal_data_in_body` | Request body contains key, value, category, mutability |
| 84 | `apiAcceptImplicitFact_sends_existing_fact_id_when_tier4` | Tier 4 call includes `existing_fact_id` in body |
| 85 | `apiAcceptImplicitFact_omits_existing_fact_id_when_tier3` | Tier 3 call does not include `existing_fact_id` in body |
| 86 | `apiIgnoreImplicitFact_posts_to_correct_url` | `fetch` called with correct `POST` URL |
| 87 | `apiIgnoreImplicitFact_sends_key_in_body` | Request body contains `key` |
| 88 | `apiDeleteFact_sends_delete_to_correct_url` | `fetch` called with `DELETE /api/characters/{id}/facts/{fact_id}` |

### Frontend tests — `tests/frontend/chat-component.test.js` additions

| # | Test name | Asserts |
|---|-----------|---------|
| 89 | `sse_status_extracting_sets_loading_state` | `status(extracting)` event → component in loading/busy state |
| 90 | `sse_status_extracting_does_not_show_thinking_indicator` | After `extracting` before `generating`, no thinking indicator |
| 91 | `sse_extraction_applied_sidechannel_adds_notification` | `sidechannel(extraction_applied)` → notification pushed to messages array |
| 92 | `sse_extraction_applied_notification_has_correct_scType` | Pushed notification has `scType === 'extraction_applied'` |
| 93 | `sse_implicit_fact_proposed_sidechannel_adds_notification` | `sidechannel(implicit_fact_proposed)` → notification pushed to messages array |
| 94 | `sse_implicit_fact_proposed_notification_has_correct_scType` | Pushed notification has `scType === 'implicit_fact_proposed'` |
| 95 | `sse_both_sidechannel_types_in_same_turn_produce_two_notifications` | Both `extraction_applied` and `implicit_fact_proposed` events → two separate notifications in messages |
| 96 | `sse_done_event_triggers_fact_list_refresh` | `done` event → fact-fetching function called |
| 97 | `sse_no_extraction_notifications_when_extraction_empty` | Turn with no extraction results → no extraction notifications in messages |

---

## Not in Scope for Phase 6

- **Cascade at Tier 2 extraction time** — `cascade_on_fact_edit` is not run when Tier 2 updates are applied. Downstream Inferences may be temporarily stale until the user undoes the update or the next eager pass runs.
- **Character-category fact extraction** — the extractor focuses on user and setting categories only.
- **Confidence-gated auto-apply** — Phase 6 uses the explicit/implicit distinction as the sole gate. A numeric confidence threshold gating auto-application is a future refinement.
- **Extraction-aware `captured_by` annotation** — Phase 7a work.
- **Context budget tracking for the extractor call** — Phase 7a work.
