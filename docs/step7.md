# Step 7 — Mutable and Immutable-Unset Approval Gates + Dead-Code Removal

## Overview

Step 7 completes the `set_fact` tool handler for the two remaining mutability tiers —
`Mutable` and `Immutable` (unset) — and purges the router-layer dead code that was left
in place by Steps 3–5. The `Fluid` path has been fully wired since Step 5. The other two
paths currently return stub errors ("not implemented yet") that instruct the evaluator to
leave those paths alone. This step replaces both stubs with the real suspension-and-
approval flow described in `docs/plan-v2.md`.

The mechanism itself — `asyncio.Queue` gate, SSE blocking card, and `/respond` endpoint
— is not new. `require_fact` (Step 6) already proved the full suspend-resume cycle using
`tool_gate.py`. Step 7 applies the same pattern inside `_handle_set_fact()` in
`evaluator.py`, wiring in `await_gate()` the same way `_handle_require_fact()` in
`chat_service.py` already does. The gate created by `run_turn()` before calling
`run_contradiction_loop()` is in scope for the entire duration of the evaluator call —
no extra infrastructure is needed.

The main new server-side behavior is the regeneration signal. When the user rejects a
Mutable fact update or edits a Mutable or Immutable-unset value, the character's response
must be regenerated with the updated world state. This is expressed as a new
`needs_regeneration: bool` field on `EvaluatorResult`. When `run_contradiction_loop()`
sees `needs_regeneration=True`, it continues its loop for another Character-LLM + Evaluator
pass, exactly as it already does for `verdict="contradiction"`. The World Builder is
naturally skipped on these extra passes because it only runs in `run_turn()`, not inside
the loop.

Dead code cleanup removes two `if eval_result.verdict in ("implication",
"new_inference_probabilistic"):` blocks from `routers/chat.py` and one from
`routers/implication.py`. Both verdicts were removed from `_VALID_VERDICTS` in Step 3,
making these branches unreachable. The plan also referenced removing `experience_update`
and `implication` branches from `chat_service.py`, but both were already eliminated
during Steps 3–5; the current `chat_service.py` is clean.

**Success criterion:** all tests in `tests/unit/test_evaluator_service.py`,
`tests/unit/test_chat_service.py`, `tests/integration/test_api_chat.py`,
`tests/integration/test_api_implication.py`, and the new
`tests/integration/test_set_fact_approval_live.py` pass, along with every other existing
test.

---

## What Steps 0b–6 Delivered

- **`src/memories/services/tool_gate.py`** — `create_gate(session_id, turn_id)`,
  `await_gate(session_id, turn_id) -> str | None` (raises `KeyError` if gate absent),
  `resolve_gate(session_id, turn_id, value: str | None)` (raises `asyncio.QueueFull` on
  double-resolve), `cleanup_gate(session_id, turn_id)`. Per-turn
  `asyncio.Queue(maxsize=1)` keyed by `(session_id, turn_id)`.
- **`src/memories/routers/require_fact.py`** — `POST /{session_id}/turns/{turn_id}/
  require-fact/respond` body `{"value": str | None}` → `resolve_gate(...)`. 404 on
  `KeyError`, 409 on `asyncio.QueueFull`. Mounted at `/api/sessions` in `main.py`.
- **`src/memories/services/chat_service.py`** — `run_turn()` calls `create_gate(session_id,
  turn_id)` before the `try` block and `cleanup_gate()` in `finally`. The gate is live
  for the entire duration of `run_contradiction_loop()` and therefore for any
  `run_evaluator()` called from within it.
- **`src/memories/services/evaluator.py`** — `_handle_set_fact()` fully implemented for
  Fluid paths (coerce, validate, write, emit `fact_update_fluid` sidechannel, log
  decision). For Mutable and Immutable-unset paths, it currently returns stub errors.
  `EvaluatorResult` fields: `verdict`, `new_inferences`, `violations`, `decision_log`,
  `contradiction_notifications`, `max_retries_exceeded`. No `needs_regeneration` field yet.
- **`src/memories/services/tool_gate.py`** — queue typed as
  `asyncio.Queue[str | None]`. The `require_fact` router puts raw strings or `None`. The
  `set_fact` approval router (Step 7) will put JSON-encoded decision dicts
  (`json.dumps({"action": ..., "value": ...})`), which `_handle_set_fact` decodes.
- **`tests/unit/test_evaluator_service.py`** — `test_set_fact_immutable_unset_returns_
  stub_error` and `test_set_fact_mutable_path_returns_stub_error_regardless_of_current_
  value` verify the current stubs; both are deleted in Step 7.

---

## What This Step Does NOT Change

- **`tool_gate.py`** — queue type stays `asyncio.Queue[str | None]`; no new functions.
  The new `fact_approval.py` router JSON-encodes its decision dict into a string before
  calling `resolve_gate()`. (Deviation, resolved in CT-4: a module-level `_resolved`
  ordered set was added so a double-resolve returns 409 even after `cleanup_gate` runs;
  it is bounded by `_MAX_RESOLVED_KEYS` to prevent unbounded growth.)
- **`require_fact.py`** — completely untouched.
- **`chat_service.py`** — only `run_contradiction_loop()` changes (Part E below). The
  imports, `run_turn()` body, `_handle_require_fact`, and `_REQUIRE_FACT_TOOL` are
  untouched.
- **The World Builder** (`world_builder.py`) — not called during regeneration passes.
  Regeneration runs inside `run_contradiction_loop()`, which never invokes the World
  Builder; that exclusion is structural, not a new guard.
- **`evaluator.py` prompt or tool list** — the evaluator's system prompt and `_EVALUATOR_TOOLS`
  list are unchanged. The `set_fact` tool description already mentions that an
  "Immutable path already set returns an error" — that branch is already live. Only the
  two stub branches (unset-Immutable and Mutable) are replaced.
- **Frontend** (`index.html`, `chat.js`, `chat-component.js`) — Step 8's job. The two
  new sidechannel types (`fact_update_mutable`, `fact_update_immutable_unset`) will be
  emitted by the server but produce no visible card until Step 8 wires them in.
- **`routers/implication.py` endpoints** — the four endpoints (`accept-implication`,
  `ignore-implication`, `accept-inference`, `ignore-inference`) and the Phase-6 endpoints
  remain in place; only one dead `verdict` check inside `accept_implication()` is
  removed (Part I).
- **`CLAUDE.md` / README** — Step 10.

---

## Detailed Design

### Part A — `src/memories/services/evaluator.py` — `EvaluatorResult.needs_regeneration`

Add one field to the Pydantic model, after `max_retries_exceeded`:

```python
# Before
class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    decision_log: str
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False

# After
class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    decision_log: str
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False
    needs_regeneration: bool = False
```

`needs_regeneration=True` signals to `run_contradiction_loop()` that the user chose
edit (Mutable or Immutable-unset) or reject (Mutable) and the Character LLM must run
again with the updated `facts_blob`. It is `False` for accept, dismiss, all Fluid paths,
and all contradiction paths.

---

### Part B — `src/memories/services/evaluator.py` — new imports and closure flag

**Imports to add** (after the existing `from memories.database import ...` lines):

```python
import json
from memories.services.tool_gate import await_gate
```

**Add `_regeneration_needed` closure list** at the very top of `run_evaluator()`, immediately
after `working_blob = copy.deepcopy(facts_blob)`:

```python
    working_blob: dict[str, Any] = copy.deepcopy(facts_blob)
    _regeneration_needed: list[bool] = [False]   # ← add this line
```

This list acts as a mutable closure container so that `_handle_set_fact()` can set the
flag even though `_handle_set_fact` is a nested closure (not a class method with a `self`
reference). Using a one-element list is the project's established pattern for mutable
closure state (see `nonlocal facts_blob` in `chat_service.py` — both patterns are in use).

**Update all three `return` sites** in `run_evaluator()` to pass
`needs_regeneration=_regeneration_needed[0]`:

```python
# Site 1 — cap reached with no nudge success
return (
    EvaluatorResult(
        verdict="pass",
        decision_log="(tool-call cap reached — response delivered unverified)",
        needs_regeneration=_regeneration_needed[0],
    ),
    working_blob,
)

# Site 2 — report_contradiction terminal call
return (
    EvaluatorResult(
        verdict="contradiction",
        violations=[Violation(type="contradiction", description=description)],
        decision_log=description,
        needs_regeneration=_regeneration_needed[0],
    ),
    working_blob,
)

# Site 3 — report_pass terminal call
return (
    EvaluatorResult(
        verdict="pass",
        decision_log="Response is consistent with established facts.",
        needs_regeneration=_regeneration_needed[0],
    ),
    working_blob,
)
```

---

### Part C — `src/memories/services/evaluator.py` — `_handle_set_fact` Immutable-unset branch

**Current code** (lines 346–358 of the current `evaluator.py`):

```python
        if mutability == "Immutable":
            current = _lookup_leaf_value(working_blob, path)
            if current is not None:
                return (
                    f"Error: {path} is Immutable and already set to {current!r}. "
                    "You may not change it. If the character's response conflicts with "
                    "this, call report_contradiction instead."
                )
            return (
                f"Error: {path} is Immutable and unset. The approval flow for unset "
                "Immutable facts is not implemented yet — leave this path alone."
            )
```

**Replacement** (keep the "already set" branch exactly; replace only the stub `return`):

```python
        if mutability == "Immutable":
            current = _lookup_leaf_value(working_blob, path)
            if current is not None:
                return (
                    f"Error: {path} is Immutable and already set to {current!r}. "
                    "You may not change it. If the character's response conflicts with "
                    "this, call report_contradiction instead."
                )
            # Immutable, unset — validate the proposed value before suspending
            if leaf["Type"] == "Enum":
                proposed_match = next(
                    (c for c in leaf["Constraint"] if c.lower() == str(value).lower()),
                    None,
                )
                if proposed_match is None:
                    return (
                        f"Error: {path}: {value!r} is not a valid value. "
                        f"Valid values: {', '.join(leaf['Constraint'])}"
                    )
                proposed: str | int | float | bool | None = proposed_match
            elif leaf["Type"] == "Integer":
                try:
                    proposed = int(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    return f"Error: {path}: {value!r} is not a valid integer"
            else:
                proposed = value

            if on_event is not None:
                await on_event(
                    SSEEvent(
                        event="sidechannel",
                        data={
                            "type": "fact_update_immutable_unset",
                            "turn_id": turn_id,
                            "path": path,
                            "proposed": proposed,
                        },
                    )
                )
            raw = await await_gate(session_id, turn_id)
            decision: dict[str, str | None] = (
                json.loads(raw) if raw is not None else {"action": "dismiss"}
            )
            action = decision.get("action", "dismiss")
            await store_decision(
                db,
                character_id=character.id,
                session_id=session_id,
                turn_id=turn_id,
                pass_name="character_evaluator",  # nosec B106
                tool_name="set_fact",
                tool_args={"path": path, "value": str(value)},
                user_input={"action": action, "value": decision.get("value")},
            )
            if action == "accept":
                _set_leaf(working_blob, path, proposed)
                await db_set_facts(db, character.id, working_blob)
                return f"Wrote {path} = {proposed!r}. Value is now locked immutably."
            if action == "edit":
                user_val_raw = str(decision.get("value") or "")
                if leaf["Type"] == "Enum":
                    em = next(
                        (c for c in leaf["Constraint"] if c.lower() == user_val_raw.lower()),
                        None,
                    )
                    if em is None:
                        _log.warning(
                            "set_fact approval: user edit %r for %s has no Enum match "
                            "— storing verbatim",
                            user_val_raw,
                            path,
                        )
                    user_val: str | int | float | bool | None = em if em is not None else user_val_raw
                elif leaf["Type"] == "Integer":
                    try:
                        user_val = int(user_val_raw)
                    except ValueError:
                        _log.warning(
                            "set_fact approval: user edit %r for %s is not a valid "
                            "integer — storing verbatim",
                            user_val_raw,
                            path,
                        )
                        user_val = user_val_raw
                else:
                    user_val = user_val_raw
                _set_leaf(working_blob, path, user_val)
                await db_set_facts(db, character.id, working_blob)
                _regeneration_needed[0] = True
                return (
                    f"Wrote {path} = {user_val!r}. Response will be regenerated "
                    "with this value."
                )
            # dismiss
            return f"No value recorded for {path}. Do not rely on the invented value."
```

The "already set" error branch is **unchanged** — it is already correct and tested.

---

### Part D — `src/memories/services/evaluator.py` — `_handle_set_fact` Mutable branch

**Current code** (lines 359–363 of the current `evaluator.py`, the `# Mutable` comment
and stub `return`):

```python
        # Mutable
        return (
            f"Error: {path} is Mutable. The approval flow for Mutable facts is not "
            "implemented yet — leave this path alone."
        )
```

**Replacement** (mirrors the Immutable-unset branch above with accept/edit/reject instead of
accept/edit/dismiss):

```python
        # Mutable — validate the proposed value before suspending
        if leaf["Type"] == "Enum":
            proposed_match = next(
                (c for c in leaf["Constraint"] if c.lower() == str(value).lower()),
                None,
            )
            if proposed_match is None:
                return (
                    f"Error: {path}: {value!r} is not a valid value. "
                    f"Valid values: {', '.join(leaf['Constraint'])}"
                )
            proposed = proposed_match
        elif leaf["Type"] == "Integer":
            try:
                proposed = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return f"Error: {path}: {value!r} is not a valid integer"
        else:
            proposed = value

        if on_event is not None:
            await on_event(
                SSEEvent(
                    event="sidechannel",
                    data={
                        "type": "fact_update_mutable",
                        "turn_id": turn_id,
                        "path": path,
                        "proposed": proposed,
                    },
                )
            )
        raw = await await_gate(session_id, turn_id)
        decision = json.loads(raw) if raw is not None else {"action": "reject"}
        action = decision.get("action", "reject")
        await store_decision(
            db,
            character_id=character.id,
            session_id=session_id,
            turn_id=turn_id,
            pass_name="character_evaluator",  # nosec B106
            tool_name="set_fact",
            tool_args={"path": path, "value": str(value)},
            user_input={"action": action, "value": decision.get("value")},
        )
        if action == "accept":
            _set_leaf(working_blob, path, proposed)
            await db_set_facts(db, character.id, working_blob)
            return f"Wrote {path} = {proposed!r}."
        if action == "edit":
            user_val_raw = str(decision.get("value") or "")
            if leaf["Type"] == "Enum":
                em = next(
                    (c for c in leaf["Constraint"] if c.lower() == user_val_raw.lower()),
                    None,
                )
                if em is None:
                    _log.warning(
                        "set_fact approval: user edit %r for %s has no Enum match "
                        "— storing verbatim",
                        user_val_raw,
                        path,
                    )
                user_val = em if em is not None else user_val_raw
            elif leaf["Type"] == "Integer":
                try:
                    user_val = int(user_val_raw)
                except ValueError:
                    _log.warning(
                        "set_fact approval: user edit %r for %s is not a valid "
                        "integer — storing verbatim",
                        user_val_raw,
                        path,
                    )
                    user_val = user_val_raw
            else:
                user_val = user_val_raw
            _set_leaf(working_blob, path, user_val)
            await db_set_facts(db, character.id, working_blob)
            _regeneration_needed[0] = True
            return f"Wrote {path} = {user_val!r}. Response will be regenerated with this value."
        # reject
        _regeneration_needed[0] = True
        return "Change rejected. Response will be regenerated without this update."
```

**Note on `proposed` type annotation**: in both branches, `proposed` may be assigned as
`proposed_match` (a `str`) or as `int(value)` or as `value` (a `str` coming from
`args.get("value")`). The variable is typed as `str | int | float | bool | None`
by assignment inference. No explicit annotation needed; this matches `_set_leaf`'s `value`
parameter type.

---

### Part E — `src/memories/services/chat_service.py` — `run_contradiction_loop()` handles `needs_regeneration`

**File:** `src/memories/services/chat_service.py`

**Current loop exit condition** (inside `for attempt in range(max_retries + 1):`):

```python
        if ev.verdict != "contradiction":
            eval_result = ev
            break
```

**Replacement:**

```python
        if ev.verdict != "contradiction" and not ev.needs_regeneration:
            eval_result = ev
            break
```

**Current max-retries guard** (at the bottom of the loop):

```python
        if attempt == max_retries:
            ev.max_retries_exceeded = True
            eval_result = ev
            break
```

This block is **unchanged**. When `needs_regeneration=True` consumes the last retry, the
response is delivered as-is with `max_retries_exceeded=True` — the same defensive posture
as contradiction-retry exhaustion. This is intentional: the caller must act eventually.

**No other changes** to `run_contradiction_loop()`. The `contradiction_hints` and
`contradiction_notifications` lists are only extended on `ev.verdict == "contradiction"`,
so a `needs_regeneration`-only continuation does not accumulate contradiction noise.

**Why regeneration works without an extra flag**: when `_handle_set_fact` writes an
edit value to `working_blob` and `run_evaluator()` returns that blob,
`run_contradiction_loop()` captures it via `ev, facts_blob = await run_evaluator(...)`.
On the next iteration, `build_system_prompt()` is called with the new `facts_blob`, so
the regenerated Character LLM sees the updated world state. The edit-value is in the DB
and the blob; no extra thread-through is needed.

---

### Part F — New `src/memories/routers/fact_approval.py`

New file, modelled on `require_fact.py`. The decision is JSON-encoded into the gate
string so `tool_gate.py`'s type (`asyncio.Queue[str | None]`) is unchanged.

```python
"""Accept, edit, reject, or dismiss endpoint for set_fact approval cards.

The client POSTs here when the user acts on a fact_update_mutable or
fact_update_immutable_unset blocking card surfaced by the Character Evaluator's
_handle_set_fact handler (see evaluator.run_evaluator._handle_set_fact).
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from memories.services.tool_gate import resolve_gate

router = APIRouter()


class _ApprovalBody(BaseModel):
    action: Literal["accept", "edit", "reject", "dismiss"]
    value: str | None = None


@router.post("/{session_id}/turns/{turn_id}/set-fact/respond")
async def respond_to_set_fact(
    session_id: int,
    turn_id: int,
    body: _ApprovalBody,
) -> dict[str, str]:
    """Resolve an active set_fact approval gate.

    Returns immediately without waiting for the SSE stream to resume.
    """
    payload = json.dumps({"action": body.action, "value": body.value})
    try:
        resolve_gate(session_id, turn_id, payload)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No pending set_fact for session {session_id} turn {turn_id}",
        ) from exc
    except asyncio.QueueFull as exc:
        raise HTTPException(status_code=409, detail="Already resolved") from exc

    return {"status": "ok"}
```

**`action` semantics:**
- `accept` — use the proposed LLM value (no `value` field needed)
- `edit` — use the user-supplied `value` field; regenerate
- `reject` — discard proposed change; regenerate (Mutable only)
- `dismiss` — discard proposed change; deliver as-is (Immutable-unset only)

The server does not validate that `reject` is only used for Mutable paths — it relies on
the frontend to send the correct action per card type. If `reject` arrives for an
Immutable-unset path, `_handle_set_fact` will set `_regeneration_needed[0] = True` via
the `else` branch (same as `reject` for Mutable). This is acceptable.

---

### Part G — `src/memories/main.py` — mount new router

```python
# Before (imports)
from memories.routers import (
    characters,
    chat,
    decisions,
    experiences,
    facts,
    implication,
    inferences,
    require_fact,
    schema,
    sessions,
)

# After (imports) — add fact_approval
from memories.routers import (
    characters,
    chat,
    decisions,
    experiences,
    fact_approval,
    facts,
    implication,
    inferences,
    require_fact,
    schema,
    sessions,
)
```

```python
# After existing require_fact mount line
app.include_router(require_fact.router, prefix="/api/sessions", tags=["require_fact"])
app.include_router(fact_approval.router, prefix="/api/sessions", tags=["fact_approval"])
```

---

### Part H — Dead code removal: `src/memories/routers/chat.py`

**Remove lines 104–119** (the two `implication`/`new_inference_probabilistic` verdict
checks and their bodies):

```python
# DELETE lines 104–105 (inside msg_data construction):
        if eval_result.verdict in ("implication", "new_inference_probabilistic"):
            msg_data["ungrounded"] = True

# DELETE lines 112–119 (the entire sidechannel emit block):
        # Emit sidechannel for non-contradiction violations / probabilistic inferences
        if eval_result.verdict in ("implication", "new_inference_probabilistic"):
            sc_payload: dict[str, object] = {
                "type": eval_result.verdict,
                "turn_id": turn_id,
                "violations": [v.model_dump() for v in eval_result.violations],
                "new_inferences": [i.model_dump() for i in eval_result.new_inferences],
            }
            yield f"event: sidechannel\ndata: {json.dumps(sc_payload)}\n\n"
```

After the removal, `msg_data` no longer has an `ungrounded` key, and no `implication`-
type sidechannel is ever emitted. The `done` event follows immediately after `message`.

---

### Part I — Dead code removal: `src/memories/routers/implication.py`

**Remove lines 156–160** inside `accept_implication()` (the final `if ev.verdict in ...`
block):

```python
# DELETE these lines from accept_implication():
    if ev.verdict in ("implication", "new_inference_probabilistic"):
        response["ungrounded"] = True
        response["violations"] = [v.model_dump() for v in ev.violations]
        response["new_inferences"] = [i.model_dump() for i in ev.new_inferences]
```

The `response` dict returned by `accept_implication()` will always be
`{"content": new_content, "turn_id": turn_id}` — no `ungrounded` key.

---

## Transitional State After Step 7

After Step 7 and before Step 8:

- All three mutability tiers are fully wired server-side. A `set_fact` call on any path
  now does the right thing: Fluid writes immediately, Mutable suspends for user approval,
  Immutable-already-set triggers contradiction, Immutable-unset suspends for user approval.
- Both new sidechannel types (`fact_update_mutable`, `fact_update_immutable_unset`) are
  emitted but are inert in the UI — `chat.js`'s `buildNotificationFromSidechannel()`
  returns `null` for unknown types, so no card appears. Step 8 wires them in with the
  four-part commit rule.
- `POST /api/sessions/{id}/turns/{id}/set-fact/respond` is live and functional.
- The character's response is withheld (not yet stored in DB) when the evaluator suspends
  on a Mutable or Immutable-unset gate. Accept and dismiss deliver the existing response.
  Edit and reject trigger regeneration within `run_contradiction_loop()`, which re-runs
  the Character LLM with the updated `facts_blob` before storing any assistant message.
- Dead code is gone: no `ungrounded` flag in `message` events, no implication sidechannel
  from `chat.py`, no implication verdict check in `accept_implication()`.
- `routers/implication.py`'s `accept-implication` endpoint remains mounted (it is still
  needed to handle the old fact model during the transition; full removal is Step 9+).

---

## Test Plan

### `tests/unit/test_evaluator_service.py` — changes

**Tests to delete:**
- `test_set_fact_immutable_unset_returns_stub_error` — stub behavior is replaced
- `test_set_fact_mutable_path_returns_stub_error_regardless_of_current_value` — stub
  behavior is replaced

**Tests to add** (new section: `# _handle_set_fact — immutable-unset path`):

All immutable-unset tests require a gate. Use a `create_gate(1, 1)` / `cleanup_gate(1, 1)`
pair wrapping each test via `try`/`finally`, or a fixture. The pattern mirrors the
`test_chat_service.py` approach: `asyncio.gather(run_evaluator(...), _resolver())` where
`_resolver` waits for the sidechannel event then calls `resolve_gate(1, 1, payload)`.

- `test_set_fact_immutable_unset_accept_writes_value` — mock returns
  `[set_fact(Character.Identity.Name, "Sarah"), report_pass]`; resolver puts
  `json.dumps({"action": "accept"})` after the sidechannel event; assert
  `blob["Character"]["Identity"]["Name"]["Value"] == "Sarah"` and `result.verdict ==
  "pass"` and `result.needs_regeneration == False`.
- `test_set_fact_immutable_unset_accept_logs_decision` — same flow; assert a `decisions`
  row with `tool_name="set_fact"`, `tool_args={"path": "Character.Identity.Name",
  "value": "Sarah"}`, `user_input={"action": "accept", "value": None}`.
- `test_set_fact_immutable_unset_edit_writes_user_value_and_sets_needs_regeneration` —
  resolver puts `json.dumps({"action": "edit", "value": "Alice"})`; assert blob has
  `"Alice"`, `result.needs_regeneration == True`.
- `test_set_fact_immutable_unset_dismiss_does_not_write_and_no_regeneration` — resolver
  puts `json.dumps({"action": "dismiss"})`; assert `blob` has no `Name` entry under
  `Character.Identity`, `result.needs_regeneration == False`.
- `test_set_fact_immutable_unset_emits_sidechannel_card` — `on_event` collector; assert
  an event with `data["type"] == "fact_update_immutable_unset"`, `data["path"] ==
  "Character.Identity.Name"`, and `data["proposed"] == "Sarah"` is emitted before
  `run_evaluator()` returns.
- `test_set_fact_immutable_unset_invalid_enum_returns_error_before_suspension` — mock
  returns `set_fact(Character.Appearance.Body.Build, "Gigantic")` (not in Constraint)
  + `report_pass`; NO gate; assert no suspension occurs (the error is returned to the
  LLM inside the same round), `blob` has no `Build` entry, `result.needs_regeneration
  == False`.
- `test_set_fact_immutable_unset_edit_enum_coerced_case_insensitively` — resolver puts
  `{"action": "edit", "value": "athletic"}` for a Constraint path; assert stored value
  is `"Athletic"`.
- `test_set_fact_immutable_unset_edit_integer_coerced_from_string` — use
  `Setting.Temporal.Current-Year` monkeypatched to `Immutable`; resolver puts
  `{"action": "edit", "value": "2025"}`; assert stored value is `int(2025)`.

**Tests to add** (new section: `# _handle_set_fact — mutable path`):

- `test_set_fact_mutable_accept_writes_value` — Mutable path `Character.Identity.Occupation`
  (Mutable, String); mock returns `[set_fact(...), report_pass]`; resolver puts
  `json.dumps({"action": "accept"})`; assert blob has `"engineer"`,
  `result.needs_regeneration == False`.
- `test_set_fact_mutable_accept_logs_decision` — same; assert decision row with
  `user_input={"action": "accept", "value": None}`.
- `test_set_fact_mutable_edit_writes_user_value_and_sets_needs_regeneration` — resolver
  puts `json.dumps({"action": "edit", "value": "doctor"})`; assert blob has `"doctor"`,
  `result.needs_regeneration == True`.
- `test_set_fact_mutable_reject_does_not_write_and_sets_needs_regeneration` — resolver
  puts `json.dumps({"action": "reject"})`; assert blob unchanged,
  `result.needs_regeneration == True`.
- `test_set_fact_mutable_emits_sidechannel_card` — `on_event` collector; assert event
  with `data["type"] == "fact_update_mutable"`, correct `path` and `proposed`.
- `test_set_fact_mutable_invalid_enum_returns_error_before_suspension` — same pattern as
  the immutable-unset invalid-enum test; no gate, no suspension.

**Tests to add** (new section: `# EvaluatorResult.needs_regeneration`):

- `test_evaluator_result_needs_regeneration_defaults_to_false` — `report_pass`-only mock;
  assert `result.needs_regeneration == False`.
- `test_evaluator_result_needs_regeneration_false_on_contradiction` — `report_contradiction`
  mock; assert `result.needs_regeneration == False`.

---

### `tests/unit/test_chat_service.py` — additions

New section: `# Step 7 additions — needs_regeneration handling`:

- `test_run_contradiction_loop_needs_regeneration_causes_extra_iteration` — mock the
  evaluator to return `needs_regeneration=True` on the first call, then `needs_regeneration=
  False` on the second call. Assert the Character LLM is invoked twice (two `chat_with_tools`
  calls for the character; two evaluator calls total — four total HTTP calls counting World
  Builder and both evaluators). The simplest mock: first evaluator call returns
  `report_pass` but with `_regeneration_needed[0] = True` set — but since we can't
  easily set that from outside, mock `run_evaluator` directly:
  
  ```python
  import unittest.mock
  from memories.services.evaluator import EvaluatorResult
  
  call_count = 0
  async def _fake_evaluator(*args, **kwargs):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
          return EvaluatorResult(verdict="pass", decision_log="regen", 
                                 needs_regeneration=True), {}
      return EvaluatorResult(verdict="pass", decision_log="ok"), {}
  
  with unittest.mock.patch("memories.services.chat_service.run_evaluator", _fake_evaluator):
      with respx.mock:
          respx.post(_CHAT_URL).mock(side_effect=[
              _mock_world_builder(),
              _mock_ok("Response 1"),  # char LLM attempt 1
              _mock_ok("Response 2"),  # char LLM attempt 2
          ])
          content, _, _ = await run_contradiction_loop(db, 1, 1, model, messages,
                                                        character, {}, "hi", ollama,
                                                        on_event=None)
  assert content == "Response 2"
  assert call_count == 2
  ```

- `test_run_contradiction_loop_needs_regeneration_does_not_increment_contradiction_count`
  — same `_fake_evaluator` pattern; assert `eval_result.contradiction_notifications` is
  empty (the extra iteration is a mutable-regen, not a contradiction).

- `test_run_contradiction_loop_needs_regeneration_uses_updated_blob` — `_fake_evaluator`
  that on its first call modifies `facts_blob` (by returning an updated blob with a new
  value) and sets `needs_regeneration=True`; on its second call captures the `facts_blob`
  argument; assert the second call received the updated blob.

---

### `tests/integration/test_api_chat.py` — updates

**Tests to update:**
- `test_message_event_has_no_ungrounded_key` — this test may not exist yet; add it: after
  a complete turn (with `_mock_ok()` for the character LLM), assert the parsed `message`
  event dict does not contain `"ungrounded"`. This verifies Part H's dead code removal.
- Any existing test that asserts `msg_data["ungrounded"]` must be deleted (grep the file;
  as of Step 6, none should exist since the field was never set in normal passes).

---

### New `tests/integration/test_set_fact_approval_live.py`

Real end-to-end coverage through the production `POST /api/sessions/{id}/messages` SSE
endpoint. Must use a real uvicorn server (same `live_client` fixture pattern as
`test_require_fact_live.py` — copy the `_bound_socket`, `_wait_until`, `_SseCollector`,
and `live_client` fixture verbatim or import them from a shared `conftest`).

**Tests:**

- `test_set_fact_mutable_sidechannel_emitted_before_done` — mock `_OLLAMA_CHAT_URL` with
  World-Builder no-op, `[set_fact(Character.Identity.Occupation, "surgeon"), report_pass]`,
  then no more calls needed; start consuming SSE stream; assert a `sidechannel` event with
  `type == "fact_update_mutable"` arrives before `done`.
- `test_set_fact_mutable_accept_delivers_response` — after observing the sidechannel,
  `POST .../set-fact/respond` with `{"action": "accept"}`; assert the eventual `message`
  event content matches the character-LLM mock, and stream ends with `done`.
- `test_set_fact_mutable_accept_writes_fact_to_db` — same flow; after stream completes,
  `get_facts(db, char.id)` has `Character.Identity.Occupation.Value == "surgeon"`.
- `test_set_fact_mutable_reject_triggers_regeneration` — after sidechannel, POST
  `{"action": "reject"}`; mock includes a second character-LLM response; assert the
  `message` event content matches the SECOND mocked response (the regenerated one). Total
  mock sequence: `[world_builder_noop, set_fact+report_pass, char_llm_R2, report_pass_R2]`.
- `test_set_fact_mutable_edit_writes_user_value` — POST `{"action": "edit", "value":
  "nurse"}`; stream completes; `get_facts(db, char.id)` has
  `Character.Identity.Occupation.Value == "nurse"`.
- `test_set_fact_immutable_unset_sidechannel_emitted` — same pattern; mock the evaluator
  calling `set_fact(Character.Identity.Name, "Elena")`; assert
  `type == "fact_update_immutable_unset"`, `proposed == "Elena"`.
- `test_set_fact_immutable_unset_dismiss_delivers_without_writing` — POST
  `{"action": "dismiss"}`; stream completes; `get_facts(db, char.id)` has no `Name`
  value under `Character.Identity`.
- `test_set_fact_respond_unknown_turn_returns_404` — `POST .../turns/9999/set-fact/respond`
  with no matching gate → 404.
- `test_set_fact_respond_double_resolve_returns_409` — POST twice to the same gate → 409.

Each test creates its own character/session inline (same pattern as
`test_require_fact_live.py`).

---

### `tests/integration/test_api_implication.py` — updates

- Any test that asserts `response["ungrounded"]` on the result of `accept_implication()`
  (Part I removal) must be updated to not expect that key. Grep the file for `"ungrounded"`.

---

## Edge Cases

- **Gate contention between `require_fact` and `set_fact`**: the gate is a single
  `asyncio.Queue(maxsize=1)` per `(session_id, turn_id)`. The Character LLM's
  `require_fact` and the Evaluator's `set_fact` suspension both use the same queue. Since
  the two passes run sequentially (Character LLM first, then Evaluator), they cannot both
  be waiting on the gate simultaneously. There is no race to guard against.

- **Multiple `set_fact` suspensions in one evaluator pass**: if the evaluator batches
  multiple `set_fact` calls in one model response (e.g.
  `[set_fact(Mutable_A), set_fact(Mutable_B), report_pass]`), they are processed
  sequentially by `chat_with_tools()`. The first suspension blocks the second. The user
  sees one blocking card, resolves it, and only then sees the second card. This is
  correct and requires no extra handling — it falls out of the sequential processing
  already in place.

- **Enum validation loop risk** (see `project_enum_validation_prompt_risk` memory): the
  strict validation BEFORE suspension returns a single error message instructing the LLM
  to use a valid value. Unlike the earlier enum risk (where the model loops through all
  valid values), this is a pre-suspension error returned to the LLM tool-call result.
  The LLM can self-correct with one valid value on its next tool call. Returning the
  full `Constraint` list in the error message is deliberate — the LLM needs it to pick a
  valid value. If the LLM loops through all values in successive retries, the
  `MAX_TOOL_CALL_ROUNDS` cap stops it before it suspends.

- **`reject` vs `dismiss` sent to the wrong path type**: if the frontend sends `reject`
  for an Immutable-unset card (or `dismiss` for a Mutable card), the server's
  `_handle_set_fact` handler falls through to its `else` branch, which sets
  `_regeneration_needed[0] = True` for Immutable-unset (unintended regeneration) or
  skips the write and does NOT regenerate for Mutable (unintended delivery of the
  original response). Step 8's type-aware frontend cards prevent this by only showing
  the correct action buttons per card type; no server-side validation is added here.

- **`on_event is None` with a suspension**: if `on_event` is `None` (no SSE stream, e.g.
  a test that calls `run_evaluator()` directly without an event callback), the sidechannel
  emit is skipped but `await_gate()` still suspends. A test that expects suspension with
  `on_event=None` must still call `resolve_gate()` concurrently. Most unit tests will
  pass a real `on_event` and use it as the signal to fire the resolver.

- **ASGITransport SSE limitation**: the `test_set_fact_approval_live.py` tests use a real
  uvicorn server (not the `ASGITransport`-backed `client` fixture) for the same reason
  as `test_require_fact_live.py`: `ASGITransport` buffers the full response, which
  deadlocks a test that must POST to `set-fact/respond` while the SSE stream is still
  open.

- **Regeneration and the `max_retries` budget**: mutable edit/reject regeneration passes
  consume the same `MAX_CONTRADICTION_RETRIES` budget as contradiction retries. If a
  user edits a fact 3 times in one turn and the model keeps triggering more mutable
  updates, the loop exits after 3 total iterations. The last generated response is
  delivered with `max_retries_exceeded=True`. This is an unlikely scenario in practice.

- **`working_blob` threading across multiple mutable suspensions**: `working_blob` is
  the single deep-copied blob used throughout one `run_evaluator()` invocation. Each
  accepted `set_fact` write (`_set_leaf(working_blob, ...)` + `db_set_facts(...)`)
  updates it in place. The final returned blob reflects all writes from all tool calls
  in that pass. `run_contradiction_loop()` captures it via `ev, facts_blob = await
  run_evaluator(...)`, so the next Character-LLM attempt uses the fully-updated blob.

---

## Post-Implementation Cleanup Tasks

### CT-1: Regeneration re-runs the Character LLM with a stale system prompt

**Decided:** Fix before proceeding

The headline feature of Step 7 — "the character's response must be regenerated with the
updated world state" (Overview) — does not actually deliver an updated response. Part E's
rationale claims that on a `needs_regeneration` continuation, "`build_system_prompt()` is
called with the new `facts_blob`, so the regenerated Character LLM sees the updated world
state." But `run_contradiction_loop()` in [chat_service.py](src/memories/services/chat_service.py#L395-L413)
never calls `build_system_prompt()`. It rebuilds `messages = list(base_messages)` each
iteration, and `base_messages` (with its system prompt at index 0) is built once in
`run_turn()` at [chat_service.py:539](src/memories/services/chat_service.py#L539) before
the loop and is never rebuilt. Unlike the contradiction path — which appends an explicit
hint message so the LLM knows what to fix — the `needs_regeneration` path appends nothing
(`contradiction_hints` stays empty). So the regenerated Character LLM gets the pre-edit
facts in its system prompt and no signal about the change. For a Mutable edit
(Occupation → "doctor"), the character re-generates from the identical stale prompt, likely
implies the original value again ("engineer"), and the evaluator re-suspends on the same
path — burning `MAX_CONTRADICTION_RETRIES` while never reflecting the user's edit. The
unit test `test_run_contradiction_loop_needs_regeneration_uses_updated_blob` only asserts
the second `run_evaluator` call received the updated blob; it never checks the Character
LLM's prompt, so the gap is invisible to the suite.

**What to do:**
1. In `run_contradiction_loop()`, after `ev, facts_blob = await run_evaluator(...)` returns
   the updated blob, rebuild the system prompt from it before the next Character-LLM
   iteration. `build_system_prompt` is already imported; the loop already receives
   `character`, `inferences`, and `experiences`. Rebuild with
   `build_system_prompt(character, facts_blob, inferences, experiences)` and replace the
   system entry of `base_messages` (index 0) so the regenerated prompt renders the current
   fact values.
2. Guard the rebuild so it only fires when `facts_blob` actually changed (e.g. on
   `needs_regeneration` or contradiction continuations), to avoid needless work on the
   clean-break path.
3. Strengthen `test_run_contradiction_loop_needs_regeneration_uses_updated_blob` (or add a
   sibling test) to assert the *second* `chat_with_tools` character call received a system
   message reflecting the updated blob — not just that `run_evaluator` did.

### CT-2: Unreachable duplicate `_handle_set_fact` added to `run_contradiction_loop()`

**Decided:** Fix before proceeding

The spec ("What This Step Does NOT Change") states chat_service.py's only change is Part E's
loop-exit condition and that `_handle_require_fact` / `_REQUIRE_FACT_TOOL` are untouched.
The implementation instead added a ~166-line `_handle_set_fact` closure to
`run_contradiction_loop()` ([chat_service.py:230-393](src/memories/services/chat_service.py#L230-L393))
and registered it in the character LLM's handler dict at
[chat_service.py:411](src/memories/services/chat_service.py#L411)
(`{"require_fact": _handle_require_fact, "set_fact": _handle_set_fact}`). But the tool
*list* advertised to the model is still `[_REQUIRE_FACT_TOOL]` — `set_fact` is never sent
to Ollama, so the model cannot call it. `chat_with_tools()` only dispatches handlers for
tool calls the model actually emits, so this closure is unreachable. Coverage confirms it:
chat_service.py fell to 72% with lines 236-393 unexercised. Worse, this duplicate has
*different* semantics from the evaluator's version — it writes edits but never sets any
regeneration flag — so if a future change adds `set_fact` to the character tool list, it
would silently get the wrong (non-regenerating) behavior. This is dead, duplicated, and
subtly-divergent code that contradicts the spec.

**What to do:**
1. Delete the entire `_handle_set_fact` closure from `run_contradiction_loop()`
   (chat_service.py:230-393).
2. Revert the handler registration at line 411 back to `{"require_fact": _handle_require_fact}`.
3. Remove the now-unused `import json` at [chat_service.py:7](src/memories/services/chat_service.py#L7)
   (its only uses were lines 301 and 365 inside the deleted closure).
4. Re-run `uv run pytest` to confirm nothing depended on the dead handler (coverage shows
   nothing does) and chat_service.py coverage recovers.

### CT-3: Ruff RUF059 failure — unused `ev` in `accept_implication()`

**Decided:** Fix before proceeding

Part I removed the `if ev.verdict in (...)` block from `accept_implication()`, which held
the only uses of `ev`. The unpacking at
[implication.py:133](src/memories/routers/implication.py#L133)
(`new_content, _, ev = await run_contradiction_loop(...)`) now leaves `ev` unused, and
`uv run ruff check src/` fails with `RUF059 Unpacked variable 'ev' is never used`. Per the
project's ruff-in-pre-commit convention this blocks a clean commit. The comment at
[implication.py:147](src/memories/routers/implication.py#L147)
("Replace the stored message and clear the ungrounded flag") is also now stale — there is
no ungrounded flag to clear.

**What to do:**
1. Change the unpacking at implication.py:133 to `new_content, _, _ = await run_contradiction_loop(...)`.
2. Update the stale comment at line 147 to drop the "clear the ungrounded flag" clause.
3. Confirm `uv run ruff check src/` passes.

### CT-4: `tool_gate._resolved` grows unbounded and deviates from the "unchanged" spec claim

**Decided:** Fix in follow-up

The spec lists `tool_gate.py` under "What This Step Does NOT Change" ("no new functions"),
but the implementation added a module-level `_resolved: set[tuple[int, int]]` and rewired
`await_gate`/`resolve_gate`/`cleanup_gate` in [tool_gate.py](src/memories/services/tool_gate.py)
to make a double-resolve return 409 even after `cleanup_gate` has run (needed for
`test_set_fact_respond_double_resolve_returns_409`). The change is justified, but
`_resolved` is intentionally never cleared — `cleanup_gate` leaves entries in place — so
the set grows by one `(session_id, turn_id)` entry for every resolved gate over the
server's lifetime. On a long-running local server this is a slow unbounded leak. It is
tiny per entry, hence follow-up rather than blocking, but it should be bounded or the
deviation documented.

**What to do:**
1. Bound `_resolved` growth — e.g. cap it to the most-recent N keys, or evict entries once
   a turn is provably complete (a later `turn_id` for the same `session_id` supersedes
   earlier ones), while preserving the after-cleanup double-resolve → 409 behavior the
   tests rely on.
2. Update the Step 7 spec's "What This Step Does NOT Change" note to acknowledge the
   tool_gate change, so future reviewers do not treat the deviation as accidental.
