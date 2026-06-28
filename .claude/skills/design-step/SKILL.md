---
name: design-step
description: >
  Write a detailed technical design doc for a specific implementation step in
  the memories project. Trigger when the user invokes /design-step N (where N
  is a step number from docs/plan-v2.md) or asks to write a design doc,
  implementation plan, or technical spec for a specific step. Reads
  plan-v2.md, current source files, and prior step docs to produce
  docs/stepN.md in the project's established format.
---

# Design Step

Your job is to write `docs/stepN.md` — a technical design doc detailed enough
that a fresh Claude session can implement both the tests and the business logic
from it alone, without re-deriving any decisions.

## 1. Identify the step

The step number or name comes from the skill argument. If missing, ask.

Read `docs/plan-v2.md` in full, then locate the step in the **Implementation
Sketch** section. Note: what it builds, what it explicitly defers to later
steps, and what it requires from previous steps.

## 2. Orient yourself in the prior work

Read the two most recently completed step docs (e.g. `docs/step2.md` and
`docs/step3.md`). Their "What Steps X Delivered" sections tell you the actual
current state of the codebase — treat this as ground truth for what exists.

Then read `CLAUDE.md` for architecture conventions and test layout.

## 3. Read the source files this step will touch

Derive the file list from the plan sketch. Read every file that will change.
Do not skip this — the design doc must reference real function signatures, real
field names, and real line numbers. If a function doesn't exist yet, say so.

Typical files to check: `src/memories/database.py`, relevant service files
under `src/memories/services/`, the affected router under
`src/memories/routers/`, `src/memories/models/__init__.py`, and the test files
under `tests/unit/` and `tests/integration/`.

## 4. Write docs/stepN.md

Use this section structure, in order:

**`# Step N — Title`**

**`## Overview`**  
Two to four paragraphs. What does this step accomplish? What is the transitional
state it leaves the system in? End with a **Success criterion:** sentence that
names specific test files that must pass.

---

**`## What Steps X and Y Delivered`**  
Bullet list of concrete deliverables from previous steps this one depends on:
file names, function signatures, table names. Be specific. This section lets an
implementer verify they have the right foundation before starting.

---

**`## What This Step Does NOT Change`**  
Explicit out-of-scope list. Name specific files and functions that are
deliberately left alone. This prevents scope creep during implementation.

---

**`## Detailed Design`**  
One subsection per component, ordered by dependency (lower-level dependencies
first). Each subsection must name the file and include:
- Before/after function signatures as fenced code blocks when a signature changes
- Field additions or removals with types
- Logic description precise enough to implement without ambiguity
- Any imports that need to be added or removed

**`## Transitional State After Step N`**  
Describe the system state after this step completes but before the next step
lands. Name what still doesn't work and confirm that this is accepted.

---

**`## Test Plan`**  
One subsection per test file (`tests/unit/test_foo.py — changes`). Under each:

- **Tests to delete:** exact `test_function_name` and one-line reason
- **Tests to update:** exact `test_function_name` and what changes (signature,
  fixture name, assertion format)
- **Tests to add:** exact `test_function_name` and one sentence describing what
  it verifies

Every test name must be a valid Python identifier. Vague names like "tests for
the new feature" are not acceptable.

---

**`## Edge Cases`**  
Bullet list of non-obvious boundary conditions. For each: what the condition is
and how the implementation must handle it.

Always check whether any of these project-specific pitfalls apply to this step:

- **Enum validation loop risk:** if the step adds Enum fact validation, error
  messages must return exactly one replacement hint — not the full constraint
  list. If the model receives a list of valid values in the error text, it tends
  to loop through all of them in successive retries.
- **ASGITransport SSE limitation:** `ASGITransport` (used in integration test
  fixtures) buffers the full response before returning it. Any test that makes
  concurrent HTTP requests while an SSE stream is open must use a real uvicorn
  server via `pytest-anyio` with `server_url`, not the `client` fixture. Flag
  this in Edge Cases if the step involves SSE endpoints.
- **Pydantic silent field drops:** if any Pydantic model field is being renamed,
  warn that mock payloads in tests must be updated. Pydantic ignores unknown
  fields by default, so old field names in test fixtures will silently produce
  empty values without raising an error.

---

**`## Post-Implementation Cleanup Tasks`**  
Leave this section present but empty. The `/review-step` skill populates it
after implementation.

## 5. Commit

```
git add docs/stepN.md
git commit -m "docs(stepN): <brief description of what the step does>"
```
