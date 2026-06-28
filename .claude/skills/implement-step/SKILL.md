---
name: implement-step
description: >
  Implement the business logic for a specific step in the memories project,
  making a pre-written test suite pass. Trigger when the user invokes
  /implement-step N or asks to implement a step, write the business logic for
  a step, or make the tests pass for a step. Reads docs/stepN.md for the
  design spec, runs the existing failing tests to understand what's needed,
  and implements only what the spec describes. Does not touch test files.
---

# Implement Step

Your job is to make the failing tests for step N pass by implementing the
business logic described in `docs/stepN.md`. The test suite already exists
(written by `/write-step-tests`). Do not modify test files.

## 1. Read the design doc

The step number comes from the skill argument. Read `docs/stepN.md` in full.
Pay particular attention to:
- **Detailed Design** — exact signatures, field names, logic
- **What This Step Does NOT Change** — the explicit out-of-scope list; do not
  implement anything on this list even if it seems helpful
- **Edge Cases** — handle these deliberately, not as afterthoughts

Also read `CLAUDE.md` for architecture conventions and the key patterns section.

## 2. See the current failure picture

Before writing any code, run the tests to understand what's failing and why:

```
uv run pytest --no-cov -x 2>&1 | head -80
```

This tells you the exact import errors, attribute errors, and assertion failures
you need to resolve, and in what order.

## 3. Implement in dependency order

Work bottom-up: DB schema and repository functions before services, services
before routers. This keeps each step compilable and testable as you go.

Typical order for this project:
1. DDL changes in `src/memories/database.py` (new tables, dropped columns,
   new repository functions)
2. Model changes in `src/memories/models/__init__.py`
3. Service-layer changes (`src/memories/services/`)
4. Router changes (`src/memories/routers/`)
5. Import cleanup in each modified file

After each file, run the tests for that file's layer to catch regressions early.

## 4. Implementation rules

**Do not add anything the tests don't cover.** If the spec mentions something
that has no corresponding test, implement only what is needed for the tests
that exist. Do not design for the next step.

**Do not touch test files.** If a test seems wrong, re-read the spec before
concluding there is a test error. The tests were written from the spec; the
implementation should match both.

**Do not add comments** unless they capture a non-obvious invariant. The code
should be self-documenting via names.

## 5. Pre-commit gate

Before declaring done, run the full pre-commit suite:

```
uv run pre-commit run --all-files
```

The hooks that most commonly fail on new code:

**ruff E501** — lines over 100 characters. Break the line. Long string literals
can use implicit concatenation. Long function signatures can use one-arg-per-line
style.

**ruff B904** — bare `raise X` inside an `except` block. Use `raise X from err`
to chain the exception, or `raise X from None` to explicitly suppress the chain.
This fires frequently when wrapping library exceptions in project-specific ones.

**mypy** — runs on `src/` only (not `tests/`). Fix all type errors. Common
causes: missing `Optional`, wrong return type, unguarded `None` access.

**bandit** — security scan. The most common false positive in this project is
`B101` (assert in non-test code); the `pyproject.toml` bandit config skips it
in `src/`. If you get a real bandit finding, fix it — don't skip the check.

## 6. Final test run

After pre-commit passes, run the full suite with coverage:

```
uv run pytest
```

The project enforces **80% overall coverage**. If you fall below threshold,
add coverage to the untested branches before committing — do not add
`# pragma: no cover` unless a branch is genuinely unreachable.

## 7. Commit

```
git add src/
git commit -m "feat(stepN): <brief description of what was implemented>"
```

If a pre-commit hook auto-formats files, stage those changes too before
committing.
