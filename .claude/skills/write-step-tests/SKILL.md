---
name: write-step-tests
description: >
  Write the test suite for a specific implementation step in the memories
  project. Trigger when the user invokes /write-step-tests N or asks to write
  tests, implement the test plan, or add tests for a step. Reads docs/stepN.md
  for the test plan, reads existing test fixtures and helpers for conventions,
  and writes all tests listed in the plan. Tests are expected to fail after
  this step — the business logic is not implemented yet.
---

# Write Step Tests

Your job is to implement the test plan from `docs/stepN.md`. When you finish,
the test suite should be syntactically valid and all new tests should **fail**
with an error or assertion failure — not pass. Passing tests at this stage mean
the business logic already exists, which is not the expected state.

## 1. Read the design doc

The step number comes from the skill argument. Read `docs/stepN.md` in full,
paying particular attention to the **Test Plan** section. It lists, per file:
- Tests to delete
- Tests to update (signature or fixture changes only)
- Tests to add (with exact test names and what each verifies)

## 2. Read the existing test infrastructure

Before writing anything, read:
- `tests/unit/conftest.py` — unit fixtures: `character`, `session`, `fact`,
  `ollama`, `make_ollama_ndjson()`, `make_evaluator_ndjson()`
- `tests/integration/conftest.py` — integration fixtures: `db`, `client`,
  overrides for `get_db` and `get_ollama`
- The existing test file(s) you are modifying, to understand current patterns

The helpers `make_ollama_ndjson()` and `make_evaluator_ndjson()` in
`tests/unit/conftest.py` build mock Ollama NDJSON responses. Use them for all
unit tests that mock Ollama HTTP calls. Integration tests use `respx` to
intercept the actual httpx calls made by `OllamaClient`.

## 3. Implement the test plan

Work through each test file in the plan:

**Deletions first** — remove the listed tests entirely. Do not leave them
commented out.

**Updates next** — apply the described changes (signature, fixture name,
assertion format). Do not change what the test verifies, only how it invokes
the code.

**Additions last** — write each new test. Match the exact function name from
the plan. Each test should:
- Have one clear assertion
- Use the fixtures already defined in conftest (don't duplicate fixture logic
  inline)
- Not import or call any function that doesn't exist yet — if the function
  being tested doesn't exist, the test should import it anyway (it will fail
  at import time or at call time, which is the expected failure mode)

## 4. Pitfalls to avoid

**ASGITransport and SSE:** the `client` fixture in `tests/integration/conftest.py`
uses `ASGITransport`, which buffers the full HTTP response before returning it.
This means any test that needs to make a second HTTP request *while an SSE
stream is open* (e.g. sending a `require_fact` response while a chat stream is
active) cannot use the `client` fixture. It must spin up a real uvicorn server.
If the step doc calls this out in Edge Cases, follow the existing pattern in
`tests/integration/test_api_implication.py` (or wherever real-server tests
were last added).

**Pydantic field renames:** if the step renames a Pydantic model field, update
every mock payload in the test files that constructs that model. Pydantic
ignores unknown fields by default — a payload with the old field name will
silently populate the new field with its default value, producing a test that
passes for the wrong reason.

**Evaluator mock shape:** `make_evaluator_ndjson()` builds the evaluator
response. If Step 3 or later removed fields from `EvaluatorResult` (e.g.
`experience_updates`), do not pass those fields to `make_evaluator_ndjson()`.
Check the current signature before calling it.

## 5. Verify syntax and expected failures

Run the affected test files only:

```
uv run pytest --no-cov tests/unit/test_foo.py tests/integration/test_bar.py -x
```

Expected outcomes:
- **`ImportError` or `AttributeError`** for functions/classes that don't exist
  yet — this is correct
- **`AssertionError`** for tests that can import the target but the logic isn't
  there — also correct
- **Unexplained passes** — investigate; the logic may already exist, or the
  test may be asserting something trivially true

Then run ruff on the test files:

```
uv run ruff check tests/
```

Fix any violations before committing. Common ones:
- **E501**: lines over 100 characters — split the line
- **B904**: `raise X` inside an `except` block without `from` — add `from err`
  or `from None`

## 6. Commit

```
git add tests/
git commit -m "test(stepN): <brief description of what tests cover>"
```
