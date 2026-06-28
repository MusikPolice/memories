---
name: review-step
description: >
  Adversarially review a completed implementation step in the memories project.
  Trigger when the user invokes /review-step N or asks for an adversarial
  review, post-implementation review, or cleanup audit for a step. Reads the
  design doc and all changed source and test files, hunts for gaps between spec
  and implementation, and appends numbered CT findings to the design doc's
  Post-Implementation Cleanup Tasks section.
---

# Review Step

Your job is to act as an adversary. You are looking for what is wrong, missing,
misleading, or likely to break — not to validate what is working. Every finding
that goes into the design doc is a finding that a future session will fix, so
prioritise signal over noise: a specific, actionable problem is worth more than
a vague observation.

## 1. Gather material

**Read the design doc:** `docs/stepN.md`. This is the spec. Anything the spec
says should be done that isn't done is a finding.

**Find the step's commits:**
```
git log --oneline | grep "stepN"
```
This gives you the commit SHAs for the test commit (`test(stepN):`) and the
implementation commit (`feat(stepN):`). If there are fix commits too, include
them.

**Read the diff:**
```
git show <sha> --stat
git diff <earliest-sha>^..<latest-sha>
```

**Read every changed source file in full.** Do not rely on the diff alone —
the diff shows what changed, but bugs often hide in what didn't change but
should have.

## 2. Systematic review angles

Work through each of these in order. For each, note every finding.

### A. Spec vs. implementation gaps
Compare the **Detailed Design** section against the implementation line by line.
Look for:
- Functions, fields, or endpoints described in the spec that are absent from
  the code
- Implementations that deviate from the spec's prescribed signature or logic
- Stub implementations that return placeholder values instead of real ones
- Columns described in DDL that are not parsed in `_parse_*` functions
- Fields added to Pydantic models but not wired into callers

### B. Test coverage completeness
For each **Tests to add** entry in the Test Plan, confirm the test exists and
actually tests what the plan says it does. Look for:
- Tests listed in the plan that were never written
- Tests that import the right function but assert on a trivially true condition
  (e.g. `assert result is not None` when the plan said to assert a specific value)
- Tests that pass because of **unrelated content** — check that the assertion
  is satisfied by the feature under test, not by coincidental string content
  elsewhere in the output. The clearest example: a test asserting `"Chicago"
  in system_prompt` that passes because "Chicago" appears in a schema description
  string, not because the extraction result reached the prompt.

### C. Dead code
Look for:
- Functions or classes defined but never called after the refactor
- Imports that were not removed when the thing they import was removed
- Parameters accepted by a function that are never used in the body
- Private helpers (`_foo`) that exist but have no callers

### D. Pydantic field rename pitfalls
If any Pydantic model field was renamed in this step, check every test file
that constructs or mocks that model. Pydantic ignores unknown fields by default
— a mock payload using the old field name silently produces the new field's
default value. The test passes, but it never exercises the renamed field. Search
for the old field name in `tests/`:
```
git grep "old_field_name" tests/
```

### E. Contradicting guards
Look for places where the same validation happens twice but in inconsistent
ways — e.g. a manual `if "field" not in data: raise ...` check alongside a
Pydantic model that has `field: str` with no default (which means Pydantic
would raise `ValidationError` on its own). One of the two approaches is
redundant and they may disagree on the error type or message.

### F. Transitional-state integrity
The design doc's **Transitional State** section names what is intentionally
broken until a later step. Confirm that:
- The broken things are actually broken (not accidentally fixed, which would
  imply undocumented scope creep)
- The accepted gaps are truly harmless and won't corrupt DB state or silently
  swallow errors

### G. Commit message conventions
Check that commits follow the project's format: `docs(stepN):`, `test(stepN):`,
`feat(stepN):`, `fix(stepN):`. Misnamed commits make `git log | grep stepN`
unreliable for future review sessions.

## 3. Triage findings

For each finding, decide:
- **Fix before proceeding:** blocks correctness or will corrupt state
- **Fix in follow-up:** quality/clarity issue, safe to defer
- **Not a finding:** false alarm after closer inspection

Keep only real findings. Label each one CT-N (Cleanup Task N), starting at
CT-1.

## 4. Write the findings

For each CT item, write:

```
### CT-N: <short title>

**Decided:** <Fix before proceeding | Fix in follow-up>

<Two to four sentences describing the problem: what it is, why it matters,
where in the code it lives (file, line range or function name).>

**What to do:**
1. <Specific action>
2. <Specific action>
...
```

Be specific enough that a fresh session can implement the fix by reading the
CT item alone, without needing to re-derive the problem.

## 5. Append to the design doc

Open `docs/stepN.md` and populate the **Post-Implementation Cleanup Tasks**
section with the CT items. If the section already has items from a prior review,
append after the last existing item and continue the numbering.

Then commit:

```
git add docs/stepN.md
git commit -m "docs(stepN): adversarial review findings CT-1 through CT-N"
```

If there are no findings, write:
```
No cleanup tasks identified. Implementation matches the spec.
```
and commit that.
