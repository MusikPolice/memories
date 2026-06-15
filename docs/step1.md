# Step 1 — Schema JSON File, Loader, and `GET /api/schema` Endpoint

## Why this step exists

Every subsequent step depends on the schema being a first-class, loadable artefact. The
World Builder (Step 4) validates paths before writing. The Character Evaluator (Step 5)
needs mutability to decide which gate to open. Both evaluator and World Builder prompts
need a rendered schema section (Step 3). The frontend fact tree (Step 8) reads the schema
once at startup to build type-aware controls.

None of that can be built until the schema file exists, a loader can read and cache it,
and the API endpoint exposes it. Step 1 creates all three, so Steps 2–8 have something
stable to depend on.

This step does **not** change any prompts, DB schema, or turn orchestration. It adds one
JSON file, one module, one router, and their tests. The existing system continues to
function identically after this step; nothing calls the new code yet.

**Success criterion:** all tests in `tests/unit/test_schema_loader.py` and
`tests/integration/test_api_schema.py` pass. `GET /api/schema` returns the schema JSON
in a running dev server.

---

## Part A — `src/memories/fact_schema.json`

The schema file mirrors the illustrative example from `docs/plan-v2.md` verbatim. This is
the canonical starting point; it will be fleshed out before implementation of the passes
that use it (Steps 3–5).

The file lives at `src/memories/fact_schema.json`, co-located with the package source so
that `Path(__file__).parent / "fact_schema.json"` resolves correctly from
`schema_loader.py` regardless of the working directory the server is launched from.

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

No application logic encodes or depends on the specific paths or values in this file; the
loader navigates the tree generically. Adding, renaming, or removing paths in this file
between steps requires no code changes — only the loader test that spot-checks the
returned structure would need updating.

---

## Part B — `src/memories/schema_loader.py`

A single module with four public functions. All are synchronous (no async); JSON parsing
and tree traversal are fast CPU-bound operations that do not benefit from async.

### Module-level cache

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_schema: dict[str, Any] | None = None
_SCHEMA_PATH = Path(__file__).parent / "fact_schema.json"
```

The cache is a module-level variable. It is populated on the first call to `load_schema()`
and reused on every subsequent call. It is never invalidated during a server run — schema
evolution happens between deployments, not at runtime.

Unit tests that need an isolated schema pass one explicitly to functions that accept an
optional `schema` argument, so they never touch `_schema` or the file system.

A `_reset_schema_cache()` helper (not part of the public API) sets `_schema = None`. It
exists for test isolation only: `tests/unit/test_schema_loader.py` calls it in a teardown
fixture to prevent one test's `load_schema()` call from warming the cache before another
test that needs a fresh read.

### `load_schema() -> dict[str, Any]`

Reads `fact_schema.json`, parses it, caches the result, and returns it. Subsequent calls
return the cached dict directly without re-reading the file.

```python
def load_schema() -> dict[str, Any]:
    global _schema
    if _schema is None:
        _schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _schema
```

Raises `FileNotFoundError` (from `Path.read_text`) or `json.JSONDecodeError` if the file
is missing or malformed. These are startup-time failures — if they occur in production,
the process should crash rather than serve requests with no schema. No try/except is added.

### `apply_mask(blob: dict[str, Any], schema: dict[str, Any] | None = None) -> dict[str, Any]`

Given a fact blob as stored in `character_facts.facts_json` (a nested dict where leaves
are `{"Value": ...}` objects) and the schema tree, returns a new dict containing only the
paths present in the schema. Paths in the blob that have no matching entry in the schema
are silently dropped.

This is used by `get_facts()` (Step 2) on every read, so that legacy facts and
paths removed from the schema age out naturally without a migration.

```python
def apply_mask(
    blob: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if schema is None:
        schema = load_schema()
    return _mask_node(blob, schema)


def _mask_node(
    blob_node: dict[str, Any],
    schema_node: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, blob_value in blob_node.items():
        if key not in schema_node:
            continue  # path not in schema — silently drop
        schema_child = schema_node[key]
        if "Type" in schema_child:
            # Leaf node — keep the blob value as-is ({"Value": ...})
            result[key] = blob_value
        else:
            # Grouping node — recurse; only include if the result is non-empty
            if isinstance(blob_value, dict):
                masked = _mask_node(blob_value, schema_child)
                if masked:
                    result[key] = masked
    return result
```

**Invariants:**
- The returned dict is always a new object; the input `blob` is never mutated.
- A grouping that has no valid children after masking is dropped entirely (the
  `if masked:` guard). This keeps the blob sparse — empty groupings are meaningless.
- A leaf's `{"Value": ...}` wrapper is passed through unchanged. The mask does not
  validate the value or its type; that is the job of `check_write_permitted` and the
  write handler.
- If `blob_value` for a known grouping path is not a `dict` (malformed blob), that entry
  is silently dropped. In practice this should never occur since the write path always
  produces well-structured blobs.

### `check_write_permitted(path: str, schema: dict[str, Any] | None = None) -> str`

Validates that `path` (a dot-notation string like `"Character.Identity.Name"`) exists in
the schema as a leaf and returns its `Mutability` value (`"Immutable"`, `"Mutable"`, or
`"Fluid"`).

Raises `ValueError` with a descriptive message if the path does not exist in the schema or
if the path resolves to a grouping node rather than a leaf. The `ValueError` message is
designed to be useful as a tool error returned directly to the LLM — it should tell the
model what went wrong and hint at how to correct the call.

```python
def check_write_permitted(
    path: str,
    schema: dict[str, Any] | None = None,
) -> str:
    if schema is None:
        schema = load_schema()
    parts = path.split(".")
    node: Any = schema
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(
                f"Unknown schema path: {path!r}. "
                "You may only write to paths listed in the Fact Schema."
            )
        node = node[part]
    if "Type" not in node:
        raise ValueError(
            f"Path {path!r} is a grouping, not a writable leaf. "
            "Specify a full path to a leaf (e.g. 'Character.Identity.Name')."
        )
    return node["Mutability"]
```

This function is called by `set_fact` and `author_set_facts` server-side handlers in Steps
4 and 5. It is also the foundation for the enum validation that happens inside those
handlers (the handler reads `node["Type"]` and `node.get("Constraint")` from the schema
after `check_write_permitted` confirms the path is valid; the handler is free to re-walk
the tree or cache the node reference — that is a Step 4/5 concern, not Step 1).

### `render_schema_for_prompt(schema: dict[str, Any] | None = None) -> str`

Returns a fully formatted `## Fact Schema` section for injection into the Character
Evaluator and World Builder system prompts. The section enumerates every leaf grouped by
mutability level (Immutable first, then Mutable, then Fluid), with one line per leaf.

For `String` and `Integer` leaves, the suffix is the leaf's `Description`.
For `Enum` leaves, the suffix is the `Constraint` list joined with ` | ` (the description
is not shown in the evaluator prompt — the valid values matter more than the prose
explanation; the description is shown in the character system prompt, which is Step 3).

```
## Fact Schema
You may UPDATE values for paths listed below.
You may NOT create new paths.

IMMUTABLE (cannot be changed once set):
  Character.Identity.Name — The character's given name, typically in the format 'First Last'
  Character.Identity.Age — The character's age in years. ...
  Character.Appearance.Body.Height — The character's height, expressed in the user's preferred unit
  User.Identity.Name — The user's name as they prefer to be addressed by the character

MUTABLE (contextually appropriate; surfaced to user for approval):
  Character.Appearance.Body.Hair-Colour — The character's natural or current hair colour. ...
  Character.Appearance.Outfit.Top — What the character is wearing on their upper body
  Character.Appearance.Outfit.Bottom — What the character is wearing on their lower body
  Character.Appearance.Outfit.Shoes — The character's footwear
  Setting.Temporal.Current-Year — The year in which the narrative is currently set
  Setting.Location.Name — The name of the current location ...
  Setting.Location.Space — Interior | Exterior

FLUID (applied silently; no approval needed):
  Character.State-Of-Mind.Mood — Calm | Anxious | Angry | Sad | Joyful | Neutral | Guarded | Excited
  Character.State-Of-Mind.Energy — Exhausted | Tired | Neutral | Alert | Energised
  Setting.Location.Description — A brief description of the immediate surroundings and atmosphere
```

Implementation sketch:

```python
def render_schema_for_prompt(schema: dict[str, Any] | None = None) -> str:
    if schema is None:
        schema = load_schema()

    leaves = _collect_leaves(schema)

    by_mutability: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "Immutable": [],
        "Mutable": [],
        "Fluid": [],
    }
    for path, leaf in leaves:
        bucket = leaf["Mutability"]
        by_mutability[bucket].append((path, leaf))

    lines = [
        "## Fact Schema",
        "You may UPDATE values for paths listed below.",
        "You may NOT create new paths.",
    ]

    labels = {
        "Immutable": "IMMUTABLE (cannot be changed once set)",
        "Mutable": "MUTABLE (contextually appropriate; surfaced to user for approval)",
        "Fluid": "FLUID (applied silently; no approval needed)",
    }

    for mutability in ("Immutable", "Mutable", "Fluid"):
        group = by_mutability[mutability]
        if not group:
            continue
        lines.append("")
        lines.append(f"{labels[mutability]}:")
        for path, leaf in group:
            if leaf["Type"] == "Enum":
                suffix = " | ".join(leaf["Constraint"])
            else:
                suffix = leaf["Description"]
            lines.append(f"  {path} — {suffix}")

    return "\n".join(lines)


def _collect_leaves(
    node: dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, dict[str, Any]]]:
    leaves: list[tuple[str, dict[str, Any]]] = []
    for key, child in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if "Type" in child:
            leaves.append((path, child))
        else:
            leaves.extend(_collect_leaves(child, path))
    return leaves
```

`_collect_leaves` is a private helper; it does a depth-first walk and returns `(path,
leaf_dict)` pairs in the natural key-insertion order of the schema (Python 3.7+ dicts are
ordered). The evaluator and World Builder prompt sections will therefore always list leaves
in the same order as the schema file — stable, predictable, and easy to compare against
the file in code review.

---

## Part C — `src/memories/routers/schema.py`

A single-endpoint router that returns `fact_schema.json` verbatim as JSON. It requires no
DB access, no auth, and no character scoping. The schema is the same for every character
and changes only with deployments.

```python
"""Schema introspection endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from memories.schema_loader import load_schema

router = APIRouter()


@router.get("/schema")
async def get_schema() -> dict[str, Any]:
    """Return the fact schema as JSON.

    Loaded once at first request and cached in memory for the lifetime of the process.
    """
    return load_schema()
```

The return type annotation `dict[str, Any]` causes FastAPI to serialise the cached dict
directly as JSON. No Pydantic model is needed — the schema is freeform and the type
system cannot describe its recursive structure without defeating the purpose of returning
it verbatim.

### Registration in `main.py`

Add one import and one `include_router` call, following the existing pattern. The prefix
`/api` gives the full path `GET /api/schema`.

```python
# In the imports block:
from memories.routers import schema

# In the include_router block:
app.include_router(schema.router, prefix="/api", tags=["schema"])
```

No lifespan change is needed. The schema is loaded lazily on the first call to
`load_schema()`. This is acceptable because the first call will be triggered by the first
HTTP request after startup — a negligible delay, and one that is always faster than
Ollama warmup.

---

## Tests

### Test schema fixture

Both unit and integration test files reference a common minimal test schema. Unit tests
define it as a module-level constant so functions can be called with `schema=_TEST_SCHEMA`
rather than going through the file-system loader:

```python
_TEST_SCHEMA: dict[str, Any] = {
    "Character": {
        "Identity": {
            "Name": {
                "Type": "String",
                "Mutability": "Immutable",
                "Description": "Character's full name",
            },
            "Age": {
                "Type": "Integer",
                "Mutability": "Immutable",
                "Description": "Age in years",
            },
        },
        "State-Of-Mind": {
            "Mood": {
                "Type": "Enum",
                "Constraint": ["Calm", "Anxious", "Neutral"],
                "Mutability": "Fluid",
                "Description": "Current mood",
            },
        },
        "Appearance": {
            "Outfit": {
                "Top": {
                    "Type": "String",
                    "Mutability": "Mutable",
                    "Description": "Upper-body clothing",
                },
            },
        },
    },
}
```

This covers all three mutability levels, an Enum type, an Integer type, and two levels of
nesting — enough to exercise every branch in `apply_mask`, `check_write_permitted`, and
`render_schema_for_prompt`.

A module-level autouse fixture in `tests/unit/test_schema_loader.py` calls
`schema_loader._reset_schema_cache()` before and after each test to ensure the file-based
`load_schema()` tests don't share state with each other:

```python
@pytest.fixture(autouse=True)
def _reset_cache() -> Any:
    schema_loader._reset_schema_cache()
    yield
    schema_loader._reset_schema_cache()
```

---

### Test file: `tests/unit/test_schema_loader.py`

---

#### `load_schema` tests

**`test_load_schema_returns_dict`**

Calls `load_schema()` with no arguments. Asserts the return value is a `dict`.

---

**`test_load_schema_has_top_level_keys`**

Calls `load_schema()`. Asserts `"Character"`, `"User"`, and `"Setting"` are all present
as top-level keys.

---

**`test_load_schema_caches_result`**

Calls `load_schema()` twice. Asserts both calls return the same object (`is` check, not
just equality). This verifies the module-level cache is populated on the first call and
returned directly on the second.

---

**`test_load_schema_leaf_has_required_fields`**

Navigates to a known leaf: `load_schema()["Character"]["Identity"]["Name"]`. Asserts it
contains `"Type"`, `"Mutability"`, and `"Description"` keys.

---

**`test_load_schema_enum_leaf_has_constraint`**

Navigates to `load_schema()["Character"]["State-Of-Mind"]["Mood"]`. Asserts it contains
`"Constraint"` and that the value is a list.

---

**`test_load_schema_grouping_has_no_type`**

Navigates to `load_schema()["Character"]["Identity"]`. Asserts `"Type"` is **not** a key
(confirming it is a grouping, not a leaf).

---

#### `apply_mask` tests

All tests pass `schema=_TEST_SCHEMA` explicitly.

---

**`test_apply_mask_empty_blob_returns_empty_dict`**

```python
result = apply_mask({}, schema=_TEST_SCHEMA)
assert result == {}
```

---

**`test_apply_mask_known_leaf_passes_through`**

Blob: `{"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}`.

Asserts result equals the input blob unchanged (the known path is preserved).

---

**`test_apply_mask_unknown_top_level_key_dropped`**

Blob: `{"OldCategory": {"some": {"Value": "thing"}}}`.

Asserts result is `{}` (the top-level key has no match in the schema).

---

**`test_apply_mask_unknown_nested_key_dropped`**

Blob:
```python
{
    "Character": {
        "Identity": {
            "Name": {"Value": "Sarah"},
            "InventedField": {"Value": "something"},
        }
    }
}
```

Asserts result is `{"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}`. The known
leaf is kept; the unknown sibling is dropped.

---

**`test_apply_mask_unknown_top_level_drops_silently_alongside_known`**

Blob:
```python
{
    "Character": {"Identity": {"Name": {"Value": "Sarah"}}},
    "LegacyCategory": {"Name": {"Value": "old"}},
}
```

Asserts result contains only `"Character"`. `"LegacyCategory"` is dropped entirely.

---

**`test_apply_mask_grouping_with_all_unknown_children_dropped`**

Blob: `{"Character": {"Identity": {"UnknownA": {"Value": "x"}, "UnknownB": {"Value": "y"}}}}`.

All children under `Character.Identity` are unknown. After masking, `Character.Identity`
is empty; the `if masked:` guard means `Character.Identity` itself is dropped. Asserts
result is `{}`.

---

**`test_apply_mask_multiple_valid_paths_all_kept`**

Blob:
```python
{
    "Character": {
        "Identity": {"Name": {"Value": "Sarah"}, "Age": {"Value": 30}},
        "State-Of-Mind": {"Mood": {"Value": "Calm"}},
    }
}
```

Asserts all three leaves are present in the result unchanged.

---

**`test_apply_mask_preserves_leaf_value_wrapper`**

Blob: `{"Character": {"Identity": {"Name": {"Value": "Sarah", "Extra": "stuff"}}}}`.

Asserts the leaf value is passed through as-is — the mask does not inspect or strip
the contents of the `{"Value": ...}` wrapper. This confirms the mask's sole responsibility
is path validation, not value validation.

---

#### `check_write_permitted` tests

All tests pass `schema=_TEST_SCHEMA` explicitly.

---

**`test_check_write_permitted_immutable_returns_immutable`**

```python
result = check_write_permitted("Character.Identity.Name", schema=_TEST_SCHEMA)
assert result == "Immutable"
```

---

**`test_check_write_permitted_mutable_returns_mutable`**

```python
result = check_write_permitted("Character.Appearance.Outfit.Top", schema=_TEST_SCHEMA)
assert result == "Mutable"
```

---

**`test_check_write_permitted_fluid_returns_fluid`**

```python
result = check_write_permitted("Character.State-Of-Mind.Mood", schema=_TEST_SCHEMA)
assert result == "Fluid"
```

---

**`test_check_write_permitted_integer_leaf_returns_mutability`**

```python
result = check_write_permitted("Character.Identity.Age", schema=_TEST_SCHEMA)
assert result == "Immutable"
```

Confirms Integer leaves are handled identically to String leaves.

---

**`test_check_write_permitted_unknown_path_raises_value_error`**

```python
with pytest.raises(ValueError):
    check_write_permitted("Character.Identity.Unknown", schema=_TEST_SCHEMA)
```

---

**`test_check_write_permitted_unknown_top_level_raises_value_error`**

```python
with pytest.raises(ValueError):
    check_write_permitted("Setting.Location.Name", schema=_TEST_SCHEMA)
```

`Setting` is not in `_TEST_SCHEMA`. Confirms the error fires on the very first missing
segment, not only on missing leaf keys.

---

**`test_check_write_permitted_grouping_path_raises_value_error`**

```python
with pytest.raises(ValueError):
    check_write_permitted("Character.Identity", schema=_TEST_SCHEMA)
```

`Character.Identity` is a grouping (no `"Type"` key). Asserts `ValueError` is raised
with a message explaining that a full leaf path is required.

---

**`test_check_write_permitted_error_message_contains_path`**

```python
with pytest.raises(ValueError, match="Character.Identity.Unknown"):
    check_write_permitted("Character.Identity.Unknown", schema=_TEST_SCHEMA)
```

Asserts the invalid path appears in the error message string. This matters because the
message is returned verbatim to the LLM as the tool result — it must be informative.

---

**`test_check_write_permitted_grouping_error_message_is_helpful`**

```python
with pytest.raises(ValueError, match="grouping"):
    check_write_permitted("Character.Identity", schema=_TEST_SCHEMA)
```

Asserts the word `"grouping"` (or a comparable term) appears in the error message so the
LLM understands it has provided an incomplete path.

---

#### `render_schema_for_prompt` tests

All tests pass `schema=_TEST_SCHEMA` explicitly.

---

**`test_render_schema_for_prompt_contains_header`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "## Fact Schema" in result
```

---

**`test_render_schema_for_prompt_contains_cannot_create_instruction`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "NOT create new paths" in result
```

---

**`test_render_schema_for_prompt_has_immutable_section`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "IMMUTABLE" in result
```

---

**`test_render_schema_for_prompt_has_mutable_section`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "MUTABLE" in result
```

---

**`test_render_schema_for_prompt_has_fluid_section`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "FLUID" in result
```

---

**`test_render_schema_for_prompt_lists_immutable_leaf_paths`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "Character.Identity.Name" in result
assert "Character.Identity.Age" in result
```

---

**`test_render_schema_for_prompt_lists_mutable_leaf_paths`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "Character.Appearance.Outfit.Top" in result
```

---

**`test_render_schema_for_prompt_lists_fluid_leaf_paths`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
assert "Character.State-Of-Mind.Mood" in result
```

---

**`test_render_schema_for_prompt_enum_shows_constraint_values`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
# Mood is an Enum with Constraint ["Calm", "Anxious", "Neutral"]
assert "Calm | Anxious | Neutral" in result
```

---

**`test_render_schema_for_prompt_string_shows_description`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
# Name is a String — its Description should appear
assert "Character's full name" in result
```

---

**`test_render_schema_for_prompt_omits_grouping_nodes`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
# "Character.Identity" is a grouping, not a leaf — it must not appear as its own line
lines = result.splitlines()
assert not any(line.strip().startswith("Character.Identity —") for line in lines)
```

---

**`test_render_schema_for_prompt_sections_in_correct_order`**

```python
result = render_schema_for_prompt(schema=_TEST_SCHEMA)
immutable_pos = result.index("IMMUTABLE")
mutable_pos = result.index("MUTABLE")
fluid_pos = result.index("FLUID")
assert immutable_pos < mutable_pos < fluid_pos
```

Immutable section appears before Mutable, which appears before Fluid. This matches the
evaluator prompt sketch in `docs/plan-v2.md` and the ordering expectation for Steps 3–5.

---

**`test_render_schema_for_prompt_empty_section_omitted`**

Create a minimal schema with only Immutable leaves (no Mutable, no Fluid). Asserts the
output does not contain a `MUTABLE` or `FLUID` header. This tests the `if not group:
continue` guard.

```python
immutable_only_schema: dict[str, Any] = {
    "Character": {
        "Identity": {
            "Name": {
                "Type": "String",
                "Mutability": "Immutable",
                "Description": "Name",
            }
        }
    }
}
result = render_schema_for_prompt(schema=immutable_only_schema)
assert "MUTABLE" not in result
assert "FLUID" not in result
```

---

### Test file: `tests/integration/test_api_schema.py`

Integration tests use the standard `client` fixture from `tests/integration/conftest.py`.
No Ollama mocking is needed — the schema endpoint makes no Ollama calls.

The tests exercise the *real* `fact_schema.json`, not `_TEST_SCHEMA`. This means they
double as a smoke test that the schema file is well-formed and contains the expected
structure. If someone edits `fact_schema.json` and breaks a known path, these tests
will catch it.

---

**`test_get_schema_returns_200`**

```python
async def test_get_schema_returns_200(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert response.status_code == 200
```

---

**`test_get_schema_content_type_is_json`**

```python
async def test_get_schema_content_type_is_json(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "application/json" in response.headers["content-type"]
```

---

**`test_get_schema_has_character_key`**

```python
async def test_get_schema_has_character_key(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "Character" in response.json()
```

---

**`test_get_schema_has_user_key`**

```python
async def test_get_schema_has_user_key(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "User" in response.json()
```

---

**`test_get_schema_has_setting_key`**

```python
async def test_get_schema_has_setting_key(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "Setting" in response.json()
```

---

**`test_get_schema_character_identity_name_is_leaf`**

```python
async def test_get_schema_character_identity_name_is_leaf(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    leaf = data["Character"]["Identity"]["Name"]
    assert "Type" in leaf
    assert "Mutability" in leaf
    assert "Description" in leaf
```

Spot-check that a known path navigates to a leaf with the required fields.

---

**`test_get_schema_leaf_type_is_string`**

```python
async def test_get_schema_leaf_type_is_string(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["Identity"]["Name"]["Type"] == "String"
```

---

**`test_get_schema_leaf_mutability_is_immutable`**

```python
async def test_get_schema_leaf_mutability_is_immutable(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["Identity"]["Name"]["Mutability"] == "Immutable"
```

---

**`test_get_schema_enum_leaf_has_constraint_list`**

```python
async def test_get_schema_enum_leaf_has_constraint_list(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    mood = data["Character"]["State-Of-Mind"]["Mood"]
    assert mood["Type"] == "Enum"
    assert isinstance(mood["Constraint"], list)
    assert len(mood["Constraint"]) > 0
```

---

**`test_get_schema_fluid_leaf_mutability`**

```python
async def test_get_schema_fluid_leaf_mutability(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["State-Of-Mind"]["Mood"]["Mutability"] == "Fluid"
```

---

**`test_get_schema_mutable_leaf_mutability`**

```python
async def test_get_schema_mutable_leaf_mutability(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["Appearance"]["Outfit"]["Top"]["Mutability"] == "Mutable"
```

---

**`test_get_schema_no_db_access_required`**

This test deliberately calls `GET /api/schema` from a minimal `AsyncClient` that has no
DB dependency override. Since `schema.router` calls only `load_schema()` (no `Depends(get_db)`),
it should succeed even if the DB fixture is not available.

```python
async def test_get_schema_no_db_access_required() -> None:
    # Use a fresh client with no DB override — schema endpoint must not touch the DB
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/schema")
    assert response.status_code == 200
```

This confirms the endpoint's independence — it will remain fast and always-available even
if the DB is unhealthy. This is a safety net for future refactors that might accidentally
add a `db` dependency to the schema router.

---

## Files changed by this step

| Action | File | Notes |
|---|---|---|
| Add | `src/memories/fact_schema.json` | The schema tree; edit freely between steps |
| Add | `src/memories/schema_loader.py` | `load_schema`, `apply_mask`, `check_write_permitted`, `render_schema_for_prompt` |
| Add | `src/memories/routers/schema.py` | `GET /api/schema` |
| Modify | `src/memories/main.py` | Import and register `schema.router` at prefix `/api` |
| Add | `tests/unit/test_schema_loader.py` | All unit tests above |
| Add | `tests/integration/test_api_schema.py` | All integration tests above |

No DB changes. No prompt changes. No frontend changes. No changes to any existing router.
The existing system is untouched — this step adds new artefacts only.

---

## Dependency order relative to other steps

Step 1 is a prerequisite for:
- **Step 2** (`character_facts` table and repositories) — `get_facts()` calls `apply_mask()`
- **Step 3** (prompt changes) — evaluator and World Builder prompts call
  `render_schema_for_prompt()`; `build_system_prompt()` navigates the schema to render
  populated and unpopulated leaves
- **Step 4** (World Builder) — `author_set_facts` handler calls `check_write_permitted()`
  for path validation
- **Step 5** (Character Evaluator tool loop) — `set_fact` handler calls
  `check_write_permitted()` to get mutability
- **Step 8** (UI) — the frontend fetches `GET /api/schema` at startup

Steps 1, 2, and 3 can be worked on concurrently with Steps 0b (already complete) since
none of them involve SSE or gate coordination. Step 1 should be the first of Steps 1–3
to be merged, since 2 and 3 import from `schema_loader`.
