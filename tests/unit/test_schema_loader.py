"""Unit tests for schema_loader.py."""

from __future__ import annotations

from typing import Any

import pytest

from memories import schema_loader
from memories.schema_loader import apply_mask, check_write_permitted, render_schema_for_prompt

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


@pytest.fixture(autouse=True)
def _reset_cache() -> Any:
    schema_loader._reset_schema_cache()
    yield
    schema_loader._reset_schema_cache()


# ---------------------------------------------------------------------------
# load_schema tests
# ---------------------------------------------------------------------------


def test_load_schema_returns_dict() -> None:
    result = schema_loader.load_schema()
    assert isinstance(result, dict)


def test_load_schema_has_top_level_keys() -> None:
    result = schema_loader.load_schema()
    assert "Character" in result
    assert "User" in result
    assert "Setting" in result


def test_load_schema_caches_result() -> None:
    first = schema_loader.load_schema()
    second = schema_loader.load_schema()
    assert first is second


def test_load_schema_leaf_has_required_fields() -> None:
    leaf = schema_loader.load_schema()["Character"]["Identity"]["Name"]["First"]
    assert "Type" in leaf
    assert "Mutability" in leaf
    assert "Description" in leaf


def test_load_schema_enum_leaf_has_constraint() -> None:
    leaf = schema_loader.load_schema()["Character"]["State-Of-Mind"]["Mood"]
    assert "Constraint" in leaf
    assert isinstance(leaf["Constraint"], list)


def test_load_schema_grouping_has_no_type() -> None:
    grouping = schema_loader.load_schema()["Character"]["Identity"]
    assert "Type" not in grouping


# ---------------------------------------------------------------------------
# apply_mask tests
# ---------------------------------------------------------------------------


def test_apply_mask_empty_blob_returns_empty_dict() -> None:
    result = apply_mask({}, schema=_TEST_SCHEMA)
    assert result == {}


def test_apply_mask_known_leaf_passes_through() -> None:
    blob = {"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}
    result = apply_mask(blob, schema=_TEST_SCHEMA)
    assert result == blob


def test_apply_mask_unknown_top_level_key_dropped() -> None:
    blob = {"OldCategory": {"some": {"Value": "thing"}}}
    result = apply_mask(blob, schema=_TEST_SCHEMA)
    assert result == {}


def test_apply_mask_unknown_nested_key_dropped() -> None:
    blob = {
        "Character": {
            "Identity": {
                "Name": {"Value": "Sarah"},
                "InventedField": {"Value": "something"},
            }
        }
    }
    result = apply_mask(blob, schema=_TEST_SCHEMA)
    assert result == {"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}


def test_apply_mask_unknown_top_level_drops_silently_alongside_known() -> None:
    blob = {
        "Character": {"Identity": {"Name": {"Value": "Sarah"}}},
        "LegacyCategory": {"Name": {"Value": "old"}},
    }
    result = apply_mask(blob, schema=_TEST_SCHEMA)
    assert "Character" in result
    assert "LegacyCategory" not in result


def test_apply_mask_grouping_with_all_unknown_children_dropped() -> None:
    blob = {"Character": {"Identity": {"UnknownA": {"Value": "x"}, "UnknownB": {"Value": "y"}}}}
    result = apply_mask(blob, schema=_TEST_SCHEMA)
    assert result == {}


def test_apply_mask_multiple_valid_paths_all_kept() -> None:
    blob = {
        "Character": {
            "Identity": {"Name": {"Value": "Sarah"}, "Age": {"Value": 30}},
            "State-Of-Mind": {"Mood": {"Value": "Calm"}},
        }
    }
    result = apply_mask(blob, schema=_TEST_SCHEMA)
    assert result["Character"]["Identity"]["Name"] == {"Value": "Sarah"}
    assert result["Character"]["Identity"]["Age"] == {"Value": 30}
    assert result["Character"]["State-Of-Mind"]["Mood"] == {"Value": "Calm"}


def test_apply_mask_preserves_leaf_value_wrapper() -> None:
    blob = {"Character": {"Identity": {"Name": {"Value": "Sarah", "Extra": "stuff"}}}}
    result = apply_mask(blob, schema=_TEST_SCHEMA)
    assert result["Character"]["Identity"]["Name"] == {"Value": "Sarah", "Extra": "stuff"}


# ---------------------------------------------------------------------------
# check_write_permitted tests
# ---------------------------------------------------------------------------


def test_check_write_permitted_immutable_returns_immutable() -> None:
    result = check_write_permitted("Character.Identity.Name", schema=_TEST_SCHEMA)
    assert result == "Immutable"


def test_check_write_permitted_mutable_returns_mutable() -> None:
    result = check_write_permitted("Character.Appearance.Outfit.Top", schema=_TEST_SCHEMA)
    assert result == "Mutable"


def test_check_write_permitted_fluid_returns_fluid() -> None:
    result = check_write_permitted("Character.State-Of-Mind.Mood", schema=_TEST_SCHEMA)
    assert result == "Fluid"


def test_check_write_permitted_integer_leaf_returns_mutability() -> None:
    result = check_write_permitted("Character.Identity.Age", schema=_TEST_SCHEMA)
    assert result == "Immutable"


def test_check_write_permitted_unknown_path_raises_value_error() -> None:
    with pytest.raises(ValueError):
        check_write_permitted("Character.Identity.Unknown", schema=_TEST_SCHEMA)


def test_check_write_permitted_unknown_top_level_raises_value_error() -> None:
    with pytest.raises(ValueError):
        check_write_permitted("Setting.Location.Name", schema=_TEST_SCHEMA)


def test_check_write_permitted_grouping_path_raises_value_error() -> None:
    with pytest.raises(ValueError):
        check_write_permitted("Character.Identity", schema=_TEST_SCHEMA)


def test_check_write_permitted_error_message_contains_path() -> None:
    with pytest.raises(ValueError, match=r"Character\.Identity\.Unknown"):
        check_write_permitted("Character.Identity.Unknown", schema=_TEST_SCHEMA)


def test_check_write_permitted_grouping_error_message_is_helpful() -> None:
    with pytest.raises(ValueError, match="grouping"):
        check_write_permitted("Character.Identity", schema=_TEST_SCHEMA)


# ---------------------------------------------------------------------------
# render_schema_for_prompt tests
# ---------------------------------------------------------------------------


def test_render_schema_for_prompt_contains_header() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "## Fact Schema" in result


def test_render_schema_for_prompt_contains_cannot_create_instruction() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "NOT create new paths" in result


def test_render_schema_for_prompt_has_immutable_section() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "IMMUTABLE" in result


def test_render_schema_for_prompt_has_mutable_section() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "MUTABLE" in result


def test_render_schema_for_prompt_has_fluid_section() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "FLUID" in result


def test_render_schema_for_prompt_lists_immutable_leaf_paths() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "Character.Identity.Name" in result
    assert "Character.Identity.Age" in result


def test_render_schema_for_prompt_lists_mutable_leaf_paths() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "Character.Appearance.Outfit.Top" in result


def test_render_schema_for_prompt_lists_fluid_leaf_paths() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "Character.State-Of-Mind.Mood" in result


def test_render_schema_for_prompt_enum_shows_constraint_values() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "Calm | Anxious | Neutral" in result


def test_render_schema_for_prompt_string_shows_description() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    assert "Character's full name" in result


def test_render_schema_for_prompt_omits_grouping_nodes() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    lines = result.splitlines()
    assert not any(line.strip().startswith("Character.Identity —") for line in lines)


def test_render_schema_for_prompt_sections_in_correct_order() -> None:
    result = render_schema_for_prompt(schema=_TEST_SCHEMA)
    immutable_pos = result.index("IMMUTABLE")
    mutable_pos = result.index("MUTABLE")
    fluid_pos = result.index("FLUID")
    assert immutable_pos < mutable_pos < fluid_pos


def test_render_schema_for_prompt_empty_section_omitted() -> None:
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
    lines = result.splitlines()
    assert not any(line.strip().startswith("MUTABLE") for line in lines)
    assert not any(line.strip().startswith("FLUID") for line in lines)
