"""Unit tests for memories.services.prompt_builder.build_system_prompt."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memories.models import Character, Experience, Inference
from memories.services.prompt_builder import build_system_prompt

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 4, 12, 0, 0)

_CHARACTER = Character(
    id=1,
    name="Elara Voss",
    modelfile_base="qwen3:7b",
    created_at=_NOW,
)

# A minimal populated blob using real schema paths
_FACTS_BLOB: dict[str, Any] = {
    "Character": {
        "Identity": {"Name": {"Value": "Elara Voss"}},
    }
}


# ---------------------------------------------------------------------------
# Retained from before: character name
# ---------------------------------------------------------------------------


def test_character_name_appears_in_prompt() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "Elara Voss" in prompt


# ---------------------------------------------------------------------------
# Step 3 additions — schema-tree rendering
# ---------------------------------------------------------------------------


def test_world_state_header_present() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "## World State" in prompt


def test_mutability_preamble_present() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    prompt_upper = prompt.upper()
    assert "IMMUTABLE" in prompt_upper
    assert "MUTABLE" in prompt_upper
    assert "FLUID" in prompt_upper


def test_populated_leaf_shows_value() -> None:
    blob: dict[str, Any] = {"Character": {"Identity": {"Name": {"Value": "Sarah"}}}}
    prompt = build_system_prompt(_CHARACTER, blob)
    assert "Value: Sarah" in prompt


def test_unpopulated_leaf_shows_not_set() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "Value: (not set)" in prompt


def test_enum_leaf_always_shows_valid_values() -> None:
    # Character.State-Of-Mind.Mood is Enum — valid values must appear regardless of blob state
    prompt = build_system_prompt(_CHARACTER, {})
    assert "Valid values:" in prompt
    assert "Calm" in prompt


def test_enum_leaf_shows_valid_values_when_set() -> None:
    blob: dict[str, Any] = {"Character": {"State-Of-Mind": {"Mood": {"Value": "Anxious"}}}}
    prompt = build_system_prompt(_CHARACTER, blob)
    assert "Value: Anxious" in prompt
    assert "Valid values:" in prompt


def test_immutable_mutability_tag() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "[IMMUTABLE]" in prompt


def test_mutable_mutability_tag() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "[MUTABLE]" in prompt


def test_fluid_mutability_tag() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "[FLUID]" in prompt


def test_full_dot_path_in_leaf_label() -> None:
    # A leaf several levels deep: Character.Appearance.Hair.Colour
    prompt = build_system_prompt(_CHARACTER, {})
    assert "Character.Appearance.Hair.Colour" in prompt


def test_character_section_header() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "### Character" in prompt


def test_user_section_header() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "### User" in prompt


def test_setting_section_header() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    assert "### Setting" in prompt


def test_character_section_precedes_user_section() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    char_idx = prompt.index("### Character")
    user_idx = prompt.index("### User")
    assert char_idx < user_idx


def test_user_section_precedes_setting_section() -> None:
    prompt = build_system_prompt(_CHARACTER, {})
    user_idx = prompt.index("### User")
    setting_idx = prompt.index("### Setting")
    assert user_idx < setting_idx


def test_inferences_section_follows_schema_sections() -> None:
    inferences = [
        Inference(
            id=1,
            character_id=1,
            statement="Elara was born in 1992",
            derivation="age=34, current_year=2026",
            source_fact_ids=[],
            source_inference_ids=[],
            depth=1,
            inference_type="logical",
            status="active",
            created_at=_NOW,
        )
    ]
    prompt = build_system_prompt(_CHARACTER, {}, inferences=inferences)
    setting_idx = prompt.index("### Setting")
    inf_header_idx = prompt.index("## Your Inferences")
    assert setting_idx < inf_header_idx


# ---------------------------------------------------------------------------
# Phase 3 additions — inferences parameter (retained, signature updated)
# ---------------------------------------------------------------------------

_INFERENCES = [
    Inference(
        id=1,
        character_id=1,
        statement="Elara was born in 1992",
        derivation="age=34, current_year=2026",
        source_fact_ids=[1],
        source_inference_ids=[],
        depth=1,
        inference_type="logical",
        status="active",
        created_at=_NOW,
    ),
    Inference(
        id=2,
        character_id=1,
        statement="Elara has strong map-reading skills",
        derivation="occupation=cartographer",
        source_fact_ids=[2],
        source_inference_ids=[],
        depth=1,
        inference_type="probabilistic",
        status="active",
        created_at=_NOW,
    ),
    Inference(
        id=3,
        character_id=1,
        statement="Elara likely speaks Norwegian",
        derivation="birthplace=Oslo",
        source_fact_ids=[3],
        source_inference_ids=[],
        depth=1,
        inference_type="probabilistic",
        status="active",
        created_at=_NOW,
    ),
]


def test_inferences_section_included_when_inferences_present() -> None:
    prompt = build_system_prompt(_CHARACTER, {}, inferences=_INFERENCES[:1])
    assert "## Your Inferences" in prompt


def test_inference_statement_appears_verbatim() -> None:
    prompt = build_system_prompt(_CHARACTER, {}, inferences=_INFERENCES[:1])
    assert "Elara was born in 1992" in prompt


def test_inference_derivation_appears_in_prompt() -> None:
    prompt = build_system_prompt(_CHARACTER, {}, inferences=_INFERENCES[:1])
    assert "age=34, current_year=2026" in prompt


def test_inferences_section_absent_when_no_inferences() -> None:
    prompt = build_system_prompt(_CHARACTER, {}, inferences=[])
    assert "## Your Inferences" not in prompt


def test_inferences_section_absent_when_inferences_is_none() -> None:
    prompt = build_system_prompt(_CHARACTER, {}, inferences=None)
    assert "## Your Inferences" not in prompt


def test_multiple_inferences_all_appear() -> None:
    prompt = build_system_prompt(_CHARACTER, {}, inferences=_INFERENCES)
    assert "Elara was born in 1992" in prompt
    assert "Elara has strong map-reading skills" in prompt
    assert "Elara likely speaks Norwegian" in prompt


# ---------------------------------------------------------------------------
# Phase 5 additions — experiences parameter (retained, signature updated)
# ---------------------------------------------------------------------------

_P5_NOW = datetime(2026, 6, 7, 0, 0, 0)
_P5_CHAR = Character(id=1, name="Elara Voss", modelfile_base="qwen3:7b", created_at=_P5_NOW)


def _exp(
    exp_id: int,
    statement: str,
    source: str = "told_by_user",
) -> Experience:
    return Experience(
        id=exp_id,
        character_id=1,
        session_id=1,
        statement=statement,
        source=source,  # type: ignore[arg-type]
        approved_at=_P5_NOW,
        created_at=_P5_NOW,
    )


_P5_EXPERIENCES = [
    _exp(1, "We are currently located in Chicago", "told_by_user"),
    _exp(2, "Jon seemed uncomfortable when asked about his family", "observed"),
]


def test_experiences_section_included_when_active() -> None:
    prompt = build_system_prompt(_P5_CHAR, {}, experiences=_P5_EXPERIENCES)
    assert "## Your Experiences" in prompt


def test_experience_statement_appears_in_prompt() -> None:
    prompt = build_system_prompt(_P5_CHAR, {}, experiences=_P5_EXPERIENCES)
    assert "We are currently located in Chicago" in prompt


def test_experience_source_told_by_user_labelled() -> None:
    exp = _exp(1, "Chicago statement", "told_by_user")
    prompt = build_system_prompt(_P5_CHAR, {}, experiences=[exp])
    assert "told by user" in prompt.lower()


def test_experience_source_observed_labelled() -> None:
    exp = _exp(2, "Observed statement", "observed")
    prompt = build_system_prompt(_P5_CHAR, {}, experiences=[exp])
    assert "observed" in prompt.lower()


def test_multiple_experiences_all_appear() -> None:
    exps = [
        _exp(1, "Statement one"),
        _exp(2, "Statement two"),
        _exp(3, "Statement three"),
    ]
    prompt = build_system_prompt(_P5_CHAR, {}, experiences=exps)
    assert "Statement one" in prompt
    assert "Statement two" in prompt
    assert "Statement three" in prompt


def test_experiences_section_absent_when_empty_list() -> None:
    prompt = build_system_prompt(_P5_CHAR, {}, experiences=[])
    assert "## Your Experiences" not in prompt


def test_experiences_section_absent_when_none() -> None:
    prompt = build_system_prompt(_P5_CHAR, {}, experiences=None)
    assert "## Your Experiences" not in prompt


def test_integer_leaf_renders_value() -> None:
    blob: dict[str, Any] = {"Setting": {"Temporal": {"Current-Year": {"Value": 2026}}}}
    prompt = build_system_prompt(_CHARACTER, blob)
    assert "Value: 2026" in prompt


def test_integer_zero_renders_as_zero_not_not_set() -> None:
    # Falsy zero must not be mistaken for "unset" (guarded by `is None`, not `not value`)
    blob: dict[str, Any] = {"Character": {"Identity": {"Age": {"Value": 0}}}}
    prompt = build_system_prompt(_CHARACTER, blob)
    assert "Value: 0" in prompt


def test_experiences_section_follows_inferences() -> None:
    inferences = [
        Inference(
            id=1,
            character_id=1,
            statement="Some inference",
            derivation="from fact",
            source_fact_ids=[],
            source_inference_ids=[],
            depth=1,
            inference_type="logical",
            status="active",
            created_at=_P5_NOW,
        )
    ]
    prompt = build_system_prompt(_P5_CHAR, {}, inferences=inferences, experiences=_P5_EXPERIENCES)
    inf_idx = prompt.find("Some inference")
    exp_idx = prompt.find("We are currently located in Chicago")
    assert inf_idx != -1 and exp_idx != -1
    assert inf_idx < exp_idx, "Experiences section should appear after inferences"
