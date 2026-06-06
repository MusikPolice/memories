"""Unit tests for memories.services.prompt_builder.build_system_prompt."""

from datetime import datetime

from memories.models import Character, Fact, Inference
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

_FACTS = [
    Fact(id=1, character_id=1, key="age", value="34", created_at=_NOW),
    Fact(id=2, character_id=1, key="occupation", value="cartographer", created_at=_NOW),
    Fact(id=3, character_id=1, key="birthplace", value="Oslo", created_at=_NOW),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_character_name_appears_in_prompt() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS)
    assert "Elara Voss" in prompt


def test_all_facts_injected_as_key_value_pairs() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS)
    assert "age: 34" in prompt
    assert "occupation: cartographer" in prompt
    assert "birthplace: Oslo" in prompt


def test_fact_order_preserved() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS)
    age_pos = prompt.index("age: 34")
    occ_pos = prompt.index("occupation: cartographer")
    birth_pos = prompt.index("birthplace: Oslo")
    assert age_pos < occ_pos < birth_pos


def test_no_facts_yields_no_invention_instruction() -> None:
    prompt = build_system_prompt(_CHARACTER, [])
    # When there are no facts the prompt must explicitly forbid invention.
    assert "not invent" in prompt.lower() or "do not invent" in prompt.lower()


def test_facts_section_header_present_regardless_of_fact_count() -> None:
    """The '## Your Facts' header appears whether or not facts exist."""
    with_facts = build_system_prompt(_CHARACTER, _FACTS)
    without_facts = build_system_prompt(_CHARACTER, [])
    assert "## Your Facts" in with_facts
    assert "## Your Facts" in without_facts


# ---------------------------------------------------------------------------
# Phase 3 additions — inferences parameter
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 4, 12, 0, 0)

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
    prompt = build_system_prompt(_CHARACTER, _FACTS, inferences=_INFERENCES[:1])
    assert "## Your Inferences" in prompt


def test_inference_statement_appears_verbatim() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS, inferences=_INFERENCES[:1])
    assert "Elara was born in 1992" in prompt


def test_inference_derivation_appears_in_prompt() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS, inferences=_INFERENCES[:1])
    assert "age=34, current_year=2026" in prompt


def test_inferences_section_absent_when_no_inferences() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS, inferences=[])
    assert "## Your Inferences" not in prompt


def test_inferences_section_absent_when_inferences_is_none() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS, inferences=None)
    assert "## Your Inferences" not in prompt


def test_multiple_inferences_all_appear() -> None:
    prompt = build_system_prompt(_CHARACTER, _FACTS, inferences=_INFERENCES)
    assert "Elara was born in 1992" in prompt
    assert "Elara has strong map-reading skills" in prompt
    assert "Elara likely speaks Norwegian" in prompt


# ---------------------------------------------------------------------------
# Phase 4 additions — category sections and mutability annotations
# ---------------------------------------------------------------------------

_P4_NOW = datetime(2026, 6, 6, 0, 0, 0)
_P4_CHAR = Character(id=1, name="Elara Voss", modelfile_base="qwen3:7b", created_at=_P4_NOW)


def _f(
    fid: int,
    key: str,
    value: str,
    category: str = "character",
    mutability: str = "immutable",
) -> Fact:
    return Fact(
        id=fid,
        character_id=1,
        key=key,
        value=value,
        category=category,  # type: ignore[arg-type]
        mutability=mutability,  # type: ignore[arg-type]
        created_at=_P4_NOW,
    )


def test_character_facts_appear_under_character_section() -> None:
    facts = [_f(1, "occupation", "surgeon", category="character")]
    prompt = build_system_prompt(_P4_CHAR, facts)
    fact_idx = prompt.find("occupation: surgeon")
    assert fact_idx != -1, "Fact not found in prompt"
    before_fact = prompt[:fact_idx]
    assert (
        "Character" in before_fact
    ), "Expected a section header containing 'Character' before the fact"


def test_user_facts_appear_under_user_section() -> None:
    facts = [_f(1, "name", "Jon", category="user")]
    prompt = build_system_prompt(_P4_CHAR, facts)
    fact_idx = prompt.find("name: Jon")
    assert fact_idx != -1, "Fact not found in prompt"
    before_fact = prompt[:fact_idx]
    assert "User" in before_fact, "Expected a section header containing 'User' before the fact"


def test_setting_facts_appear_under_setting_section() -> None:
    facts = [_f(1, "time_of_day", "evening", category="setting")]
    prompt = build_system_prompt(_P4_CHAR, facts)
    fact_idx = prompt.find("time_of_day: evening")
    assert fact_idx != -1, "Fact not found in prompt"
    before_fact = prompt[:fact_idx]
    assert (
        "Setting" in before_fact
    ), "Expected a section header containing 'Setting' before the fact"


def test_section_omitted_when_no_facts_in_category() -> None:
    facts = [_f(1, "occupation", "surgeon", category="character")]
    prompt = build_system_prompt(_P4_CHAR, facts)
    lines = prompt.split("\n")
    user_headers = [ln for ln in lines if ln.startswith("##") and "User" in ln]
    assert len(user_headers) == 0, "User section should be omitted when no user facts exist"


def test_all_three_category_sections_rendered() -> None:
    facts = [
        _f(1, "user_name", "Jon", category="user"),
        _f(2, "occupation", "surgeon", category="character"),
        _f(3, "location", "office", category="setting"),
    ]
    prompt = build_system_prompt(_P4_CHAR, facts)
    lines = prompt.split("\n")
    section_headers = [ln for ln in lines if ln.startswith("##")]
    assert any("User" in h for h in section_headers), "Expected a User section header"
    assert any("Character" in h for h in section_headers), "Expected a Character section header"
    assert any("Setting" in h for h in section_headers), "Expected a Setting section header"


def test_section_order_is_user_then_character_then_setting() -> None:
    facts = [
        _f(1, "user_name", "Jon", category="user"),
        _f(2, "occupation", "surgeon", category="character"),
        _f(3, "location", "office", category="setting"),
    ]
    prompt = build_system_prompt(_P4_CHAR, facts)
    user_idx = prompt.find("user_name: Jon")
    char_idx = prompt.find("occupation: surgeon")
    setting_idx = prompt.find("location: office")
    assert user_idx < char_idx, "User facts should appear before Character facts"
    assert char_idx < setting_idx, "Character facts should appear before Setting facts"


def test_facts_within_category_in_id_order() -> None:
    # Higher id listed first in input — prompt should sort by id ascending
    facts = [
        _f(3, "mood", "cheerful", category="character"),
        _f(1, "age", "33", category="character"),
    ]
    prompt = build_system_prompt(_P4_CHAR, facts)
    age_idx = prompt.find("age: 33")
    mood_idx = prompt.find("mood: cheerful")
    assert age_idx != -1 and mood_idx != -1
    assert age_idx < mood_idx, "Facts should appear in ascending id order within a category"


def test_immutable_fact_has_no_mutability_annotation() -> None:
    facts = [_f(1, "age", "33", category="character", mutability="immutable")]
    prompt = build_system_prompt(_P4_CHAR, facts)
    line = next((ln for ln in prompt.split("\n") if "age: 33" in ln), "")
    assert "[" not in line, "Immutable facts should have no annotation brackets"


def test_low_mutability_fact_has_annotation() -> None:
    facts = [_f(1, "clothing", "dark coat", category="character", mutability="low")]
    prompt = build_system_prompt(_P4_CHAR, facts)
    line = next((ln for ln in prompt.split("\n") if "clothing: dark coat" in ln), "")
    assert "[low-mutability" in line, "Low-mutability fact should have a [low-mutability annotation"


def test_high_mutability_fact_has_annotation() -> None:
    facts = [_f(1, "mood", "cheerful", category="character", mutability="high")]
    prompt = build_system_prompt(_P4_CHAR, facts)
    line = next((ln for ln in prompt.split("\n") if "mood: cheerful" in ln), "")
    assert "[fluid" in line, "High-mutability fact should have a [fluid annotation"


def test_no_facts_message_when_all_categories_empty() -> None:
    prompt = build_system_prompt(_P4_CHAR, [])
    lines = prompt.split("\n")
    # No category-specific section headers should appear
    category_headers = [
        ln for ln in lines if ln.startswith("##") and any(c in ln for c in ("User", "Setting"))
    ]
    assert len(category_headers) == 0


def test_inferences_section_follows_all_fact_sections() -> None:
    facts = [
        _f(1, "user_name", "Jon", category="user"),
        _f(2, "occupation", "surgeon", category="character"),
        _f(3, "location", "office", category="setting"),
    ]
    inferences = [
        Inference(
            id=1,
            character_id=1,
            statement="Elara was born in 1993",
            derivation="age=33, year=2026",
            source_fact_ids=[],
            source_inference_ids=[],
            depth=1,
            inference_type="logical",
            status="active",
            created_at=_P4_NOW,
        )
    ]
    prompt = build_system_prompt(_P4_CHAR, facts, inferences=inferences)
    user_idx = prompt.find("user_name: Jon")
    char_idx = prompt.find("occupation: surgeon")
    setting_idx = prompt.find("location: office")
    inf_header_idx = prompt.find("## Your Inferences")
    assert inf_header_idx != -1, "Expected ## Your Inferences header"
    assert user_idx < inf_header_idx, "Inferences section should follow User facts"
    assert char_idx < inf_header_idx, "Inferences section should follow Character facts"
    assert setting_idx < inf_header_idx, "Inferences section should follow Setting facts"


def test_inferences_absent_when_none_provided() -> None:
    facts = [_f(1, "occupation", "surgeon", category="character")]
    prompt = build_system_prompt(_P4_CHAR, facts, inferences=None)
    assert "## Your Inferences" not in prompt


def test_single_fact_per_category_renders_correctly() -> None:
    facts = [
        _f(1, "name", "Jon", category="user"),
        _f(2, "occupation", "surgeon", category="character"),
    ]
    prompt = build_system_prompt(_P4_CHAR, facts)
    assert "name: Jon" in prompt
    assert "occupation: surgeon" in prompt
    # Each fact appears under its respective category section header
    name_idx = prompt.find("name: Jon")
    occ_idx = prompt.find("occupation: surgeon")
    before_name = prompt[:name_idx]
    before_occ = prompt[:occ_idx]
    assert "User" in before_name
    assert "Character" in before_occ
