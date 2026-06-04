"""Unit tests for memories.services.prompt_builder.build_system_prompt."""

from datetime import datetime

from memories.models import Character, Fact
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
