"""Unit tests for memories.services.extraction_service — Phase 6.

Tests 1-8:  build_extractor_prompt — prompt content
Tests 9-17: parse_extraction_result — JSON parsing
Tests 18-19: run_fact_extractor — full Ollama interaction

All tests fail before Phase 6 is implemented (stubs raise NotImplementedError).
"""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest
import respx

from memories.models import Character, Fact, Inference
from memories.services.extraction_service import (
    ExtractionParseError,
    ExtractionResult,
    build_extractor_prompt,
    parse_extraction_result,
    run_fact_extractor,
)
from memories.services.ollama_client import OllamaClient
from tests.unit.conftest import OLLAMA_BASE_URL, make_extractor_ndjson, make_ollama_ndjson

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_NOW = datetime(2026, 1, 1)

_CHARACTER = Character(
    id=1,
    name="Alice",
    modelfile_base="qwen3:7b",
    current_model_name=None,
    created_at=_NOW,
)

_FACTS = [
    Fact(
        id=1,
        character_id=1,
        key="occupation",
        value="surgeon",
        category="character",
        mutability="immutable",
        created_at=_NOW,
    ),
    Fact(
        id=2,
        character_id=1,
        key="home_city",
        value="Reykjavik",
        category="user",
        mutability="low",
        created_at=_NOW,
    ),
]

_INFERENCES = [
    Inference(
        id=10,
        character_id=1,
        statement="Alice was born in 1993",
        derivation="age=33, current_year=2026",
        source_fact_ids=[1],
        source_inference_ids=[],
        depth=1,
        inference_type="logical",
        status="active",
        created_at=_NOW,
    ),
]

_USER_MSG = "We're meeting in Chicago next week."


# ---------------------------------------------------------------------------
# build_extractor_prompt — prompt content
# ---------------------------------------------------------------------------


def test_extractor_prompt_includes_user_message() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES)
    assert _USER_MSG in prompt


def test_extractor_prompt_includes_all_existing_facts() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES)
    assert "occupation" in prompt
    assert "surgeon" in prompt
    assert "home_city" in prompt
    assert "Reykjavik" in prompt


def test_extractor_prompt_includes_fact_ids() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES)
    assert "[1]" in prompt
    assert "[2]" in prompt


def test_extractor_prompt_includes_fact_categories() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES)
    assert "character" in prompt
    assert "user" in prompt


def test_extractor_prompt_includes_fact_mutability() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES)
    assert "immutable" in prompt
    assert "low" in prompt


def test_extractor_prompt_includes_inferences() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES)
    assert "Alice was born in 1993" in prompt


def test_extractor_prompt_omits_inferences_section_when_none() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, _FACTS, [])
    assert "Alice was born in 1993" not in prompt


def test_extractor_prompt_no_facts_shows_placeholder() -> None:
    prompt = build_extractor_prompt(_USER_MSG, _CHARACTER, [], [])
    # Must not be an empty/blank section — should have a placeholder
    assert (
        "none" in prompt.lower()
        or "no facts" in prompt.lower()
        or "no established" in prompt.lower()
    )


# ---------------------------------------------------------------------------
# parse_extraction_result — JSON parsing
# ---------------------------------------------------------------------------


def test_parse_extraction_result_new_facts_only() -> None:
    data = {
        "new_facts": [
            {
                "key": "location",
                "value": "Chicago",
                "category": "setting",
                "mutability": "low",
                "source_quote": "We're meeting in Chicago",
            }
        ],
        "fact_updates": [],
        "implicit_proposals": [],
    }
    result = parse_extraction_result(json.dumps(data))
    assert isinstance(result, ExtractionResult)
    assert len(result.new_facts) == 1
    assert result.new_facts[0].key == "location"
    assert result.new_facts[0].value == "Chicago"


def test_parse_extraction_result_fact_updates_only() -> None:
    data = {
        "new_facts": [],
        "fact_updates": [
            {
                "fact_id": 2,
                "key": "home_city",
                "old_value": "Reykjavik",
                "new_value": "Chicago",
                "source_quote": "I moved to Chicago",
            }
        ],
        "implicit_proposals": [],
    }
    result = parse_extraction_result(json.dumps(data))
    assert isinstance(result, ExtractionResult)
    assert len(result.fact_updates) == 1
    assert result.fact_updates[0].fact_id == 2
    assert result.fact_updates[0].old_value == "Reykjavik"
    assert result.fact_updates[0].new_value == "Chicago"


def test_parse_extraction_result_implicit_proposals_only() -> None:
    data = {
        "new_facts": [],
        "fact_updates": [],
        "implicit_proposals": [
            {
                "key": "mood",
                "value": "anxious",
                "category": "user",
                "mutability": "high",
                "source_quote": "feeling off all week",
            }
        ],
    }
    result = parse_extraction_result(json.dumps(data))
    assert isinstance(result, ExtractionResult)
    assert len(result.implicit_proposals) == 1
    assert result.implicit_proposals[0].key == "mood"


def test_parse_extraction_result_implicit_proposal_new_has_no_existing_fact_id() -> None:
    data = {
        "new_facts": [],
        "fact_updates": [],
        "implicit_proposals": [
            {
                "key": "mood",
                "value": "anxious",
                "category": "user",
                "mutability": "high",
                "source_quote": "feeling off",
            }
        ],
    }
    result = parse_extraction_result(json.dumps(data))
    assert result.implicit_proposals[0].existing_fact_id is None


def test_parse_extraction_result_implicit_proposal_update_has_existing_fact_id() -> None:
    data = {
        "new_facts": [],
        "fact_updates": [],
        "implicit_proposals": [
            {
                "key": "home_city",
                "value": "Chicago",
                "category": "user",
                "mutability": "low",
                "source_quote": "just got home in Chicago",
                "existing_fact_id": 2,
                "old_value": "Reykjavik",
            }
        ],
    }
    result = parse_extraction_result(json.dumps(data))
    assert result.implicit_proposals[0].existing_fact_id == 2
    assert result.implicit_proposals[0].old_value == "Reykjavik"


def test_parse_extraction_result_all_three_lists_populated() -> None:
    data = {
        "new_facts": [
            {
                "key": "location",
                "value": "Chicago",
                "category": "setting",
                "mutability": "low",
                "source_quote": "q1",
            }
        ],
        "fact_updates": [
            {
                "fact_id": 2,
                "key": "home_city",
                "old_value": "Reykjavik",
                "new_value": "Chicago",
                "source_quote": "q2",
            }
        ],
        "implicit_proposals": [
            {
                "key": "mood",
                "value": "excited",
                "category": "user",
                "mutability": "high",
                "source_quote": "q3",
            }
        ],
    }
    result = parse_extraction_result(json.dumps(data))
    assert len(result.new_facts) == 1
    assert len(result.fact_updates) == 1
    assert len(result.implicit_proposals) == 1


def test_parse_extraction_result_all_empty() -> None:
    data = {"new_facts": [], "fact_updates": [], "implicit_proposals": []}
    result = parse_extraction_result(json.dumps(data))
    assert isinstance(result, ExtractionResult)
    assert result.new_facts == []
    assert result.fact_updates == []
    assert result.implicit_proposals == []


def test_parse_extraction_result_invalid_json_raises_error() -> None:
    with pytest.raises(ExtractionParseError):
        parse_extraction_result("this is not json at all")


def test_parse_extraction_result_missing_required_fields_raises_error() -> None:
    # new_fact entry missing "value" — required field
    data = {
        "new_facts": [
            {"key": "location", "category": "setting", "mutability": "low", "source_quote": "q"}
        ],
        "fact_updates": [],
        "implicit_proposals": [],
    }
    with pytest.raises(ExtractionParseError):
        parse_extraction_result(json.dumps(data))


# ---------------------------------------------------------------------------
# run_fact_extractor — full Ollama interaction
# ---------------------------------------------------------------------------


@respx.mock
async def test_run_fact_extractor_returns_extraction_result(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=make_extractor_ndjson()))
    result = await run_fact_extractor(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES, ollama)
    assert isinstance(result, ExtractionResult)


@respx.mock
async def test_run_fact_extractor_on_parse_error_raises_extraction_parse_error(
    ollama: OllamaClient,
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("this is not valid json"))
    )
    with pytest.raises(ExtractionParseError):
        await run_fact_extractor(_USER_MSG, _CHARACTER, _FACTS, _INFERENCES, ollama)
