"""Unit tests for memories.services.evaluator."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import httpx
import pytest
import respx

from memories.models import Character, Experience, Inference
from memories.services.evaluator import (
    EvaluatorParseError,
    EvaluatorResult,
    Violation,
    build_evaluator_prompt,
    run_evaluator,
)
from memories.services.ollama_client import OllamaClient
from tests.unit.conftest import OLLAMA_BASE_URL, make_evaluator_ndjson, make_ollama_ndjson

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_CHARACTER = Character(
    id=1,
    name="Alice",
    modelfile_base="qwen3:7b",
    current_model_name=None,
    created_at=__import__("datetime").datetime(2024, 1, 1),
)

# Step 3: facts are a schema-constrained blob, not a list of Fact rows.
_FACTS_BLOB: dict[str, Any] = {
    "Character": {
        "Identity": {"Occupation": {"Value": "surgeon"}},
        "Background": {"Hometown": {"Value": "Reykjavik"}},
    }
}

_USER_MSG = "Where are you from?"
_CHAR_RESPONSE = "I grew up in Reykjavik, actually."


def _eval_json(
    verdict: str = "pass",
    new_inferences: list[dict] | None = None,
    violations: list[dict] | None = None,
    decision_log: str = "Clean.",
) -> bytes:
    return make_ollama_ndjson(
        json.dumps(
            {
                "verdict": verdict,
                "new_inferences": new_inferences or [],
                "violations": violations or [],
                "decision_log": decision_log,
            }
        )
    )


# ---------------------------------------------------------------------------
# build_evaluator_prompt — basic content (signature updated to blob)
# ---------------------------------------------------------------------------


def test_evaluator_prompt_includes_all_facts() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "Character.Identity.Occupation" in prompt
    assert "surgeon" in prompt
    assert "Character.Background.Hometown" in prompt
    assert "Reykjavik" in prompt


def test_evaluator_prompt_includes_character_response() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert _CHAR_RESPONSE in prompt


def test_evaluator_prompt_includes_user_message() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert _USER_MSG in prompt


def test_evaluator_prompt_no_facts_uses_fallback_text() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, {}, _USER_MSG, _CHAR_RESPONSE)
    # No specific blob values appear
    assert "surgeon" not in prompt
    # Empty blob produces an unset note
    assert "unset" in prompt.lower()


def test_evaluator_prompt_with_contradiction_hints_lists_them() -> None:
    hints = ["character said 'London' but hometown is Reykjavik"]
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, hints)
    assert hints[0] in prompt


# ---------------------------------------------------------------------------
# run_evaluator — request shape
# ---------------------------------------------------------------------------


@respx.mock
async def test_evaluator_request_sends_think_false(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json()))
    await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("think") is False


@respx.mock
async def test_evaluator_request_sends_format_json(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json()))
    await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("format") == "json"


# ---------------------------------------------------------------------------
# run_evaluator — verdict parsing
# ---------------------------------------------------------------------------


@respx.mock
async def test_evaluator_parses_pass_verdict(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json("pass")))
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "pass"


@respx.mock
async def test_evaluator_parses_contradiction_verdict(ollama: OllamaClient) -> None:
    violations = [{"type": "contradiction", "description": "Character said London"}]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_eval_json("contradiction", violations=violations))
    )
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "contradiction"
    assert result.violations[0].type == "contradiction"


@respx.mock
async def test_evaluator_parses_new_inference_logical(ollama: OllamaClient) -> None:
    inferences = [
        {
            "inference_type": "logical",
            "statement": "Alice was born in 1991",
            "derivation": "age=33, year=2024",
            "source_fact_paths": ["Character.Identity.Age"],
            "source_inference_ids": [],
        }
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=_eval_json("new_inference_logical", new_inferences=inferences)
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "new_inference_logical"
    assert result.new_inferences[0].inference_type == "logical"


@respx.mock
async def test_evaluator_parses_new_inference_probabilistic(ollama: OllamaClient) -> None:
    inferences = [
        {
            "inference_type": "probabilistic",
            "statement": "Alice works long hours",
            "derivation": "occupation=surgeon",
            "source_fact_paths": ["Character.Identity.Occupation"],
            "source_inference_ids": [],
        }
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=_eval_json("new_inference_probabilistic", new_inferences=inferences)
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "new_inference_probabilistic"
    assert result.new_inferences[0].inference_type == "probabilistic"


@respx.mock
async def test_evaluator_contradiction_priority_overrides_other_verdict(
    ollama: OllamaClient,
) -> None:
    violations = [
        {"type": "contradiction", "description": "said London not Reykjavik"},
    ]
    # Model returns a valid non-contradiction verdict but has a contradiction violation
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=_eval_json("new_inference_logical", violations=violations)
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "contradiction"


# ---------------------------------------------------------------------------
# run_evaluator — error handling
# ---------------------------------------------------------------------------


@respx.mock
async def test_evaluator_raises_parse_error_on_non_json(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("This is not JSON at all."))
    )
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)


@respx.mock
async def test_evaluator_strips_markdown_code_fence(ollama: OllamaClient) -> None:
    fenced = (
        "```json\n"
        + json.dumps(
            {
                "verdict": "pass",
                "new_inferences": [],
                "violations": [],
                "decision_log": "Clean.",
            }
        )
        + "\n```"
    )
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=make_ollama_ndjson(fenced)))
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "pass"


@respx.mock
async def test_evaluator_raises_parse_error_on_unescaped_quote_in_string(
    ollama: OllamaClient,
) -> None:
    raw = '{"verdict": "pass", "decision_log": "height 5\'6" tall"}'
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=make_ollama_ndjson(raw)))
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)


@respx.mock
async def test_evaluator_raises_parse_error_on_missing_verdict(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"decision_log": "no verdict here"}))
        )
    )
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)


@respx.mock
async def test_evaluator_raises_parse_error_on_unknown_verdict(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_ollama_ndjson(
                json.dumps({"verdict": "made_up_verdict", "decision_log": "hmm"})
            ),
        )
    )
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)


# ---------------------------------------------------------------------------
# run_evaluator — return type
# ---------------------------------------------------------------------------


@respx.mock
async def test_run_evaluator_returns_evaluator_result_type(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json()))
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert isinstance(result, EvaluatorResult)


# ---------------------------------------------------------------------------
# Phase 3 additions — inferences parameter (signature updated)
# ---------------------------------------------------------------------------

_EVAL_NOW = datetime(2026, 1, 1)

_ESTABLISHED_INFERENCE = Inference(
    id=10,
    character_id=1,
    statement="Alice was born in 1993",
    derivation="age=33, current_year=2026",
    source_fact_ids=[1],
    source_inference_ids=[],
    depth=1,
    inference_type="logical",
    status="active",
    created_at=_EVAL_NOW,
)


def test_evaluator_prompt_includes_established_inferences() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER,
        _FACTS_BLOB,
        _USER_MSG,
        _CHAR_RESPONSE,
        inferences=[_ESTABLISHED_INFERENCE],
    )
    assert "Alice was born in 1993" in prompt


def test_evaluator_prompt_no_inferences_uses_fallback() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, inferences=[]
    )
    assert "(no inferences established yet)" in prompt


def test_evaluator_prompt_includes_inference_ids() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER,
        _FACTS_BLOB,
        _USER_MSG,
        _CHAR_RESPONSE,
        inferences=[_ESTABLISHED_INFERENCE],
    )
    assert "[10]" in prompt


@respx.mock
async def test_evaluator_accepts_inferences_parameter(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json()))
    result = await run_evaluator(
        _CHARACTER,
        _FACTS_BLOB,
        _USER_MSG,
        _CHAR_RESPONSE,
        ollama,
        inferences=[_ESTABLISHED_INFERENCE],
    )
    assert isinstance(result, EvaluatorResult)


# ---------------------------------------------------------------------------
# Phase 4 — only the immutable contradiction instruction is retained
# ---------------------------------------------------------------------------


def test_evaluator_prompt_contains_immutable_contradiction_instruction() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    prompt_lower = prompt.lower()
    assert "immutable" in prompt_lower
    assert "contradiction" in prompt_lower


# ---------------------------------------------------------------------------
# Phase 5 additions — experiences parameter (retained, signature updated)
# ---------------------------------------------------------------------------

_P5_NOW = datetime(2026, 6, 7)

_P5_EXPERIENCES = [
    Experience(
        id=5,
        character_id=1,
        session_id=1,
        statement="We are currently located in Chicago",
        source="told_by_user",
        approved_at=_P5_NOW,
        created_at=_P5_NOW,
    ),
    Experience(
        id=7,
        character_id=1,
        session_id=1,
        statement="Jon seemed uncomfortable when asked about his family",
        source="observed",
        approved_at=_P5_NOW,
        created_at=_P5_NOW,
    ),
]


def test_evaluator_prompt_includes_active_experiences() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "We are currently located in Chicago" in prompt
    assert "Jon seemed uncomfortable" in prompt


def test_evaluator_prompt_no_experiences_uses_fallback() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, experiences=[]
    )
    assert "no active experiences" in prompt.lower()


def test_evaluator_prompt_includes_experience_ids() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "[5]" in prompt
    assert "[7]" in prompt


def test_evaluator_prompt_includes_experience_source_label() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "told_by_user" in prompt or "told by user" in prompt.lower()
    assert "observed" in prompt


@respx.mock
async def test_run_evaluator_accepts_experiences_parameter(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_evaluator_ndjson("pass"))
    )
    result = await run_evaluator(
        _CHARACTER,
        _FACTS_BLOB,
        _USER_MSG,
        _CHAR_RESPONSE,
        ollama,
        experiences=_P5_EXPERIENCES,
    )
    assert isinstance(result, EvaluatorResult)


# ---------------------------------------------------------------------------
# Step 3 additions — new tests for schema section, blob rendering, narrowed verdicts
# ---------------------------------------------------------------------------


def test_evaluator_prompt_includes_schema_section() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "## Fact Schema" in prompt


def test_evaluator_prompt_includes_immutable_paths() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "IMMUTABLE" in prompt


def test_evaluator_prompt_includes_fact_values_section() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "## Current Fact Values" in prompt


def test_evaluator_prompt_renders_populated_path() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "Character.Identity.Occupation" in prompt
    assert "surgeon" in prompt


def test_evaluator_prompt_empty_blob_produces_unset_note() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, {}, _USER_MSG, _CHAR_RESPONSE)
    assert "(all other schema paths are unset)" in prompt


@respx.mock
async def test_evaluator_raises_parse_error_on_implication_verdict(
    ollama: OllamaClient,
) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json("implication")))
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)


@respx.mock
async def test_evaluator_raises_parse_error_on_experience_update_verdict(
    ollama: OllamaClient,
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_eval_json("experience_update"))
    )
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)


@respx.mock
async def test_new_inference_source_fact_paths_are_strings(ollama: OllamaClient) -> None:
    inferences = [
        {
            "inference_type": "logical",
            "statement": "Alice was born in 1991",
            "derivation": "age=33, year=2024",
            "source_fact_paths": ["Character.Identity.Age"],
            "source_inference_ids": [],
        }
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=_eval_json("new_inference_logical", new_inferences=inferences)
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    paths = result.new_inferences[0].source_fact_paths
    assert paths == ["Character.Identity.Age"]
    assert all(isinstance(p, str) for p in paths)


def test_violation_has_no_suggested_fact_field() -> None:
    """Violation no longer carries a suggested_fact field — removed in Step 3."""
    v = Violation(type="contradiction", description="immutable fact was contradicted")
    assert not hasattr(v, "suggested_fact")
