"""Unit tests for memories.services.evaluator."""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest
import respx

from memories.models import Character, Fact, Inference
from memories.services.evaluator import (
    EvaluatorParseError,
    EvaluatorResult,
    build_evaluator_prompt,
    run_evaluator,
)
from memories.services.ollama_client import OllamaClient
from tests.unit.conftest import OLLAMA_BASE_URL, make_ollama_ndjson

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_CHARACTER = Character(
    id=1,
    name="Alice",
    modelfile_base="qwen3:7b",
    current_model_name=None,
    created_at=__import__("datetime").datetime(2024, 1, 1),
)
_FACTS = [
    Fact(
        id=1,
        character_id=1,
        key="occupation",
        value="surgeon",
        created_at=__import__("datetime").datetime(2024, 1, 1),
    ),
    Fact(
        id=2,
        character_id=1,
        key="birthplace",
        value="Reykjavik",
        created_at=__import__("datetime").datetime(2024, 1, 1),
    ),
]
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
# build_evaluator_prompt
# ---------------------------------------------------------------------------


def test_evaluator_prompt_includes_all_facts() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE)
    assert "occupation: surgeon" in prompt
    assert "birthplace: Reykjavik" in prompt


def test_evaluator_prompt_includes_character_response() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE)
    assert _CHAR_RESPONSE in prompt


def test_evaluator_prompt_includes_user_message() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE)
    assert _USER_MSG in prompt


def test_evaluator_prompt_no_facts_uses_fallback_text() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, [], _USER_MSG, _CHAR_RESPONSE)
    # The character's specific fact values must not appear in the facts section
    assert "surgeon" not in prompt
    # Some indicator that there are no facts
    assert "none" in prompt.lower() or "no facts" in prompt.lower()


def test_evaluator_prompt_with_contradiction_hints_lists_them() -> None:
    hints = ["character said 'London' but birthplace is Reykjavik"]
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, hints)
    assert hints[0] in prompt


# ---------------------------------------------------------------------------
# run_evaluator — request shape
# ---------------------------------------------------------------------------


@respx.mock
async def test_evaluator_request_sends_think_false(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json()))
    await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("think") is False


@respx.mock
async def test_evaluator_request_sends_format_json(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json()))
    await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("format") == "json"


# ---------------------------------------------------------------------------
# run_evaluator — verdict parsing
# ---------------------------------------------------------------------------


@respx.mock
async def test_evaluator_parses_pass_verdict(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json("pass")))
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "pass"


@respx.mock
async def test_evaluator_parses_contradiction_verdict(ollama: OllamaClient) -> None:
    violations = [
        {"type": "contradiction", "description": "Character said London", "suggested_fact": None}
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_eval_json("contradiction", violations=violations))
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "contradiction"
    assert result.violations[0].type == "contradiction"


@respx.mock
async def test_evaluator_parses_implication_verdict(ollama: OllamaClient) -> None:
    violations = [
        {
            "type": "implication",
            "description": "Character implied having a sister",
            "suggested_fact": {"key": "siblings", "value": "one sister"},
        }
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_eval_json("implication", violations=violations))
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "implication"
    assert result.violations[0].suggested_fact == {"key": "siblings", "value": "one sister"}


@respx.mock
async def test_evaluator_parses_new_inference_logical(ollama: OllamaClient) -> None:
    inferences = [
        {
            "inference_type": "logical",
            "statement": "Alice was born in 1991",
            "derivation": "age=33, year=2024",
            "source_fact_ids": [1],
            "source_inference_ids": [],
        }
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=_eval_json("new_inference_logical", new_inferences=inferences)
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "new_inference_logical"
    assert result.new_inferences[0].inference_type == "logical"


@respx.mock
async def test_evaluator_parses_new_inference_probabilistic(ollama: OllamaClient) -> None:
    inferences = [
        {
            "inference_type": "probabilistic",
            "statement": "Alice works long hours",
            "derivation": "occupation=surgeon",
            "source_fact_ids": [1],
            "source_inference_ids": [],
        }
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=_eval_json("new_inference_probabilistic", new_inferences=inferences)
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "new_inference_probabilistic"
    assert result.new_inferences[0].inference_type == "probabilistic"


@respx.mock
async def test_evaluator_coerces_experience_update_to_pass(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_eval_json("experience_update"))
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "pass"


@respx.mock
async def test_evaluator_contradiction_priority_overrides_implication(
    ollama: OllamaClient,
) -> None:
    violations = [
        {"type": "implication", "description": "implied a sibling", "suggested_fact": None},
        {
            "type": "contradiction",
            "description": "said London not Reykjavik",
            "suggested_fact": None,
        },
    ]
    # Model returns "implication" but one violation is a contradiction
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_eval_json("implication", violations=violations))
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
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
        await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)


@respx.mock
async def test_evaluator_strips_markdown_code_fence(ollama: OllamaClient) -> None:
    # Some models wrap their JSON in ```json...``` despite being told to return only the object.
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
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "pass"


@respx.mock
async def test_evaluator_raises_parse_error_on_unescaped_quote_in_string(
    ollama: OllamaClient,
) -> None:
    # LLM produced a string containing a literal " (e.g. 5'6" height) — invalid JSON.
    # The chat service catches EvaluatorParseError and falls back to a pass verdict.
    raw = '{"verdict": "pass", "decision_log": "height 5\'6" tall"}'
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=make_ollama_ndjson(raw)))
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)


@respx.mock
async def test_evaluator_raises_parse_error_on_missing_verdict(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"decision_log": "no verdict here"}))
        )
    )
    with pytest.raises(EvaluatorParseError):
        await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)


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
        await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)


# ---------------------------------------------------------------------------
# run_evaluator — return type
# ---------------------------------------------------------------------------


@respx.mock
async def test_run_evaluator_returns_evaluator_result_type(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_eval_json()))
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert isinstance(result, EvaluatorResult)


# ---------------------------------------------------------------------------
# Phase 3 additions — inferences parameter
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
        _FACTS,
        _USER_MSG,
        _CHAR_RESPONSE,
        inferences=[_ESTABLISHED_INFERENCE],
    )
    assert "Alice was born in 1993" in prompt


def test_evaluator_prompt_no_inferences_uses_fallback() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, inferences=[])
    assert "(no inferences established yet)" in prompt


def test_evaluator_prompt_includes_inference_ids() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER,
        _FACTS,
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
        _FACTS,
        _USER_MSG,
        _CHAR_RESPONSE,
        ollama,
        inferences=[_ESTABLISHED_INFERENCE],
    )
    assert isinstance(result, EvaluatorResult)
