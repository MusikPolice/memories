"""Unit tests for memories.services.evaluator."""

from __future__ import annotations

import json
from datetime import datetime

import httpx
import pytest
import respx

from memories.models import Character, Experience, Fact, Inference
from memories.services.evaluator import (
    EvaluatorParseError,
    EvaluatorResult,
    ExperienceUpdate,
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
async def test_evaluator_returns_experience_update_verdict_not_coerced(
    ollama: OllamaClient,
) -> None:
    # Phase 5: experience_update should be returned as-is, not coerced to pass.
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_eval_json("experience_update"))
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "experience_update"


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


# ---------------------------------------------------------------------------
# Phase 4 additions — category/mutability labels in evaluator prompt
# ---------------------------------------------------------------------------

_P4_NOW = datetime(2026, 1, 1)

_P4_FACTS = [
    Fact(
        id=1,
        character_id=1,
        key="occupation",
        value="surgeon",
        category="character",
        mutability="immutable",
        created_at=_P4_NOW,
    ),
    Fact(
        id=2,
        character_id=1,
        key="mood",
        value="cheerful",
        category="character",
        mutability="high",
        created_at=_P4_NOW,
    ),
    Fact(
        id=3,
        character_id=1,
        key="location",
        value="Chicago",
        category="setting",
        mutability="low",
        created_at=_P4_NOW,
    ),
]


def test_evaluator_prompt_includes_category_for_each_fact() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _P4_FACTS, _USER_MSG, _CHAR_RESPONSE)
    assert "category: character" in prompt
    assert "category: setting" in prompt


def test_evaluator_prompt_includes_mutability_for_each_fact() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _P4_FACTS, _USER_MSG, _CHAR_RESPONSE)
    assert "mutability: immutable" in prompt
    assert "mutability: high" in prompt
    assert "mutability: low" in prompt


def test_evaluator_prompt_labels_user_category_facts() -> None:
    facts = [
        Fact(
            id=1,
            character_id=1,
            key="user_name",
            value="Jon",
            category="user",
            mutability="immutable",
            created_at=_P4_NOW,
        )
    ]
    prompt = build_evaluator_prompt(_CHARACTER, facts, _USER_MSG, _CHAR_RESPONSE)
    assert "category: user" in prompt


def test_evaluator_prompt_labels_setting_category_facts() -> None:
    facts = [
        Fact(
            id=1,
            character_id=1,
            key="city",
            value="Chicago",
            category="setting",
            mutability="low",
            created_at=_P4_NOW,
        )
    ]
    prompt = build_evaluator_prompt(_CHARACTER, facts, _USER_MSG, _CHAR_RESPONSE)
    assert "category: setting" in prompt


def test_evaluator_prompt_contains_immutable_contradiction_instruction() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _P4_FACTS, _USER_MSG, _CHAR_RESPONSE)
    prompt_lower = prompt.lower()
    assert "immutable" in prompt_lower
    assert "contradiction" in prompt_lower


def test_evaluator_prompt_contains_high_mutability_implication_instruction() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _P4_FACTS, _USER_MSG, _CHAR_RESPONSE)
    prompt_lower = prompt.lower()
    # The prompt must explain that high-mutability changes return implication
    assert "high" in prompt_lower
    assert "implication" in prompt_lower


def test_evaluator_prompt_contains_low_mutability_implication_instruction() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _P4_FACTS, _USER_MSG, _CHAR_RESPONSE)
    prompt_lower = prompt.lower()
    assert "low" in prompt_lower
    assert "implication" in prompt_lower


def test_evaluator_prompt_format_for_fact_with_all_fields() -> None:
    facts = [
        Fact(
            id=5,
            character_id=1,
            key="mood",
            value="cheerful",
            category="character",
            mutability="high",
            created_at=_P4_NOW,
        )
    ]
    prompt = build_evaluator_prompt(_CHARACTER, facts, _USER_MSG, _CHAR_RESPONSE)
    # Each fact line should include: [id] key: value (category: X, mutability: Y)
    assert "[5]" in prompt
    assert "mood: cheerful" in prompt
    assert "category: character" in prompt
    assert "mutability: high" in prompt


# ---------------------------------------------------------------------------
# Regression: high-mutability fact domain must block new_inference_* verdicts
# ---------------------------------------------------------------------------


def test_evaluator_prompt_instructs_mandatory_high_mutability_scan() -> None:
    """Prompt must direct the model to scan high-mutability facts BEFORE inferring."""
    prompt = build_evaluator_prompt(_CHARACTER, _P4_FACTS, _USER_MSG, _CHAR_RESPONSE)
    prompt_lower = prompt.lower()
    # The prompt must contain language that makes the scan mandatory / first step
    assert "mandatory" in prompt_lower or "first" in prompt_lower
    assert "scan" in prompt_lower or "check each" in prompt_lower or "check every" in prompt_lower


def test_evaluator_prompt_blocks_new_inference_when_high_mutability_fact_covers_domain() -> None:
    """Prompt must explicitly state that new_inference_* is invalid when a high-mutability
    Fact already covers the same domain."""
    prompt = build_evaluator_prompt(_CHARACTER, _P4_FACTS, _USER_MSG, _CHAR_RESPONSE)
    prompt_lower = prompt.lower()
    # Must say new_inference is not appropriate / never valid for covered domains
    assert "new_inference" in prompt_lower
    assert (
        "never" in prompt_lower
        or "not valid" in prompt_lower
        or "not appropriate" in prompt_lower
        or "only" in prompt_lower
    )


def test_evaluator_prompt_gives_stress_level_as_high_mutability_implication_example() -> None:
    """Prompt must include a concrete example showing a stress/mood change against an
    existing high-mutability fact must return implication, not new_inference_probabilistic."""
    facts = [
        Fact(
            id=6,
            character_id=1,
            key="stress_level",
            value="low",
            category="character",
            mutability="high",
            created_at=_P4_NOW,
        )
    ]
    prompt = build_evaluator_prompt(
        _CHARACTER,
        facts,
        "You scratched my car!",
        "My stress level is shooting up just thinking about how much paint is gone.",
    )
    prompt_lower = prompt.lower()
    # Prompt must mention that changes to high-mutability facts are implication not inference
    assert "implication" in prompt_lower
    # Prompt must contain either an explicit stress_level example or the domain-coverage rule
    assert (
        "stress" in prompt_lower
        or "existing fact" in prompt_lower
        or "no existing fact" in prompt_lower
    )


def test_evaluator_prompt_high_mutability_mood_change_must_not_use_new_inference() -> None:
    """When a high-mutability mood fact exists, the prompt must not permit new_inference_*
    to classify a mood shift — it must direct the model toward implication."""
    facts = [
        Fact(
            id=2,
            character_id=1,
            key="mood",
            value="happy",
            category="character",
            mutability="high",
            created_at=_P4_NOW,
        )
    ]
    prompt = build_evaluator_prompt(
        _CHARACTER,
        facts,
        "Stop being so cheerful!",
        "I feel really anxious and stressed right now.",
    )
    prompt_lower = prompt.lower()
    # Must have guidance that high-mutability changes require implication
    assert "implication" in prompt_lower
    assert "high" in prompt_lower


# ---------------------------------------------------------------------------
# Phase 5 additions — experiences parameter and experience_update verdict
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
        _CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "We are currently located in Chicago" in prompt
    assert "Jon seemed uncomfortable" in prompt


def test_evaluator_prompt_no_experiences_uses_fallback() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, experiences=[])
    assert "no active experiences" in prompt.lower()


def test_evaluator_prompt_includes_experience_ids() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "[5]" in prompt
    assert "[7]" in prompt


def test_evaluator_prompt_includes_experience_source_label() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "told_by_user" in prompt or "told by user" in prompt.lower()
    assert "observed" in prompt


def test_evaluator_prompt_contains_experience_update_instructions() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "experience_update" in prompt


def test_evaluator_prompt_contains_experience_update_json_schema() -> None:
    prompt = build_evaluator_prompt(
        _CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, experiences=_P5_EXPERIENCES
    )
    assert "experience_updates" in prompt


@respx.mock
async def test_run_evaluator_accepts_experiences_parameter(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_evaluator_ndjson("pass"))
    )
    result = await run_evaluator(
        _CHARACTER,
        _FACTS,
        _USER_MSG,
        _CHAR_RESPONSE,
        ollama,
        experiences=_P5_EXPERIENCES,
    )
    assert isinstance(result, EvaluatorResult)


@respx.mock
async def test_run_evaluator_returns_experience_update_verdict(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_evaluator_ndjson("experience_update"))
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "experience_update"


@respx.mock
async def test_run_evaluator_parses_experience_updates_list(ollama: OllamaClient) -> None:
    exp_updates = [
        {"contradicted_experience_id": 5, "description": "Character implies they are in New York"},
        {"contradicted_experience_id": 7, "description": "Character says family is fine"},
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_evaluator_ndjson("experience_update", experience_updates=exp_updates),
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert len(result.experience_updates) == 2
    assert isinstance(result.experience_updates[0], ExperienceUpdate)


@respx.mock
async def test_run_evaluator_experience_update_not_coerced_to_pass(
    ollama: OllamaClient,
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_evaluator_ndjson("experience_update"))
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "experience_update"
    assert result.verdict != "pass"


def test_evaluator_result_model_has_experience_updates_field() -> None:
    result = EvaluatorResult(verdict="pass", decision_log="ok")
    assert hasattr(result, "experience_updates")
    assert result.experience_updates == []


@respx.mock
async def test_contradiction_takes_priority_over_experience_update(
    ollama: OllamaClient,
) -> None:
    violations = [
        {"type": "contradiction", "description": "Contradicts fact", "suggested_fact": None}
    ]
    exp_updates = [{"contradicted_experience_id": 5, "description": "Also contradicts experience"}]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_evaluator_ndjson(
                "contradiction", violations=violations, experience_updates=exp_updates
            ),
        )
    )
    result = await run_evaluator(_CHARACTER, _FACTS, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert result.verdict == "contradiction"
