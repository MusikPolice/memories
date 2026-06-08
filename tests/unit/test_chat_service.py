"""Unit tests for memories.services.chat_service.run_turn.

These tests use a real in-memory SQLite database (via the shared ``db``
fixture) and a mocked Ollama HTTP layer (via respx).  They test the
orchestration contract of run_turn — what it reads, what it writes, and in
what order — not the HTTP or SQL layers themselves.

Phase 2: every successful run_turn call makes TWO Ollama requests:
  calls[0] — character LLM
  calls[1] — evaluator LLM
All tests that complete a turn successfully must therefore mock both calls.
"""

from __future__ import annotations

import json

import aiosqlite
import httpx
import pytest
import respx

from memories.database import (
    _embedding_to_blob,
    create_experience,
    create_fact,
    get_decisions,
    get_experiences,
    get_inferences,
    get_messages,
)
from memories.exceptions import NotFoundError, SessionEndedError
from memories.models import Character, Session
from memories.services.chat_service import run_turn
from memories.services.evaluator import EvaluatorResult
from memories.services.experience_service import get_active_experiences
from memories.services.ollama_client import OllamaClient, OllamaConnectionError
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_embed_response,
    make_evaluator_ndjson,
    make_ollama_ndjson,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"


def _mock_ok(content: str = "I am fine, thank you.") -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(content))


def _mock_eval(
    verdict: str = "pass",
    new_inferences: list[dict] | None = None,
    violations: list[dict] | None = None,
) -> httpx.Response:
    return httpx.Response(200, content=make_evaluator_ndjson(verdict, new_inferences, violations))


def _mock_turn(
    character_content: str = "I am fine, thank you.",
    evaluator_verdict: str = "pass",
    new_inferences: list[dict] | None = None,
    violations: list[dict] | None = None,
) -> list[httpx.Response]:
    """Return side_effect list for one complete turn (character + evaluator)."""
    return [_mock_ok(character_content), _mock_eval(evaluator_verdict, new_inferences, violations)]


# ---------------------------------------------------------------------------
# Tests: what run_turn sends to the character LLM (calls[0])
# ---------------------------------------------------------------------------


async def test_system_message_is_first_in_ollama_request(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    body = json.loads(route.calls[0].request.content)
    assert body["messages"][0]["role"] == "system"


async def test_history_included_in_ollama_request(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("First response."))
        await run_turn(db, session.id, "Turn one", ollama)

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Second response."))
        await run_turn(db, session.id, "Turn two", ollama)

    body = json.loads(route.calls[0].request.content)
    roles = [m["role"] for m in body["messages"]]
    assert "user" in roles
    assert "assistant" in roles


async def test_history_ordered_by_turn_id(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Reply one."))
        await run_turn(db, session.id, "First message", ollama)

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Reply two."))
        await run_turn(db, session.id, "Second message", ollama)

    body = json.loads(route.calls[0].request.content)
    conversation = [m for m in body["messages"] if m["role"] != "system"]
    assert conversation[0]["content"] == "First message"
    assert conversation[1]["content"] == "Reply one."
    assert conversation[2]["content"] == "Second message"


async def test_new_user_message_appended_last(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "My specific question", ollama)

    body = json.loads(route.calls[0].request.content)
    assert body["messages"][-1]["role"] == "user"
    assert body["messages"][-1]["content"] == "My specific question"


async def test_facts_reflected_in_system_message(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await create_fact(db, character_id=character.id, key="birthplace", value="Reykjavik")

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Where are you from?", ollama)

    body = json.loads(route.calls[0].request.content)
    system_content: str = body["messages"][0]["content"]
    assert "birthplace" in system_content
    assert "Reykjavik" in system_content


# ---------------------------------------------------------------------------
# Tests: what run_turn writes to the DB
# ---------------------------------------------------------------------------


async def test_user_message_stored_before_llm_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """The user message must be persisted even if the Ollama call subsequently fails."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(OllamaConnectionError):
            await run_turn(db, session.id, "Hi there", ollama)

    messages = await get_messages(db, session.id)
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "Hi there"


async def test_assistant_message_stored_after_llm_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Stored assistant reply."))
        await run_turn(db, session.id, "Hello", ollama)

    messages = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "Stored assistant reply." for m in messages)


async def test_turn_ids_increment(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("First."))
        await run_turn(db, session.id, "Message one", ollama)

    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Second."))
        await run_turn(db, session.id, "Message two", ollama)

    messages = await get_messages(db, session.id)
    turn_ids = [m.turn_id for m in messages]
    assert sorted(set(turn_ids)) == [1, 2]


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


async def test_run_turn_raises_on_unknown_session(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    with pytest.raises(NotFoundError):
        await run_turn(db, session_id=9999, user_content="Hello", ollama=ollama)


async def test_run_turn_raises_on_ended_session(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import end_session

    await end_session(db, session.id)
    with pytest.raises(SessionEndedError):
        await run_turn(db, session.id, "Hello", ollama)


# ---------------------------------------------------------------------------
# Phase 2: evaluator integration
# ---------------------------------------------------------------------------


async def test_evaluator_called_after_character_response(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Character response"))
        await run_turn(db, session.id, "Hello", ollama)
    # Two calls: character then evaluator
    assert len(route.calls) == 2


async def test_evaluator_called_with_character_response(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Character says hi"))
        await run_turn(db, session.id, "Hello", ollama)
    # The evaluator prompt (calls[1]) should contain the character's response
    eval_body = json.loads(route.calls[1].request.content)
    user_msgs = [m for m in eval_body["messages"] if m["role"] == "user"]
    assert any("Character says hi" in m["content"] for m in user_msgs)


async def test_run_turn_returns_evaluator_result(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        _, _, _, eval_result, _ = await run_turn(db, session.id, "Hello", ollama)
    assert isinstance(eval_result, EvaluatorResult)


async def test_pass_verdict_response_stored_and_returned(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Clean reply."))
        content, _, _turn_id, _, _ = await run_turn(db, session.id, "Hello", ollama)
    assert content == "Clean reply."
    msgs = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "Clean reply." for m in msgs)


async def test_pass_verdict_decision_stored(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)
    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 1
    assert decisions[0].verdict == "pass"


async def test_contradiction_response_not_stored_on_first_attempt(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """A contradicting response must NOT be stored; only the clean one is."""
    contradiction_violation = [
        {
            "type": "contradiction",
            "description": "said London not Reykjavik",
            "suggested_fact": None,
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("I'm from London."),  # character (contradicts)
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("I'm from Reykjavik."),  # character regenerated (clean)
                _mock_eval("pass"),  # evaluator (pass)
            ]
        )
        await run_turn(db, session.id, "Where are you from?", ollama)
    msgs = await get_messages(db, session.id)
    # No "London" message stored; only the clean "Reykjavik" response
    assert not any("London" in m.content for m in msgs if m.role == "assistant")
    assert any("Reykjavik" in m.content for m in msgs if m.role == "assistant")


async def test_contradiction_triggers_second_character_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "wrong city", "suggested_fact": None}
    ]
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Wrong answer."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Correct answer."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "Hello", ollama)
    # 4 calls total: char1, eval1, char2, eval2
    assert len(route.calls) == 4


async def test_contradiction_second_call_messages_include_system_note(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "wrong city", "suggested_fact": None}
    ]
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Wrong."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Right."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "Hello", ollama)
    # calls[2] is the second character call (after contradiction)
    body = json.loads(route.calls[2].request.content)
    all_content = " ".join(m["content"] for m in body["messages"])
    assert "wrong city" in all_content.lower() or "contradiction" in all_content.lower()


async def test_contradiction_loop_exits_on_pass(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "error", "suggested_fact": None}
    ]
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Bad."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Good."),
                _mock_eval("pass"),
            ]
        )
        content, _, _, eval_result, _ = await run_turn(db, session.id, "Hello", ollama)
    assert content == "Good."
    assert eval_result.verdict == "pass"
    assert len(route.calls) == 4


async def test_contradiction_loop_final_response_is_stored(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "error", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Bad."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Clean response."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "Hello", ollama)
    msgs = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "Clean response." for m in msgs)


async def test_contradiction_max_retries_exceeded_delivers_anyway(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """After MAX_CONTRADICTION_RETRIES the response is delivered regardless."""
    from memories.services import chat_service

    max_retries = chat_service.MAX_CONTRADICTION_RETRIES
    contradiction_violation = [
        {"type": "contradiction", "description": "always wrong", "suggested_fact": None}
    ]
    # Need (max_retries + 1) pairs of [character, evaluator] calls
    side_effects: list[httpx.Response] = []
    for _ in range(max_retries + 1):
        side_effects.append(_mock_ok("Still wrong."))
        side_effects.append(_mock_eval("contradiction", violations=contradiction_violation))

    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=side_effects)
        content, _, _, eval_result, _ = await run_turn(db, session.id, "Hello", ollama)

    assert content == "Still wrong."
    assert eval_result.max_retries_exceeded is True


async def test_contradiction_notifications_collected_per_iteration(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "bad", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Wrong 1."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Wrong 2."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Correct."),
                _mock_eval("pass"),
            ]
        )
        _, _, _, eval_result, _ = await run_turn(db, session.id, "Hello", ollama)
    assert len(eval_result.contradiction_notifications) == 2


async def test_implication_verdict_tags_message_ungrounded(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    violations = [
        {
            "type": "implication",
            "description": "implied a sibling",
            "suggested_fact": {"key": "siblings", "value": "one"},
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn("I have a sister.", "implication", violations=violations)
        )
        await run_turn(db, session.id, "Family?", ollama)
    msgs = await get_messages(db, session.id)
    assistant = next(m for m in msgs if m.role == "assistant")
    assert assistant.ungrounded_implications is not None


async def test_implication_violations_stored_in_message(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    violations = [
        {
            "type": "implication",
            "description": "implied a sibling",
            "suggested_fact": {"key": "siblings", "value": "one"},
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn("I have a sister.", "implication", violations=violations)
        )
        await run_turn(db, session.id, "Family?", ollama)
    msgs = await get_messages(db, session.id)
    assistant = next(m for m in msgs if m.role == "assistant")
    assert isinstance(assistant.ungrounded_implications, list)
    assert assistant.ungrounded_implications[0].get("suggested_fact") == {
        "key": "siblings",
        "value": "one",
    }


async def test_new_inference_logical_creates_inference_row(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    inferences = [
        {
            "inference_type": "logical",
            "statement": "Born in 1991",
            "derivation": "age=33, year=2024",
            "source_fact_ids": [],
            "source_inference_ids": [],
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn(
                "I was born in 1991.", "new_inference_logical", new_inferences=inferences
            )
        )
        await run_turn(db, session.id, "When were you born?", ollama)
    stored = await get_inferences(db, character.id)
    assert any(i.statement == "Born in 1991" for i in stored)


async def test_new_inference_probabilistic_tags_message_ungrounded(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    inferences = [
        {
            "inference_type": "probabilistic",
            "statement": "Works long hours",
            "derivation": "occupation=surgeon",
            "source_fact_ids": [],
            "source_inference_ids": [],
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn(
                "I work very long hours.", "new_inference_probabilistic", new_inferences=inferences
            )
        )
        await run_turn(db, session.id, "Your schedule?", ollama)
    msgs = await get_messages(db, session.id)
    assistant = next(m for m in msgs if m.role == "assistant")
    assert assistant.ungrounded_implications is not None


async def test_new_inference_probabilistic_does_not_create_db_row(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Probabilistic inferences require user confirmation before being stored."""
    inferences = [
        {
            "inference_type": "probabilistic",
            "statement": "Works long hours",
            "derivation": "occupation=surgeon",
            "source_fact_ids": [],
            "source_inference_ids": [],
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn(
                "I work very long hours.", "new_inference_probabilistic", new_inferences=inferences
            )
        )
        await run_turn(db, session.id, "Your schedule?", ollama)
    stored = await get_inferences(db, character.id)
    assert stored == []


async def test_decision_stored_for_every_completed_turn(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Each completed turn — regardless of verdict — stores exactly one decision."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Turn one", ollama)
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Turn two", ollama)
    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 2


async def test_run_turn_falls_back_to_pass_on_evaluator_parse_error(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """EvaluatorParseError (e.g. unescaped quote in LLM output) must not crash the request."""
    # Evaluator LLM returns JSON with an unescaped " — invalid JSON that json.loads rejects.
    malformed_eval = make_ollama_ndjson('{"verdict": "pass", "decision_log": "height 5\'6" tall"}')
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[_mock_ok("I am fine."), httpx.Response(200, content=malformed_eval)]
        )
        content, _, _, result, _ = await run_turn(db, session.id, "How are you?", ollama)

    assert content == "I am fine."
    assert result.verdict == "pass"
    msgs = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "I am fine." for m in msgs)


# ---------------------------------------------------------------------------
# Phase 3 additions — inference loading and depth capping
# ---------------------------------------------------------------------------


async def test_run_turn_loads_inferences_for_character(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """When active inferences exist, run_turn includes them in the system message."""
    await create_fact(db, character_id=character.id, key="age", value="33")
    from memories.database import create_inference

    await create_inference(
        db,
        character_id=character.id,
        statement="Alice was born in 1993",
        derivation="age=33",
    )
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    # First call (calls[0]) is the character LLM — check its system message
    body = json.loads(route.calls[0].request.content)
    system_content = body["messages"][0]["content"]
    assert "Alice was born in 1993" in system_content


async def test_run_turn_system_message_includes_inference_text(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """System message content contains a known inference statement."""
    from memories.database import create_inference

    await create_inference(
        db,
        character_id=character.id,
        statement="Alice likely works weekends",
        derivation="occupation=surgeon",
    )
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Work-life balance?", ollama)

    body = json.loads(route.calls[0].request.content)
    system_msg = body["messages"][0]["content"]
    assert "Alice likely works weekends" in system_msg


async def test_lazy_inference_depth_computed_before_storing(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Lazy-discovered logical inference with an existing source at depth 2 is stored at depth 3."""
    from memories.database import create_inference

    existing = await create_inference(
        db,
        character_id=character.id,
        statement="Existing inference at depth 2",
        derivation="d",
        depth=2,
    )
    new_inf_payload = [
        {
            "inference_type": "logical",
            "statement": "Derived from depth-2 inference",
            "derivation": "from existing",
            "source_fact_ids": [],
            "source_inference_ids": [existing.id],
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn(
                "Derived statement.", "new_inference_logical", new_inferences=new_inf_payload
            )
        )
        await run_turn(db, session.id, "Tell me more", ollama)

    stored = await get_inferences(db, character.id)
    new_inf = next((i for i in stored if "Derived from depth-2" in i.statement), None)
    assert new_inf is not None
    assert new_inf.depth == 3


async def test_lazy_inference_at_max_depth_is_stored(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Inference at exactly MAX_INFERENCE_DEPTH is stored (not discarded)."""
    from memories.database import create_inference
    from memories.services.inference_service import MAX_INFERENCE_DEPTH

    existing = await create_inference(
        db,
        character_id=character.id,
        statement="Source at max-1 depth",
        derivation="d",
        depth=MAX_INFERENCE_DEPTH - 1,
    )
    new_inf_payload = [
        {
            "inference_type": "logical",
            "statement": "At exactly max depth",
            "derivation": "from source",
            "source_fact_ids": [],
            "source_inference_ids": [existing.id],
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn(
                "At the limit.", "new_inference_logical", new_inferences=new_inf_payload
            )
        )
        await run_turn(db, session.id, "Depth test", ollama)

    stored = await get_inferences(db, character.id)
    assert any("At exactly max depth" in i.statement for i in stored)


async def test_lazy_inference_exceeding_depth_cap_not_stored(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Inference whose depth would exceed MAX_INFERENCE_DEPTH is silently discarded."""
    from memories.database import create_inference
    from memories.services.inference_service import MAX_INFERENCE_DEPTH

    existing = await create_inference(
        db,
        character_id=character.id,
        statement="Source at max depth",
        derivation="d",
        depth=MAX_INFERENCE_DEPTH,
    )
    new_inf_payload = [
        {
            "inference_type": "logical",
            "statement": "Exceeds the cap",
            "derivation": "from source",
            "source_fact_ids": [],
            "source_inference_ids": [existing.id],
        }
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_mock_turn(
                "Too deep.", "new_inference_logical", new_inferences=new_inf_payload
            )
        )
        await run_turn(db, session.id, "Depth exceeded", ollama)

    stored = await get_inferences(db, character.id)
    assert not any("Exceeds the cap" in i.statement for i in stored)


async def test_evaluator_called_with_inferences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """The second Ollama call (evaluator) includes active inference statements in its prompt."""
    from memories.database import create_inference

    await create_inference(
        db,
        character_id=character.id,
        statement="Alice is probably left-handed",
        derivation="occupation=surgeon, handedness patterns",
    )
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Tell me about yourself", ollama)

    # calls[1] is the evaluator call
    eval_body = json.loads(route.calls[1].request.content)
    eval_prompt = eval_body["messages"][1]["content"]  # user message has the evaluator prompt
    assert "Alice is probably left-handed" in eval_prompt


# ---------------------------------------------------------------------------
# Phase 5 additions — experience retrieval, active set, and experience_update
# ---------------------------------------------------------------------------


async def _insert_experience(
    db: aiosqlite.Connection,
    character_id: int,
    session_id: int,
    statement: str = "User lives in Chicago",
    source: str = "told_by_user",
) -> object:
    blob = _embedding_to_blob([1.0, 0.0, 0.0, 0.0])
    return await create_experience(
        db,
        character_id=character_id,
        session_id=session_id,
        statement=statement,
        source=source,
        embedding=blob,
    )


async def test_run_turn_retrieves_experiences_for_user_message(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experience(db, character.id, session.id)
    with respx.mock:
        embed_route = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)
    assert embed_route.called


async def test_run_turn_adds_retrieved_experiences_to_active_set(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experience(db, character.id, session.id)
    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)
    active = get_active_experiences(session.id)
    assert len(active) > 0


async def test_run_turn_no_embed_call_when_no_experiences_in_db(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        embed_route = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response())
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)
    assert not embed_route.called


async def test_run_turn_includes_active_experiences_in_system_prompt(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experience(db, character.id, session.id, "User lives in Chicago")
    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)
    body = json.loads(route.calls[0].request.content)
    system_content = body["messages"][0]["content"]
    assert "User lives in Chicago" in system_content


async def test_run_turn_cold_start_embeds_previous_journal(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import create_session as _cs
    from memories.database import update_session_closing_journal

    await update_session_closing_journal(db, session.id, "A memorable conversation.")
    session2 = await _cs(db, character_id=character.id)
    await _insert_experience(db, character.id, session.id)

    with respx.mock:
        embed_route = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session2.id, "Hi there", ollama)

    # Should be called at least once for cold-start journal embedding
    assert embed_route.called
    bodies = [json.loads(c.request.content) for c in embed_route.calls]
    inputs = [b.get("input", "") for b in bodies]
    assert any("A memorable conversation." in inp for inp in inputs)


async def test_run_turn_cold_start_seeds_active_experiences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import create_session as _cs
    from memories.database import update_session_closing_journal

    await update_session_closing_journal(db, session.id, "Journal.")
    session2 = await _cs(db, character_id=character.id)
    await _insert_experience(db, character.id, session.id)

    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session2.id, "First message", ollama)

    active = get_active_experiences(session2.id)
    assert len(active) > 0


async def test_run_turn_no_cold_start_when_no_previous_session(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    # No previous session with closing journal — cold-start should not fire
    await _insert_experience(db, character.id, session.id)
    with respx.mock:
        embed_route = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    # embed IS called for user-message retrieval, but NOT for cold start journal
    call_inputs = [json.loads(c.request.content).get("input", "") for c in embed_route.calls]
    assert not any("journal" in inp.lower() for inp in call_inputs)


async def test_run_turn_cold_start_only_fires_on_first_turn(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import create_session as _cs
    from memories.database import update_session_closing_journal

    await update_session_closing_journal(db, session.id, "Previous session journal.")
    session2 = await _cs(db, character_id=character.id)
    await _insert_experience(db, character.id, session.id)

    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session2.id, "Turn one", ollama)

    with respx.mock:
        embed_route2 = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session2.id, "Turn two", ollama)

    # On the second turn, embed is called at most once (for user message), not twice
    for call in embed_route2.calls:
        body = json.loads(call.request.content)
        assert "Previous session journal." not in body.get("input", "")


async def test_run_turn_active_set_accumulates_across_turns(
    db: aiosqlite.Connection,
    character: Character,
    session: Session,
    ollama: OllamaClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blob = _embedding_to_blob([1.0, 0.0, 0.0, 0.0])
    exp1 = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Experience one",
        source="told_by_user",
        embedding=blob,
    )
    blob2 = _embedding_to_blob([0.0, 1.0, 0.0, 0.0])
    exp2 = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Experience two",
        source="observed",
        embedding=blob2,
    )

    # Limit to 1 retrieval per turn so each turn fetches a distinct experience.
    # Patching the name inside chat_service (where it was imported) is required
    # because TOP_K_EXPERIENCES is a module-level constant evaluated at import time.
    monkeypatch.setattr("memories.services.chat_service.TOP_K_EXPERIENCES", 1)

    # Turn 1: query vector aligns with exp1 → retrieves exp1
    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "First", ollama)

    # Turn 2: query vector aligns with exp2 → exp1 already active, retrieves exp2
    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([0.0, 1.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Second", ollama)

    active = get_active_experiences(session.id)
    active_ids = {e.id for e in active}
    assert exp1.id in active_ids and exp2.id in active_ids


async def test_run_turn_experience_update_deletes_experience_from_db(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    exp = await _insert_experience(db, character.id, session.id)
    exp_updates = [{"contradicted_experience_id": exp.id, "description": "Now in New York"}]

    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("We are in New York."),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson(
                        "experience_update", experience_updates=exp_updates
                    ),
                ),
            ]
        )
        await run_turn(db, session.id, "Where are we?", ollama)

    remaining = await get_experiences(db, character.id)
    assert not any(e.id == exp.id for e in remaining)


async def test_run_turn_experience_update_removes_from_active_set(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    exp = await _insert_experience(db, character.id, session.id)
    exp_updates = [{"contradicted_experience_id": exp.id, "description": "Moved to New York"}]

    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Now in New York."),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson(
                        "experience_update", experience_updates=exp_updates
                    ),
                ),
            ]
        )
        await run_turn(db, session.id, "Location?", ollama)

    active = get_active_experiences(session.id)
    assert not any(e.id == exp.id for e in active)


async def test_run_turn_experience_update_invalid_id_logs_warning_but_does_not_raise(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    exp_updates = [{"contradicted_experience_id": 99999, "description": "Non-existent"}]

    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=httpx.Response(200, content=make_embed_response()))
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Response."),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson(
                        "experience_update", experience_updates=exp_updates
                    ),
                ),
            ]
        )
        # Should not raise even though experience ID doesn't exist
        await run_turn(db, session.id, "Hello", ollama)


async def test_run_turn_passes_active_experiences_to_evaluator(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experience(db, character.id, session.id, "User lives in Chicago")
    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    # calls[1] is the evaluator call
    eval_body = json.loads(route.calls[1].request.content)
    eval_prompt = eval_body["messages"][1]["content"]
    assert "User lives in Chicago" in eval_prompt


async def test_run_turn_experience_update_promotes_logical_inferences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    exp = await _insert_experience(db, character.id, session.id)
    new_inferences = [
        {
            "inference_type": "logical",
            "statement": "Character is moving cities",
            "derivation": "mentioned New York",
            "source_fact_ids": [],
            "source_inference_ids": [],
        }
    ]
    exp_updates = [{"contradicted_experience_id": exp.id, "description": "Now in New York"}]

    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Now in New York."),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson(
                        "experience_update",
                        new_inferences=new_inferences,
                        experience_updates=exp_updates,
                    ),
                ),
            ]
        )
        await run_turn(db, session.id, "Where are we?", ollama)

    stored = await get_inferences(db, character.id)
    assert any("Character is moving cities" in i.statement for i in stored)


async def test_run_turn_experience_update_does_not_promote_probabilistic_inferences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Probabilistic inferences are silently discarded when verdict is experience_update."""
    exp = await _insert_experience(db, character.id, session.id)
    new_inferences = [
        {
            "inference_type": "probabilistic",
            "statement": "Character seems nostalgic about Chicago",
            "derivation": "tone of response",
            "source_fact_ids": [],
            "source_inference_ids": [],
        }
    ]
    exp_updates = [{"contradicted_experience_id": exp.id, "description": "Now in New York"}]

    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Now in New York."),
                httpx.Response(
                    200,
                    content=make_evaluator_ndjson(
                        "experience_update",
                        new_inferences=new_inferences,
                        experience_updates=exp_updates,
                    ),
                ),
            ]
        )
        await run_turn(db, session.id, "Where are we?", ollama)

    stored = await get_inferences(db, character.id)
    assert not any("nostalgic about Chicago" in i.statement for i in stored)


async def test_run_turn_returns_five_tuple_with_scores_dict(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=httpx.Response(200, content=make_embed_response()))
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        result = await run_turn(db, session.id, "Hello", ollama)

    assert len(result) == 5
    *_, scores = result
    assert isinstance(scores, dict)
