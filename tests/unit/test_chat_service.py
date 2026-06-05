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

from memories.database import create_fact, get_decisions, get_inferences, get_messages
from memories.exceptions import NotFoundError, SessionEndedError
from memories.models import Character, Session
from memories.services.chat_service import run_turn
from memories.services.evaluator import EvaluatorResult
from memories.services.ollama_client import OllamaClient, OllamaConnectionError
from tests.unit.conftest import OLLAMA_BASE_URL, make_evaluator_ndjson, make_ollama_ndjson

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


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
        _, _, _, eval_result = await run_turn(db, session.id, "Hello", ollama)
    assert isinstance(eval_result, EvaluatorResult)


async def test_pass_verdict_response_stored_and_returned(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Clean reply."))
        content, _, _turn_id, _ = await run_turn(db, session.id, "Hello", ollama)
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
        content, _, _, eval_result = await run_turn(db, session.id, "Hello", ollama)
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
        content, _, _, eval_result = await run_turn(db, session.id, "Hello", ollama)

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
        _, _, _, eval_result = await run_turn(db, session.id, "Hello", ollama)
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
        content, _, _, result = await run_turn(db, session.id, "How are you?", ollama)

    assert content == "I am fine."
    assert result.verdict == "pass"
    msgs = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "I am fine." for m in msgs)
