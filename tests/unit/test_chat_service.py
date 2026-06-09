"""Unit tests for memories.services.chat_service.run_turn.

These tests use a real in-memory SQLite database (via the shared ``db``
fixture) and a mocked Ollama HTTP layer (via respx).  They test the
orchestration contract of run_turn — what it reads, what it writes, and in
what order — not the HTTP or SQL layers themselves.

Every successful run_turn call makes THREE /api/chat requests (mocked via _CHAT_URL):
  calls[0] — extractor LLM (Phase 6 fact extraction)
  calls[1] — character LLM
  calls[2] — evaluator LLM
All tests that complete a turn successfully must therefore mock all three calls.

The embed call (/api/embed, mocked via _EMBED_URL) fires concurrently with calls[0]
via asyncio.gather when experiences are stored in the DB.  Tests that need the embed
call must mock _EMBED_URL separately.

DB reads (character, facts, inferences, history, segment, turn_id) are all fetched in
a single asyncio.gather after session validation.  history and segment are therefore
loaded before the current user message is stored — the current turn's user content
reaches the LLM only via the manually appended base_messages entry, not via history.
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
    create_inference,
    end_session,
    get_decisions,
    get_experiences,
    get_facts,
    get_inferences,
    get_messages,
)
from memories.exceptions import NotFoundError, SessionEndedError
from memories.models import Character, Session
from memories.services.chat_service import MAX_CONTRADICTION_RETRIES, run_turn
from memories.services.evaluator import EvaluatorResult
from memories.services.experience_service import get_active_experiences
from memories.services.extraction_service import ExtractionResult
from memories.services.inference_service import MAX_INFERENCE_DEPTH
from memories.services.ollama_client import OllamaClient, OllamaConnectionError
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_embed_response,
    make_evaluator_ndjson,
    make_extractor_ndjson,
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


def _mock_extractor() -> httpx.Response:
    """Return an empty-extraction response for the Phase 6 extractor call (calls[0])."""
    return httpx.Response(200, content=make_extractor_ndjson())


def _mock_turn(
    character_content: str = "I am fine, thank you.",
    evaluator_verdict: str = "pass",
    new_inferences: list[dict] | None = None,
    violations: list[dict] | None = None,
) -> list[httpx.Response]:
    """Return side_effect list for one complete turn (extractor + character + evaluator)."""
    return [
        _mock_extractor(),
        _mock_ok(character_content),
        _mock_eval(evaluator_verdict, new_inferences, violations),
    ]


# ---------------------------------------------------------------------------
# Tests: what run_turn sends to the character LLM (calls[1])
# ---------------------------------------------------------------------------


async def test_system_message_is_first_in_ollama_request(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    body = json.loads(route.calls[1].request.content)
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

    body = json.loads(route.calls[1].request.content)
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

    body = json.loads(route.calls[1].request.content)
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

    body = json.loads(route.calls[1].request.content)
    assert body["messages"][-1]["role"] == "user"
    assert body["messages"][-1]["content"] == "My specific question"


async def test_facts_reflected_in_system_message(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await create_fact(db, character_id=character.id, key="birthplace", value="Reykjavik")

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Where are you from?", ollama)

    body = json.loads(route.calls[1].request.content)
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
    # Three calls: extractor then character then evaluator
    assert len(route.calls) == 3


async def test_evaluator_called_with_character_response(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Character says hi"))
        await run_turn(db, session.id, "Hello", ollama)
    # The evaluator prompt (calls[2]) should contain the character's response
    eval_body = json.loads(route.calls[2].request.content)
    user_msgs = [m for m in eval_body["messages"] if m["role"] == "user"]
    assert any("Character says hi" in m["content"] for m in user_msgs)


async def test_run_turn_returns_evaluator_result(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        _, _, _, eval_result, *_ = await run_turn(db, session.id, "Hello", ollama)
    assert isinstance(eval_result, EvaluatorResult)


async def test_pass_verdict_response_stored_and_returned(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn("Clean reply."))
        content, _, _turn_id, *_ = await run_turn(db, session.id, "Hello", ollama)
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
                _mock_extractor(),  # extractor (call 0)
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
                _mock_extractor(),  # extractor (call 0)
                _mock_ok("Wrong answer."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Correct answer."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "Hello", ollama)
    # 5 calls total: extractor, char1, eval1, char2, eval2
    assert len(route.calls) == 5


async def test_contradiction_second_call_messages_include_system_note(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "wrong city", "suggested_fact": None}
    ]
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_extractor(),  # extractor (call 0)
                _mock_ok("Wrong."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Right."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "Hello", ollama)
    # calls[3] is the second character call (after contradiction)
    body = json.loads(route.calls[3].request.content)
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
                _mock_extractor(),  # extractor (call 0)
                _mock_ok("Bad."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Good."),
                _mock_eval("pass"),
            ]
        )
        content, _, _, eval_result, *_ = await run_turn(db, session.id, "Hello", ollama)
    assert content == "Good."
    assert eval_result.verdict == "pass"
    assert len(route.calls) == 5


async def test_contradiction_loop_final_response_is_stored(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "error", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_extractor(),  # extractor (call 0)
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
    contradiction_violation = [
        {"type": "contradiction", "description": "always wrong", "suggested_fact": None}
    ]
    # One extractor call + (max_retries + 1) pairs of [character, evaluator] calls
    side_effects: list[httpx.Response] = [_mock_extractor()]
    for _ in range(MAX_CONTRADICTION_RETRIES + 1):
        side_effects.append(_mock_ok("Still wrong."))
        side_effects.append(_mock_eval("contradiction", violations=contradiction_violation))

    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=side_effects)
        content, _, _, eval_result, *_ = await run_turn(db, session.id, "Hello", ollama)

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
                _mock_extractor(),  # extractor (call 0)
                _mock_ok("Wrong 1."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Wrong 2."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Correct."),
                _mock_eval("pass"),
            ]
        )
        _, _, _, eval_result, *_ = await run_turn(db, session.id, "Hello", ollama)
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
            side_effect=[
                _mock_extractor(),
                _mock_ok("I am fine."),
                httpx.Response(200, content=malformed_eval),
            ]
        )
        content, _, _, result, _, _ = await run_turn(db, session.id, "How are you?", ollama)

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
    await create_inference(
        db,
        character_id=character.id,
        statement="Alice was born in 1993",
        derivation="age=33",
    )
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    # calls[1] is the character LLM — check its system message
    body = json.loads(route.calls[1].request.content)
    system_content = body["messages"][0]["content"]
    assert "Alice was born in 1993" in system_content


async def test_run_turn_system_message_includes_inference_text(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """System message content contains a known inference statement."""
    await create_inference(
        db,
        character_id=character.id,
        statement="Alice likely works weekends",
        derivation="occupation=surgeon",
    )
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Work-life balance?", ollama)

    body = json.loads(route.calls[1].request.content)
    system_msg = body["messages"][0]["content"]
    assert "Alice likely works weekends" in system_msg


async def test_lazy_inference_depth_computed_before_storing(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Lazy-discovered logical inference with an existing source at depth 2 is stored at depth 3."""
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
    """The third Ollama call (evaluator) includes active inference statements in its prompt."""
    await create_inference(
        db,
        character_id=character.id,
        statement="Alice is probably left-handed",
        derivation="occupation=surgeon, handedness patterns",
    )
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Tell me about yourself", ollama)

    # calls[2] is the evaluator call
    eval_body = json.loads(route.calls[2].request.content)
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
    body = json.loads(route.calls[1].request.content)
    system_content = body["messages"][0]["content"]
    assert "User lives in Chicago" in system_content


async def test_run_turn_embed_query_uses_user_message(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """The embed call uses the user's message as the query, not a journal or other text."""
    await _insert_experience(db, character.id, session.id)
    with respx.mock:
        embed_route = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello there", ollama)

    inputs = [json.loads(c.request.content).get("input", "") for c in embed_route.calls]
    assert any("Hello there" in inp for inp in inputs)


async def test_run_turn_embed_and_extractor_both_called_when_experiences_present(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """With experiences in DB, embed (retrieve) and extractor (LLM) both fire before character."""
    await _insert_experience(db, character.id, session.id)
    with respx.mock:
        embed_route = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        chat_route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)
    assert embed_route.called
    assert len(chat_route.calls) == 3  # extractor + character + evaluator


async def test_run_turn_active_set_reflects_current_turn_only(
    db: aiosqlite.Connection,
    character: Character,
    session: Session,
    ollama: OllamaClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Active set after each turn contains only what scored well that turn, not prior turns."""
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

    monkeypatch.setattr("memories.services.chat_service.TOP_K_EXPERIENCES", 1)

    # Turn 1: query aligns with exp1
    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "First", ollama)

    # Turn 2: query aligns with exp2 — active set should be replaced, not accumulated
    with respx.mock:
        respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([0.0, 1.0, 0.0, 0.0]))
        )
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Second", ollama)

    active_ids = {e.id for e in get_active_experiences(session.id)}
    assert exp2.id in active_ids  # aligns with this turn's query
    assert exp1.id not in active_ids  # does not align — not carried over from turn 1


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
                _mock_extractor(),
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
                _mock_extractor(),
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
                _mock_extractor(),
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

    # calls[2] is the evaluator call
    eval_body = json.loads(route.calls[2].request.content)
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
                _mock_extractor(),
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
                _mock_extractor(),
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

    assert len(result) == 6
    *_, scores, _ = result
    assert isinstance(scores, dict)


# ---------------------------------------------------------------------------
# Phase 6 additions — fact extraction pre-turn hook
# ---------------------------------------------------------------------------


async def test_run_turn_calls_extractor_before_character_llm(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Extractor Ollama call precedes character Ollama call (respx call order)."""
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson()),  # extractor (call 0)
                _mock_ok("I hear you."),  # character (call 1)
                _mock_eval("pass"),  # evaluator (call 2)
            ]
        )
        await run_turn(db, session.id, "Hello", ollama)

    # Phase 6: extractor + character + evaluator = 3 calls
    assert len(route.calls) == 3


async def test_run_turn_auto_adds_tier1_facts(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """new_facts in extractor result → create_fact called for each before character call."""
    new_fact = {
        "key": "meeting_location",
        "value": "Chicago",
        "category": "setting",
        "mutability": "low",
        "source_quote": "We're meeting in Chicago",
    }
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson(new_facts=[new_fact])),
                _mock_ok("See you in Chicago."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "We're meeting in Chicago!", ollama)

    facts = await get_facts(db, character.id)
    assert any(f.key == "meeting_location" and f.value == "Chicago" for f in facts)


async def test_run_turn_auto_updates_tier2_facts(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """fact_updates in extractor result → update_fact called for each before character call."""
    existing = await create_fact(db, character_id=character.id, key="home_city", value="Reykjavik")
    update = {
        "fact_id": existing.id,
        "key": "home_city",
        "old_value": "Reykjavik",
        "new_value": "Chicago",
        "source_quote": "I moved to Chicago",
    }
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson(fact_updates=[update])),
                _mock_ok("Chicago is a great city."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "I moved to Chicago last month.", ollama)

    facts = await get_facts(db, character.id)
    updated = next(f for f in facts if f.key == "home_city")
    assert updated.value == "Chicago"


async def test_run_turn_does_not_write_implicit_proposals_to_db(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """implicit_proposals in extractor result → no create_fact or update_fact called for them."""
    proposal = {
        "key": "mood",
        "value": "anxious",
        "category": "user",
        "mutability": "high",
        "source_quote": "feeling off all week",
    }
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson(implicit_proposals=[proposal])),
                _mock_ok("I can hear some tension in your voice."),
                _mock_eval("pass"),
            ]
        )
        result = await run_turn(db, session.id, "I've been feeling off.", ollama)

    facts = await get_facts(db, character.id)
    assert not any(f.key == "mood" for f in facts)
    # The ExtractionResult in the return value should carry the proposals
    assert isinstance(result[-1], ExtractionResult)


async def test_run_turn_passes_tier1_and_tier2_facts_to_character(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Character system prompt includes extracted/updated fact values."""
    new_fact = {
        "key": "meeting_location",
        "value": "Chicago",
        "category": "setting",
        "mutability": "low",
        "source_quote": "We're in Chicago",
    }
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson(new_facts=[new_fact])),
                _mock_ok("See you in Chicago."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "We're in Chicago!", ollama)

    # calls[1] is the character LLM call after extraction
    assert len(route.calls) == 3
    char_body = json.loads(route.calls[1].request.content)
    system_content: str = char_body["messages"][0]["content"]
    assert "Chicago" in system_content


async def test_run_turn_character_prompt_does_not_include_implicit_proposals(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """System prompt does not contain proposed-but-unconfirmed values from implicit_proposals."""
    proposal = {
        "key": "current_feeling",
        "value": "very anxious",
        "category": "user",
        "mutability": "high",
        "source_quote": "anxious",
    }
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson(implicit_proposals=[proposal])),
                _mock_ok("Take it easy."),
                _mock_eval("pass"),
            ]
        )
        result = await run_turn(db, session.id, "Feeling anxious today.", ollama)

    assert len(route.calls) == 3
    char_body = json.loads(route.calls[1].request.content)
    system_content: str = char_body["messages"][0]["content"]
    assert "very anxious" not in system_content
    # ExtractionResult should carry the unwritten proposals
    assert isinstance(result[-1], ExtractionResult)
    assert len(result[-1].implicit_proposals) == 1


async def test_run_turn_returns_extraction_result(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """run_turn return value includes the full ExtractionResult."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson()),
                _mock_ok("Hello."),
                _mock_eval("pass"),
            ]
        )
        result = await run_turn(db, session.id, "Hello", ollama)

    assert isinstance(result[-1], ExtractionResult)


async def test_run_turn_on_extraction_failure_continues_with_empty_result(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """ExtractionParseError → turn completes; extraction result is empty."""
    # First response is invalid JSON (triggers ExtractionParseError in extractor)
    invalid_extractor = make_ollama_ndjson("not valid extraction json")
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=invalid_extractor),
                _mock_ok("Still responding."),
                _mock_eval("pass"),
            ]
        )
        result = await run_turn(db, session.id, "Hello", ollama)

    assert isinstance(result[-1], ExtractionResult)
    extraction = result[-1]
    assert extraction.new_facts == []
    assert extraction.fact_updates == []
    assert extraction.implicit_proposals == []


async def test_run_turn_on_ollama_connection_error_during_extraction_continues(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Ollama connection failure during extraction → turn continues; warning logged."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.ConnectError("connection refused"),  # extraction fails
                _mock_ok("Still responding."),
                _mock_eval("pass"),
            ]
        )
        # Phase 6: extraction failure is non-fatal; character call proceeds
        result = await run_turn(db, session.id, "Hello", ollama)

    assert isinstance(result[-1], ExtractionResult)
    assert result[-1].new_facts == []


async def test_run_turn_deduplicates_tier1_facts_that_already_exist(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Extractor proposes a new fact with existing key/category → IntegrityError caught silently."""
    # Pre-existing fact with same key
    await create_fact(db, character_id=character.id, key="occupation", value="surgeon")

    duplicate = {
        "key": "occupation",
        "value": "retired surgeon",
        "category": "character",
        "mutability": "immutable",
        "source_quote": "I used to be a surgeon",
    }
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson(new_facts=[duplicate])),
                _mock_ok("Yes, I was."),
                _mock_eval("pass"),
            ]
        )
        # Should not raise; IntegrityError on the duplicate should be swallowed
        result = await run_turn(db, session.id, "Were you a surgeon?", ollama)

    facts = await get_facts(db, character.id)
    # Original fact must still be there; duplicate (different value) was suppressed
    occupation_facts = [f for f in facts if f.key == "occupation"]
    assert len(occupation_facts) == 1
    # Phase 6: result must include ExtractionResult (fails before implementation)
    assert isinstance(result[-1], ExtractionResult)


async def test_run_turn_empty_extraction_result_does_not_change_facts(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """All three extraction lists empty → no fact writes; turn proceeds normally."""
    await create_fact(db, character_id=character.id, key="occupation", value="surgeon")

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(200, content=make_extractor_ndjson()),  # empty extraction
                _mock_ok("Hello."),
                _mock_eval("pass"),
            ]
        )
        result = await run_turn(db, session.id, "Hello", ollama)

    # Extraction call happened (3 total calls)
    assert len(route.calls) == 3
    facts = await get_facts(db, character.id)
    assert len(facts) == 1
    assert facts[0].key == "occupation"
    assert isinstance(result[-1], ExtractionResult)
    assert result[-1].new_facts == []
