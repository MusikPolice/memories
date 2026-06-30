"""Unit tests for memories.services.chat_service.run_turn.

These tests use a real in-memory SQLite database (via the shared ``db``
fixture) and a mocked Ollama HTTP layer (via respx).  They test the
orchestration contract of run_turn — what it reads, what it writes, and in
what order — not the HTTP or SQL layers themselves.

Every successful run_turn call makes THREE /api/chat requests (mocked via _CHAT_URL):
  calls[0] — World Builder LLM (Step 4)
  calls[1] — character LLM
  calls[2] — evaluator LLM (tool-calling, non-streaming; may be more than one HTTP
              request if nudge/retry behaviour is exercised)
All tests that complete a turn successfully must therefore mock all three calls.
calls[0] may be more than one HTTP request if the World Builder's mock simulates
multiple tool-call rounds — the default `_mock_world_builder()` helper used by most
tests is a single no-op round, preserving the "three calls total" assumption for
every test that does not specifically exercise World Builder fact-writing.

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
    get_facts,
    get_inferences,
    get_messages,
    set_facts,
)
from memories.exceptions import NotFoundError, SessionEndedError
from memories.models import Character, Session
from memories.services.chat_service import MAX_CONTRADICTION_RETRIES, run_turn
from memories.services.evaluator import EvaluatorResult
from memories.services.experience_service import get_active_experiences
from memories.services.ollama_client import OllamaClient, OllamaConnectionError
from memories.services.sse_events import SSEEvent
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_embed_response,
    make_multi_tool_call_response,
    make_ollama_ndjson,
    make_plain_tool_response,
    make_tool_call_response,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"


def _mock_ok(content: str = "I am fine, thank you.") -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(content))


def _mock_eval(
    verdict: str = "pass",
    violations: list[dict] | None = None,
) -> httpx.Response:
    if verdict == "contradiction":
        description = (violations or [{}])[0].get("description", "contradiction")
        return httpx.Response(
            200,
            content=make_tool_call_response("report_contradiction", {"description": description}),
        )
    return httpx.Response(200, content=make_tool_call_response("report_pass", {}))


def _mock_world_builder() -> httpx.Response:
    """Return a no-op World Builder response (no tool call) for calls[0]."""
    return httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))


def _mock_turn(
    character_content: str = "I am fine, thank you.",
    evaluator_verdict: str = "pass",
    violations: list[dict] | None = None,
) -> list[httpx.Response]:
    """Return side_effect list for one complete turn (World Builder + character + evaluator)."""
    return [
        _mock_world_builder(),
        _mock_ok(character_content),
        _mock_eval(evaluator_verdict, violations),
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
    # Three calls: World Builder then character then evaluator
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
    assert decisions[0].tool_name == "report_pass"


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
                _mock_world_builder(),  # World Builder (call 0)
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
                _mock_world_builder(),  # World Builder (call 0)
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
                _mock_world_builder(),  # World Builder (call 0)
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
                _mock_world_builder(),  # World Builder (call 0)
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
                _mock_world_builder(),  # World Builder (call 0)
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
    # One World Builder call + (max_retries + 1) pairs of [character, evaluator] calls
    side_effects: list[httpx.Response] = [_mock_world_builder()]
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
                _mock_world_builder(),  # World Builder (call 0)
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


# ---------------------------------------------------------------------------
# Phase 3 additions — inference loading
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


async def test_run_turn_embed_and_world_builder_both_called_when_experiences_present(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """With experiences in DB, embed (retrieve) and World Builder (LLM) both fire."""
    await _insert_experience(db, character.id, session.id)
    with respx.mock:
        embed_route = respx.post(_EMBED_URL).mock(
            return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
        )
        chat_route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)
    assert embed_route.called
    assert len(chat_route.calls) == 3  # World Builder + character + evaluator


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


# ---------------------------------------------------------------------------
# Step 4 additions — World Builder pre-turn hook
# ---------------------------------------------------------------------------


def _mock_world_builder_writes(path: str, value: str) -> list[httpx.Response]:
    """side_effect entries for a World Builder pass that writes one fact via author_set_facts.

    Two HTTP round-trips: the tool call itself, then the terminal plain reply that
    chat_with_tools requires before it will stop looping.
    """
    return [
        httpx.Response(
            200,
            content=make_tool_call_response(
                "author_set_facts", {"facts": [{"path": path, "value": value}]}
            ),
        ),
        httpx.Response(200, content=make_plain_tool_response("Done.")),
    ]


async def test_run_turn_world_builder_runs_before_character_llm(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """calls[0] is the World Builder (uses tools); calls[1] is the character LLM (no tools)."""
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    wb_body = json.loads(route.calls[0].request.content)
    assert wb_body["tools"][0]["function"]["name"] == "author_set_facts"
    char_body = json.loads(route.calls[1].request.content)
    assert "tools" not in char_body


async def test_run_turn_world_builder_writes_fact_to_character_facts_blob(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.State-Of-Mind.Mood", "Anxious"),
                _mock_ok("I hear you."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "I'm feeling really anxious", ollama)

    facts = await get_facts(db, character.id)
    assert facts["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Anxious"


async def test_run_turn_world_builder_fact_appears_in_character_system_prompt(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.State-Of-Mind.Mood", "Anxious"),
                _mock_ok("I hear you."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "I'm feeling really anxious", ollama)

    # calls[2] is the character LLM call (calls[0]/[1] are the World Builder's two rounds)
    char_body = json.loads(route.calls[2].request.content)
    system_content = char_body["messages"][0]["content"]
    assert "Anxious" in system_content


async def test_run_turn_world_builder_preserves_facts_not_touched_by_this_turn(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    name_blob = {"Character": {"Identity": {"Name": {"Value": "Alice"}}}}
    await set_facts(db, character_id=character.id, blob=name_blob)
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    facts = await get_facts(db, character.id)
    assert facts["Character"]["Identity"]["Name"]["Value"] == "Alice"


async def test_run_turn_world_builder_failure_continues_with_unchanged_facts(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    name_blob = {"Character": {"Identity": {"Name": {"Value": "Alice"}}}}
    await set_facts(db, character_id=character.id, blob=name_blob)
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.ConnectError("refused"),
                _mock_ok("Hi there."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "Hello", ollama)

    char_body = json.loads(route.calls[1].request.content)
    system_content = char_body["messages"][0]["content"]
    assert "Alice" in system_content


async def test_run_turn_world_builder_decision_logged_when_facts_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.Identity.Name", "Beatrice"),
                _mock_ok("Hi Beatrice."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "My name is Beatrice", ollama)

    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 2
    assert any(d.pass_name == "world_builder" for d in decisions)


async def test_run_turn_no_world_builder_decision_when_no_facts_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 1
    assert decisions[0].pass_name == "character_evaluator"


async def test_run_turn_character_llm_request_has_no_tools_key(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    char_body = json.loads(route.calls[1].request.content)
    assert "tools" not in char_body


async def test_run_turn_world_builder_sidechannel_emitted_via_on_event(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.Identity.Name", "Beatrice"),
                _mock_ok("Hi Beatrice."),
                _mock_eval("pass"),
            ]
        )
        await run_turn(db, session.id, "My name is Beatrice", ollama, on_event=_on_event)

    wb_events = [e for e in events if e.data.get("type") == "world_builder_applied"]
    assert len(wb_events) == 1


async def test_run_turn_world_builder_no_sidechannel_when_no_facts_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama, on_event=_on_event)

    assert not any(e.data.get("type") == "world_builder_applied" for e in events)


# ---------------------------------------------------------------------------
# Step 5 additions — evaluator tool-call integration
# ---------------------------------------------------------------------------


async def test_run_turn_evaluator_fluid_set_fact_writes_to_blob(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                _mock_ok("I am feeling anxious."),
                httpx.Response(
                    200,
                    content=make_multi_tool_call_response(
                        [
                            (
                                "set_fact",
                                {
                                    "path": "Character.State-Of-Mind.Mood",
                                    "value": "Anxious",
                                },
                            ),
                            ("report_pass", {}),
                        ]
                    ),
                ),
            ]
        )
        await run_turn(db, session.id, "How are you feeling?", ollama)

    facts = await get_facts(db, character.id)
    assert facts["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Anxious"


async def test_run_turn_evaluator_fluid_set_fact_emits_sidechannel(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                _mock_ok("I am feeling anxious."),
                httpx.Response(
                    200,
                    content=make_multi_tool_call_response(
                        [
                            (
                                "set_fact",
                                {
                                    "path": "Character.State-Of-Mind.Mood",
                                    "value": "Anxious",
                                },
                            ),
                            ("report_pass", {}),
                        ]
                    ),
                ),
            ]
        )
        await run_turn(db, session.id, "How are you?", ollama, on_event=_on_event)

    sc_events = [e for e in events if e.event == "sidechannel"]
    assert any(e.data.get("type") == "fact_update_fluid" for e in sc_events)


async def test_run_turn_evaluator_propose_inference_writes_row(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                _mock_ok("I work very long hours as a surgeon."),
                httpx.Response(
                    200,
                    content=make_multi_tool_call_response(
                        [
                            (
                                "propose_inference",
                                {
                                    "statement": "Alice works long hours",
                                    "derivation": "occupation=surgeon",
                                },
                            ),
                            ("report_pass", {}),
                        ]
                    ),
                ),
            ]
        )
        await run_turn(db, session.id, "Your schedule?", ollama)

    stored = await get_inferences(db, character.id)
    assert any(i.statement == "Alice works long hours" for i in stored)


async def test_run_turn_decisions_no_longer_include_evaluator_verdict_tool_name(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Regression guard: no decision row should ever have tool_name='evaluator_verdict'."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    decisions = await get_decisions(db, session.id)
    assert not any(d.tool_name == "evaluator_verdict" for d in decisions)
