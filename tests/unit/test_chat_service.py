"""Unit tests for memories.services.chat_service.run_turn.

These tests use a real in-memory SQLite database (via the shared ``db``
fixture) and a mocked Ollama HTTP layer (via respx).  They test the
orchestration contract of run_turn — what it reads, what it writes, and in
what order — not the HTTP or SQL layers themselves.

Every successful run_turn call makes THREE /api/chat requests (mocked via _CHAT_URL):
  calls[0] — World Builder LLM (Step 4)
  calls[1] — character LLM (tool-calling, non-streaming; default _mock_ok() is a
              single no-tool-call round so the "three calls total" assumption is
              preserved for tests that don't exercise require_fact)
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

import asyncio
import json

import aiosqlite
import httpx
import pytest
import respx

import memories.services.tool_gate as tool_gate_module
from memories.database import (
    _embedding_to_blob,
    create_experience,
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
from memories.services.chat_service import (
    MAX_CONTRADICTION_RETRIES,
    run_contradiction_loop,
    run_turn,
)
from memories.services.evaluator import EvaluatorResult
from memories.services.experience_service import get_active_experiences
from memories.services.ollama_client import OllamaClient, OllamaConnectionError
from memories.services.sse_events import SSEEvent
from memories.services.tool_gate import resolve_gate
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_embed_response,
    make_multi_tool_call_response,
    make_plain_tool_response,
    make_tool_call_response,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"


def _mock_ok(content: str = "I am fine, thank you.") -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response(content))


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
    """calls[0] is the World Builder (uses author_set_facts); calls[1] is the character LLM."""
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    wb_body = json.loads(route.calls[0].request.content)
    assert wb_body["tools"][0]["function"]["name"] == "author_set_facts"


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


async def test_run_turn_character_llm_tools_list_is_require_fact_only(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        route = respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    char_body = json.loads(route.calls[1].request.content)
    assert char_body["tools"][0]["function"]["name"] == "require_fact"
    assert len(char_body["tools"]) == 1


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


# ---------------------------------------------------------------------------
# Step 6 additions — require_fact
# ---------------------------------------------------------------------------


async def test_run_turn_character_require_fact_suspends_and_resumes(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "Sarah")

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Name",
                            "reason": "I need my name to introduce myself",
                            "suggested_value": "Elena",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Hi, I'm Sarah.")),
                _mock_eval("pass"),
            ]
        )
        (content, *_rest), _ = await asyncio.gather(
            run_turn(db, session.id, "What's your name?", ollama, on_event=_on_event),
            _resolver(),
        )

    assert content == "Hi, I'm Sarah."


async def test_run_turn_character_require_fact_writes_value_to_blob(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "Sarah")

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Name",
                            "reason": "Need name",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Hi, I'm Sarah.")),
                _mock_eval("pass"),
            ]
        )
        await asyncio.gather(
            run_turn(db, session.id, "Hello", ollama, on_event=_on_event),
            _resolver(),
        )

    facts = await get_facts(db, character.id)
    assert facts["Character"]["Identity"]["Name"]["Value"] == "Sarah"


async def test_run_turn_character_require_fact_dismiss_leaves_path_unset(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, None)

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Name",
                            "reason": "Need name",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("I'll proceed without it.")),
                _mock_eval("pass"),
            ]
        )
        (content, *_), _ = await asyncio.gather(
            run_turn(db, session.id, "Hello", ollama, on_event=_on_event),
            _resolver(),
        )

    # After a dismiss the model should acknowledge it can't use the value.
    assert content == "I'll proceed without it."
    facts = await get_facts(db, character.id)
    assert "Name" not in facts.get("Character", {}).get("Identity", {})


async def test_run_turn_character_require_fact_logs_decision_with_user_input(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "Sarah")

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Name",
                            "reason": "Need name",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Hi, I'm Sarah.")),
                _mock_eval("pass"),
            ]
        )
        await asyncio.gather(
            run_turn(db, session.id, "Hello", ollama, on_event=_on_event),
            _resolver(),
        )

    decisions = await get_decisions(db, session.id)
    rf_decision = next((d for d in decisions if d.tool_name == "require_fact"), None)
    assert rf_decision is not None
    assert rf_decision.pass_name == "character_llm"
    assert rf_decision.user_input == {"value": "Sarah"}
    assert rf_decision.tool_args == {"path": "Character.Identity.Name", "reason": "Need name"}


async def test_run_turn_character_require_fact_logs_decision_with_suggested_value(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "Sarah")

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Name",
                            "reason": "Need name",
                            "suggested_value": "Elena",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Hi, I'm Sarah.")),
                _mock_eval("pass"),
            ]
        )
        await asyncio.gather(
            run_turn(db, session.id, "Hello", ollama, on_event=_on_event),
            _resolver(),
        )

    decisions = await get_decisions(db, session.id)
    rf_decision = next((d for d in decisions if d.tool_name == "require_fact"), None)
    assert rf_decision is not None
    assert rf_decision.tool_args == {
        "path": "Character.Identity.Name",
        "reason": "Need name",
        "suggested_value": "Elena",
    }


async def test_run_turn_character_require_fact_emits_sidechannel_before_suspension(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "Sarah")

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Name",
                            "reason": "I need my name to respond",
                            "suggested_value": "Elena",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Hi, I'm Sarah.")),
                _mock_eval("pass"),
            ]
        )
        await asyncio.gather(
            run_turn(db, session.id, "Hello", ollama, on_event=_on_event),
            _resolver(),
        )

    rf_events = [e for e in events if e.data.get("type") == "require_fact"]
    assert len(rf_events) == 1
    assert rf_events[0].data["path"] == "Character.Identity.Name"
    assert "reason" in rf_events[0].data
    assert "suggested_value" in rf_events[0].data


async def test_run_turn_gate_removed_after_turn_completes(
    db: aiosqlite.Connection,
    character: Character,
    session: Session,
    ollama: OllamaClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memories.services.chat_service as _cs

    create_calls: list[tuple[int, int]] = []
    _original_create = _cs.create_gate
    monkeypatch.setattr(
        _cs, "create_gate", lambda s, t: (create_calls.append((s, t)), _original_create(s, t))[1]
    )

    with respx.mock:
        respx.post(_CHAT_URL).mock(side_effect=_mock_turn())
        await run_turn(db, session.id, "Hello", ollama)

    assert len(create_calls) == 1  # FAILS without implementation (create_gate never called)
    assert (session.id, 1) not in tool_gate_module._pending


async def test_run_turn_gate_removed_after_character_llm_connection_error(
    db: aiosqlite.Connection,
    character: Character,
    session: Session,
    ollama: OllamaClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import memories.services.chat_service as _cs

    create_calls: list[tuple[int, int]] = []
    _original_create = _cs.create_gate
    monkeypatch.setattr(
        _cs, "create_gate", lambda s, t: (create_calls.append((s, t)), _original_create(s, t))[1]
    )

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.ConnectError("refused"),
            ]
        )
        with pytest.raises(OllamaConnectionError):
            await run_turn(db, session.id, "Hello", ollama)

    assert len(create_calls) == 1  # FAILS without implementation (create_gate never called)
    assert (session.id, 1) not in tool_gate_module._pending


async def test_run_turn_character_require_fact_immutable_already_set_returns_error(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await set_facts(db, character.id, {"Character": {"Identity": {"Name": {"Value": "Alice"}}}})
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {"path": "Character.Identity.Name", "reason": "Need name"},
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Alice it is.")),
                _mock_eval("pass"),
            ]
        )
        content, *_ = await run_turn(db, session.id, "Hello", ollama, on_event=_on_event)

    assert content == "Alice it is."
    assert not any(e.data.get("type") == "require_fact" for e in events)
    assert len(route.calls) == 4


async def test_run_turn_character_require_fact_mutable_path_returns_error(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {"path": "Character.Identity.Occupation", "reason": "Need job"},
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("I'll respond anyway.")),
                _mock_eval("pass"),
            ]
        )
        content, *_ = await run_turn(db, session.id, "Hello", ollama, on_event=_on_event)

    assert content == "I'll respond anyway."
    assert not any(e.data.get("type") == "require_fact" for e in events)
    assert len(route.calls) == 4


async def test_run_turn_character_require_fact_unknown_path_returns_error(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {"path": "Nonexistent.Path", "reason": "Unknown"},
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Proceeding without.")),
                _mock_eval("pass"),
            ]
        )
        content, *_ = await run_turn(db, session.id, "Hello", ollama, on_event=_on_event)

    assert content == "Proceeding without."
    assert not any(e.data.get("type") == "require_fact" for e in events)
    assert len(route.calls) == 4


async def test_run_turn_character_require_fact_enum_path_coerced_case_insensitively(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "athletic")  # lowercase — should be coerced to "Athletic"

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Appearance.Body.Build",
                            "reason": "Need build to describe character",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("I have an athletic build.")),
                _mock_eval("pass"),
            ]
        )
        await asyncio.gather(
            run_turn(db, session.id, "Describe yourself", ollama, on_event=_on_event),
            _resolver(),
        )

    facts = await get_facts(db, character.id)
    assert facts["Character"]["Appearance"]["Body"]["Build"]["Value"] == "Athletic"


async def test_run_turn_character_require_fact_integer_path_coerced_from_string(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    seen = asyncio.Event()

    async def _on_event(ev: SSEEvent) -> None:
        if ev.data.get("type") == "require_fact":
            seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "34")  # string — should be coerced to int 34

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {
                            "path": "Character.Identity.Age",
                            "reason": "Need age",
                        },
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("I am 34 years old.")),
                _mock_eval("pass"),
            ]
        )
        await asyncio.gather(
            run_turn(db, session.id, "How old are you?", ollama, on_event=_on_event),
            _resolver(),
        )

    facts = await get_facts(db, character.id)
    assert facts["Character"]["Identity"]["Age"]["Value"] == 34


async def test_run_turn_character_two_sequential_require_fact_calls_same_turn(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """The single per-turn gate supports multiple sequential suspend/resume cycles."""
    name_seen = asyncio.Event()
    pronouns_seen = asyncio.Event()
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)
        if ev.data.get("type") == "require_fact":
            if ev.data.get("path") == "Character.Identity.Name":
                name_seen.set()
            elif ev.data.get("path") == "Character.Identity.Pronouns":
                pronouns_seen.set()

    async def _resolver() -> None:
        try:
            await asyncio.wait_for(name_seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "Sarah")
        try:
            await asyncio.wait_for(pronouns_seen.wait(), timeout=5.0)
        except TimeoutError:
            return
        resolve_gate(session.id, 1, "she/her")

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {"path": "Character.Identity.Name", "reason": "Need name"},
                    ),
                ),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "require_fact",
                        {"path": "Character.Identity.Pronouns", "reason": "Need pronouns"},
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Hi, I'm Sarah, she/her.")),
                _mock_eval("pass"),
            ]
        )
        (content, *_rest), _ = await asyncio.gather(
            run_turn(db, session.id, "Who are you?", ollama, on_event=_on_event),
            _resolver(),
        )

    assert content == "Hi, I'm Sarah, she/her."
    rf_events = [e for e in events if e.data.get("type") == "require_fact"]
    assert len(rf_events) == 2


# ---------------------------------------------------------------------------
# Step 7 additions — needs_regeneration handling
# ---------------------------------------------------------------------------


async def test_run_contradiction_loop_needs_regeneration_causes_extra_iteration(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    import unittest.mock

    call_count = 0

    async def _fake_evaluator(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                EvaluatorResult(verdict="pass", decision_log="regen", needs_regeneration=True),
                {},
            )
        return EvaluatorResult(verdict="pass", decision_log="ok"), {}

    model = character.current_model_name or character.modelfile_base
    messages = [{"role": "user", "content": "hi"}]
    with (
        unittest.mock.patch("memories.services.chat_service.run_evaluator", _fake_evaluator),
        respx.mock,
    ):
        route = respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Response 1"),
                _mock_ok("Response 2"),
            ]
        )
        content, _, _ = await run_contradiction_loop(
            db, session.id, 1, model, messages, character, {}, "hi", ollama
        )

    assert content == "Response 2"
    assert call_count == 2
    assert route.call_count == 2


async def test_run_contradiction_loop_needs_regeneration_does_not_increment_contradiction_count(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    import unittest.mock

    call_count = 0

    async def _fake_evaluator(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (
                EvaluatorResult(verdict="pass", decision_log="regen", needs_regeneration=True),
                {},
            )
        return EvaluatorResult(verdict="pass", decision_log="ok"), {}

    model = character.current_model_name or character.modelfile_base
    messages = [{"role": "user", "content": "hi"}]
    with (
        unittest.mock.patch("memories.services.chat_service.run_evaluator", _fake_evaluator),
        respx.mock,
    ):
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Response 1"),
                _mock_ok("Response 2"),
            ]
        )
        _, _, eval_result = await run_contradiction_loop(
            db, session.id, 1, model, messages, character, {}, "hi", ollama
        )

    assert call_count == 2
    assert eval_result.contradiction_notifications == []


async def test_run_contradiction_loop_needs_regeneration_uses_updated_blob(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    import unittest.mock

    call_count = 0
    captured_blob: dict = {}

    async def _fake_evaluator(  # type: ignore[no-untyped-def]
        db, character, session_id, turn_id, facts_blob, *args, **kwargs
    ):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            updated = dict(facts_blob)
            updated["_injected"] = True
            return (
                EvaluatorResult(verdict="pass", decision_log="regen", needs_regeneration=True),
                updated,
            )
        captured_blob.update(facts_blob)
        return EvaluatorResult(verdict="pass", decision_log="ok"), facts_blob

    model = character.current_model_name or character.modelfile_base
    messages = [{"role": "user", "content": "hi"}]
    with (
        unittest.mock.patch("memories.services.chat_service.run_evaluator", _fake_evaluator),
        respx.mock,
    ):
        respx.post(_CHAT_URL).mock(
            side_effect=[
                _mock_ok("Response 1"),
                _mock_ok("Response 2"),
            ]
        )
        await run_contradiction_loop(
            db, session.id, 1, model, messages, character, {}, "hi", ollama
        )

    assert captured_blob.get("_injected") is True


async def test_run_contradiction_loop_needs_regeneration_rebuilds_system_prompt(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """The regenerated Character LLM call must see a system prompt built from the
    evaluator's updated facts_blob, not the stale pre-edit prompt (CT-1)."""
    import unittest.mock

    from memories.services.prompt_builder import build_system_prompt

    call_count = 0

    async def _fake_evaluator(  # type: ignore[no-untyped-def]
        db, character, session_id, turn_id, facts_blob, *args, **kwargs
    ):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Simulate an approved edit writing a Name value into the blob.
            updated = {"Character": {"Identity": {"Name": {"Value": "Zaphod"}}}}
            return (
                EvaluatorResult(verdict="pass", decision_log="regen", needs_regeneration=True),
                updated,
            )
        return EvaluatorResult(verdict="pass", decision_log="ok"), facts_blob

    model = character.current_model_name or character.modelfile_base
    base_messages = [
        {"role": "system", "content": build_system_prompt(character, {})},
        {"role": "user", "content": "hi"},
    ]
    with (
        unittest.mock.patch("memories.services.chat_service.run_evaluator", _fake_evaluator),
        respx.mock,
    ):
        route = respx.post(_CHAT_URL).mock(
            side_effect=[_mock_ok("Response 1"), _mock_ok("Response 2")]
        )
        await run_contradiction_loop(
            db, session.id, 1, model, base_messages, character, {}, "hi", ollama
        )

    first_call = json.loads(route.calls[0].request.content)
    second_call = json.loads(route.calls[1].request.content)
    # First attempt renders the empty blob; the regenerated attempt renders the edit.
    assert "Zaphod" not in first_call["messages"][0]["content"]
    assert "Zaphod" in second_call["messages"][0]["content"]
