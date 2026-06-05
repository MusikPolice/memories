"""Integration tests for the chat SSE endpoint."""

from __future__ import annotations

import json

import aiosqlite
import httpx
import respx
from httpx import AsyncClient

from memories.database import create_fact, get_decisions, get_inferences, get_messages
from memories.models import Character, Session
from tests.unit.conftest import make_evaluator_ndjson, make_ollama_ndjson

_OLLAMA_CHAT_URL = "http://test-ollama-integration:11434/api/chat"


def _mock_ok(content: str = "I am fine.") -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(content))


def _mock_eval(
    verdict: str = "pass",
    new_inferences: list[dict] | None = None,
    violations: list[dict] | None = None,
) -> httpx.Response:
    return httpx.Response(200, content=make_evaluator_ndjson(verdict, new_inferences, violations))


def _mock_turn(
    character_content: str = "I am fine.",
    evaluator_verdict: str = "pass",
    new_inferences: list[dict] | None = None,
    violations: list[dict] | None = None,
) -> list[httpx.Response]:
    """Return a side_effect list for one complete turn (character + evaluator)."""
    return [_mock_ok(character_content), _mock_eval(evaluator_verdict, new_inferences, violations)]


def _parse_sse(text: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        event: dict[str, str] = {}
        for line in block.split("\n"):
            if line.startswith("event: "):
                event["event"] = line[7:]
            elif line.startswith("data: "):
                event["data"] = line[6:]
        if event:
            events.append(event)
    return events


# ---------------------------------------------------------------------------
# Phase 1 tests — updated to mock both character + evaluator calls
# ---------------------------------------------------------------------------


async def test_send_message_content_type_is_event_stream(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    assert "text/event-stream" in response.headers["content-type"]


async def test_send_message_emits_status_event_first(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    assert events[0]["event"] == "status"
    assert json.loads(events[0]["data"])["state"] == "generating"


async def test_send_message_emits_message_event(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("Hello from character."))
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    message_events = [e for e in events if e.get("event") == "message"]
    assert len(message_events) == 1
    data = json.loads(message_events[0]["data"])
    assert data["role"] == "assistant"
    assert data["content"] == "Hello from character."


async def test_send_message_emits_done_event_last(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    assert events[-1]["event"] == "done"


async def test_send_message_stores_user_message(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "User input"})
    messages = await get_messages(db, session.id)
    assert any(m.role == "user" and m.content == "User input" for m in messages)


async def test_send_message_stores_assistant_response(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("Character says hello."))
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    messages = await get_messages(db, session.id)
    assert any(m.role == "assistant" and m.content == "Character says hello." for m in messages)


async def test_ollama_receives_system_message_with_facts(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    await create_fact(db, character_id=character.id, key="birthplace", value="Reykjavik")
    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Where am I from?"}
        )
    body = json.loads(route.calls[0].request.content)
    system_content: str = body["messages"][0]["content"]
    assert "birthplace" in system_content
    assert "Reykjavik" in system_content


async def test_ollama_receives_prior_history(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("First reply."))
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "First message"})

    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("Second reply."))
        await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Second message"}
        )

    body = json.loads(route.calls[0].request.content)
    roles = [m["role"] for m in body["messages"]]
    assert "user" in roles
    assert "assistant" in roles


async def test_thinking_event_emitted_when_model_thinks(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    content=make_ollama_ndjson(
                        "My answer.", thinking="Let me consider this carefully."
                    ),
                ),
                _mock_eval(),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    thinking_events = [e for e in events if e.get("event") == "thinking"]
    assert len(thinking_events) == 1
    assert json.loads(thinking_events[0]["data"])["content"] == "Let me consider this carefully."


async def test_thinking_event_absent_when_no_thinking(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    assert not any(e.get("event") == "thinking" for e in events)


async def test_send_to_unknown_session_404(client: AsyncClient) -> None:
    response = await client.post("/api/sessions/9999/messages", json={"content": "Hello"})
    assert response.status_code == 404


async def test_send_to_ended_session_409(
    client: AsyncClient, character: Character, session: Session
) -> None:
    await client.post(f"/api/sessions/{session.id}/end")
    response = await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    assert response.status_code == 409


# ---------------------------------------------------------------------------
# Phase 2 tests — evaluator pipeline
# ---------------------------------------------------------------------------


async def test_send_message_emits_reviewing_event(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    status_events = [e for e in events if e.get("event") == "status"]
    states = [json.loads(e["data"])["state"] for e in status_events]
    assert "reviewing" in states


async def test_send_message_reviewing_event_before_message_event(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    reviewing_idx = next(
        i
        for i, e in enumerate(events)
        if e.get("event") == "status" and json.loads(e["data"])["state"] == "reviewing"
    )
    message_idx = next(i for i, e in enumerate(events) if e.get("event") == "message")
    assert reviewing_idx < message_idx


async def test_send_message_pass_verdict_no_ungrounded_field(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert not data.get("ungrounded")


async def test_send_message_pass_verdict_decision_stored(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 1
    assert decisions[0].verdict == "pass"


async def test_send_message_implication_verdict_emits_ungrounded_message(
    client: AsyncClient, character: Character, session: Session
) -> None:
    violations = [
        {
            "type": "implication",
            "description": "implied a sibling",
            "suggested_fact": {"key": "siblings", "value": "one"},
        }
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn("I have a sister.", "implication", violations=violations)
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Family?"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert data.get("ungrounded") is True


async def test_send_message_implication_verdict_emits_sidechannel(
    client: AsyncClient, character: Character, session: Session
) -> None:
    violations = [
        {
            "type": "implication",
            "description": "implied a sibling",
            "suggested_fact": {"key": "siblings", "value": "one"},
        }
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn("I have a sister.", "implication", violations=violations)
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Family?"}
        )
    events = _parse_sse(response.text)
    sidechannel_events = [e for e in events if e.get("event") == "sidechannel"]
    assert len(sidechannel_events) == 1
    data = json.loads(sidechannel_events[0]["data"])
    assert data["type"] == "implication"


async def test_send_message_implication_sidechannel_contains_violations(
    client: AsyncClient, character: Character, session: Session
) -> None:
    violations = [
        {
            "type": "implication",
            "description": "implied a sibling",
            "suggested_fact": {"key": "siblings", "value": "one"},
        }
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn("I have a sister.", "implication", violations=violations)
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Family?"}
        )
    events = _parse_sse(response.text)
    sc = next(e for e in events if e.get("event") == "sidechannel")
    data = json.loads(sc["data"])
    assert "violations" in data
    assert data["violations"][0]["suggested_fact"]["key"] == "siblings"


async def test_send_message_contradiction_emits_sidechannel_before_message(
    client: AsyncClient, character: Character, session: Session
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "wrong city", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                _mock_ok("I'm from London."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("I'm from Reykjavik."),
                _mock_eval("pass"),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Where from?"}
        )
    events = _parse_sse(response.text)
    sc_events = [e for e in events if e.get("event") == "sidechannel"]
    assert any(json.loads(e["data"])["type"] == "contradiction" for e in sc_events)
    sc_idx = next(i for i, e in enumerate(events) if e.get("event") == "sidechannel")
    msg_idx = next(i for i, e in enumerate(events) if e.get("event") == "message")
    assert sc_idx < msg_idx


async def test_send_message_contradiction_delivers_after_regeneration(
    client: AsyncClient, character: Character, session: Session
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "wrong city", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                _mock_ok("I'm from London."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("I'm from Reykjavik."),
                _mock_eval("pass"),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Where from?"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert data["content"] == "I'm from Reykjavik."


async def test_send_message_contradiction_response_not_premature(
    client: AsyncClient, character: Character, session: Session
) -> None:
    """The message event must not be emitted until contradictions are resolved."""
    contradiction_violation = [
        {"type": "contradiction", "description": "wrong city", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                _mock_ok("London."),
                _mock_eval("contradiction", violations=contradiction_violation),
                _mock_ok("Reykjavik."),
                _mock_eval("pass"),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Where from?"}
        )
    events = _parse_sse(response.text)
    message_events = [e for e in events if e.get("event") == "message"]
    assert len(message_events) == 1
    assert json.loads(message_events[0]["data"])["content"] == "Reykjavik."


async def test_send_message_max_retries_exceeded_flag_in_message(
    client: AsyncClient, character: Character, session: Session
) -> None:
    from memories.services import chat_service

    max_retries = chat_service.MAX_CONTRADICTION_RETRIES
    contradiction_violation = [
        {"type": "contradiction", "description": "always wrong", "suggested_fact": None}
    ]
    side_effects: list[httpx.Response] = []
    for _ in range(max_retries + 1):
        side_effects.append(_mock_ok("Still wrong."))
        side_effects.append(_mock_eval("contradiction", violations=contradiction_violation))

    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=side_effects)
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert data.get("contradiction_exhausted") is True


async def test_send_message_new_inference_logical_stored(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
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
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn(
                "Born in 1991.", "new_inference_logical", new_inferences=inferences
            )
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "When born?"})
    stored = await get_inferences(db, character.id)
    assert any(i.statement == "Born in 1991" for i in stored)


async def test_send_message_new_inference_probabilistic_emits_sidechannel(
    client: AsyncClient, character: Character, session: Session
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
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn(
                "I work very long hours.", "new_inference_probabilistic", new_inferences=inferences
            )
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Your schedule?"}
        )
    events = _parse_sse(response.text)
    sc_events = [e for e in events if e.get("event") == "sidechannel"]
    assert any(json.loads(e["data"])["type"] == "new_inference_probabilistic" for e in sc_events)


async def test_send_message_status_event_order_for_pass(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    event_names = [e.get("event") for e in events]
    # generating → reviewing → message → done
    assert event_names[0] == "status"
    assert json.loads(events[0]["data"])["state"] == "generating"
    reviewing_idx = next(
        i
        for i, e in enumerate(events)
        if e.get("event") == "status" and json.loads(e["data"])["state"] == "reviewing"
    )
    message_idx = next(i for i, e in enumerate(events) if e.get("event") == "message")
    done_idx = next(i for i, e in enumerate(events) if e.get("event") == "done")
    assert reviewing_idx < message_idx < done_idx
