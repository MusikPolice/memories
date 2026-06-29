"""Integration tests for the chat SSE endpoint."""

from __future__ import annotations

import json

import aiosqlite
import httpx
import respx
from httpx import AsyncClient

from memories.database import (
    _embedding_to_blob,
    create_experience,
    create_fact,
    create_inference,
    get_decisions,
    get_fact_rows,
    get_facts,
    get_inferences,
    get_messages,
    set_facts,
)
from memories.models import Character, Session
from tests.unit.conftest import (
    make_evaluator_ndjson,
    make_ollama_ndjson,
    make_plain_tool_response,
    make_tool_call_response,
)

_OLLAMA_CHAT_URL = "http://test-ollama-integration:11434/api/chat"
_EMBED_URL = "http://test-ollama-integration:11434/api/embed"
_EMBED_VEC = [1.0, 0.0, 0.0, 0.0]


def _mock_ok(content: str = "I am fine.") -> httpx.Response:
    return httpx.Response(200, content=make_ollama_ndjson(content))


def _mock_world_builder() -> httpx.Response:
    return httpx.Response(200, content=make_plain_tool_response("Nothing to extract."))


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
    """Return a side_effect list for one complete turn (World Builder + character + evaluator)."""
    return [
        _mock_world_builder(),
        _mock_ok(character_content),
        _mock_eval(evaluator_verdict, new_inferences, violations),
    ]


def _mock_world_builder_writes(path: str, value: str) -> list[httpx.Response]:
    """side_effect entries for a World Builder pass that writes one fact via author_set_facts."""
    return [
        httpx.Response(
            200,
            content=make_tool_call_response(
                "author_set_facts", {"facts": [{"path": path, "value": value}]}
            ),
        ),
        httpx.Response(200, content=make_plain_tool_response("Done.")),
    ]


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
    assert json.loads(events[0]["data"])["state"] == "extracting"


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

    body = json.loads(route.calls[1].request.content)
    roles = [m["role"] for m in body["messages"]]
    assert "user" in roles
    assert "assistant" in roles


async def test_thinking_event_emitted_when_model_thinks(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
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


async def test_send_message_emits_generating_status(
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
    assert "generating" in states


async def test_send_message_generating_event_before_message_event(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    generating_idx = next(
        i
        for i, e in enumerate(events)
        if e.get("event") == "status" and json.loads(e["data"])["state"] == "generating"
    )
    message_idx = next(i for i, e in enumerate(events) if e.get("event") == "message")
    assert generating_idx < message_idx


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
    assert decisions[0].tool_args["verdict"] == "pass"


async def test_send_message_contradiction_emits_sidechannel_before_message(
    client: AsyncClient, character: Character, session: Session
) -> None:
    contradiction_violation = [
        {"type": "contradiction", "description": "wrong city", "suggested_fact": None}
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
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
                _mock_world_builder(),
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
                _mock_world_builder(),
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
    side_effects: list[httpx.Response] = [_mock_world_builder()]
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
    # extracting → generating → reviewing → message → done
    status_events = [e for e in events if e.get("event") == "status"]
    states = [json.loads(e["data"])["state"] for e in status_events]
    assert states[0] == "extracting"
    assert states[1] == "generating"
    assert states[2] == "reviewing"
    generating_idx = next(
        i
        for i, e in enumerate(events)
        if e.get("event") == "status" and json.loads(e["data"])["state"] == "generating"
    )
    reviewing_idx = next(
        i
        for i, e in enumerate(events)
        if e.get("event") == "status" and json.loads(e["data"])["state"] == "reviewing"
    )
    message_idx = next(i for i, e in enumerate(events) if e.get("event") == "message")
    done_idx = next(i for i, e in enumerate(events) if e.get("event") == "done")
    assert generating_idx < reviewing_idx < message_idx < done_idx


async def test_send_message_reviewing_event_before_message_event(
    client: AsyncClient, character: Character, session: Session
) -> None:
    """status(reviewing) must appear before event: message for every verdict."""
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


# ---------------------------------------------------------------------------
# Phase 3 additions — inferences in chat flow
# ---------------------------------------------------------------------------


async def test_send_message_system_message_includes_inferences(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """When character has active inferences, Ollama request system message contains them."""
    await create_fact(db, character_id=character.id, key="age", value="33")
    await create_inference(
        db,
        character_id=character.id,
        statement="Alice was born in 1993",
        derivation="age=33",
    )

    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})

    body = json.loads(route.calls[1].request.content)
    system_content = body["messages"][0]["content"]
    assert "Alice was born in 1993" in system_content


async def test_send_message_no_inferences_section_when_none_exist(
    client: AsyncClient, character: Character, session: Session
) -> None:
    """When no inferences exist, the system message does not contain the inferences header."""
    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})

    body = json.loads(route.calls[1].request.content)
    system_content = body["messages"][0]["content"]
    assert "## Your Inferences" not in system_content


async def test_send_message_lazy_logical_inference_stored_with_depth(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """new_inference_logical with source_inference_ids=[existing_id] → stored with depth=2."""
    existing = await create_inference(
        db,
        character_id=character.id,
        statement="Existing at depth 1",
        derivation="d",
        depth=1,
    )
    new_inf_payload = [
        {
            "inference_type": "logical",
            "statement": "Derived from existing",
            "derivation": "from existing",
            "source_fact_ids": [],
            "source_inference_ids": [existing.id],
        }
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn(
                "Statement.", "new_inference_logical", new_inferences=new_inf_payload
            )
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Tell me"})

    stored = await get_inferences(db, character.id)
    derived = next((i for i in stored if "Derived from existing" in i.statement), None)
    assert derived is not None
    assert derived.depth == 2


async def test_send_message_lazy_inference_at_max_depth_is_stored(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Lazy inference at exactly MAX_INFERENCE_DEPTH is stored."""
    from memories.services.inference_service import MAX_INFERENCE_DEPTH

    existing = await create_inference(
        db,
        character_id=character.id,
        statement="At max-1 depth",
        derivation="d",
        depth=MAX_INFERENCE_DEPTH - 1,
    )
    new_inf_payload = [
        {
            "inference_type": "logical",
            "statement": "At exactly max depth",
            "derivation": "from existing",
            "source_fact_ids": [],
            "source_inference_ids": [existing.id],
        }
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn(
                "Statement.", "new_inference_logical", new_inferences=new_inf_payload
            )
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Tell me"})

    stored = await get_inferences(db, character.id)
    assert any("At exactly max depth" in i.statement for i in stored)


async def test_send_message_lazy_inference_exceeding_depth_not_stored(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Lazy inference at depth > MAX_INFERENCE_DEPTH is not stored."""
    from memories.services.inference_service import MAX_INFERENCE_DEPTH

    existing = await create_inference(
        db,
        character_id=character.id,
        statement="At max depth",
        derivation="d",
        depth=MAX_INFERENCE_DEPTH,
    )
    new_inf_payload = [
        {
            "inference_type": "logical",
            "statement": "Exceeds depth cap",
            "derivation": "from existing",
            "source_fact_ids": [],
            "source_inference_ids": [existing.id],
        }
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn(
                "Statement.", "new_inference_logical", new_inferences=new_inf_payload
            )
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Tell me"})

    stored = await get_inferences(db, character.id)
    assert not any("Exceeds depth cap" in i.statement for i in stored)


async def test_send_message_lazy_batch_second_inference_sees_first_in_depth_snapshot(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Depth snapshot is refreshed per-inference in a batch so B cites A correctly.

    Scenario: E exists at depth 1.  Evaluator returns [A (cites E, depth=2),
    B (cites A, should be depth=3)].  Without the snapshot-append fix, B would
    not see A in the snapshot and fall back to depth=1.
    """
    existing = await create_inference(
        db, character_id=character.id, statement="E at depth 1", derivation="e", depth=1
    )
    # Determine A's expected database id so we can reference it in B's payload
    row = await (await db.execute("SELECT COALESCE(MAX(id), 0) FROM inferences")).fetchone()
    assert row is not None
    a_expected_id = row[0] + 1

    new_inf_payload = [
        {
            "inference_type": "logical",
            "statement": "Inference A (cites E)",
            "derivation": "from E",
            "source_fact_ids": [],
            "source_inference_ids": [existing.id],
        },
        {
            "inference_type": "logical",
            "statement": "Inference B (cites A)",
            "derivation": "from A",
            "source_fact_ids": [],
            "source_inference_ids": [a_expected_id],
        },
    ]
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=_mock_turn(
                "Statement.", "new_inference_logical", new_inferences=new_inf_payload
            )
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Tell me"})

    stored = await get_inferences(db, character.id)
    inf_a = next(i for i in stored if "Inference A" in i.statement)
    inf_b = next(i for i in stored if "Inference B" in i.statement)
    assert inf_a.depth == 2
    # Without the snapshot refresh, B's compute_depth would not find A and return 1
    assert inf_b.depth == 3


# ---------------------------------------------------------------------------
# Phase 4 additions — category sections and mutability annotations in prompts
# ---------------------------------------------------------------------------


async def test_chat_system_prompt_groups_user_and_character_facts(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    await create_fact(db, character_id=character.id, key="user_name", value="Jon", category="user")
    await create_fact(
        db, character_id=character.id, key="occupation", value="surgeon", category="character"
    )

    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})

    char_call_body = json.loads(route.calls[1].request.content)
    system_prompt = char_call_body["messages"][0]["content"]
    assert "User" in system_prompt
    assert "Character" in system_prompt


async def test_chat_system_prompt_renders_all_schema_sections(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """Schema-driven prompt always renders all top-level sections, even with no facts set."""
    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})

    char_call_body = json.loads(route.calls[1].request.content)
    system_prompt = char_call_body["messages"][0]["content"]
    assert "### Character" in system_prompt
    assert "### User" in system_prompt
    assert "### Setting" in system_prompt


async def test_chat_system_prompt_shows_blob_fact_value(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """A value written to the character_facts blob appears in the schema-rendered prompt."""
    await set_facts(
        db,
        character_id=character.id,
        blob={"Character": {"Identity": {"Name": {"Value": "Alice Thornton"}}}},
    )

    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Who are you?"})

    char_call_body = json.loads(route.calls[1].request.content)
    system_prompt = char_call_body["messages"][0]["content"]
    assert "Character.Identity.Name" in system_prompt
    assert "Alice Thornton" in system_prompt


async def test_chat_system_prompt_annotates_immutable_fact(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """An IMMUTABLE schema leaf renders with the [IMMUTABLE] tag."""
    await set_facts(
        db,
        character_id=character.id,
        blob={"Character": {"Identity": {"Name": {"Value": "Alice"}}}},
    )

    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})

    char_call_body = json.loads(route.calls[1].request.content)
    system_prompt = char_call_body["messages"][0]["content"]
    assert "[IMMUTABLE]" in system_prompt


async def test_chat_system_prompt_annotates_fluid_fact(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """A FLUID schema leaf renders with the [FLUID] tag."""
    await set_facts(
        db,
        character_id=character.id,
        blob={"Setting": {"Location": {"Weather": {"Value": "Clear"}}}},
    )

    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})

    char_call_body = json.loads(route.calls[1].request.content)
    system_prompt = char_call_body["messages"][0]["content"]
    assert "[FLUID]" in system_prompt


# ---------------------------------------------------------------------------
# Phase 5 additions — experience retrieval and active set
# ---------------------------------------------------------------------------


def _mock_embed_ok() -> httpx.Response:
    return httpx.Response(200, json={"embeddings": [_EMBED_VEC]})


async def test_send_message_no_embed_call_when_no_experiences(
    client: AsyncClient, character: Character, session: Session
) -> None:
    """No experiences in DB → embed endpoint NOT called."""
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        embed_route = respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    assert not embed_route.called


async def test_send_message_embed_call_when_experiences_exist(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """One experience in DB → embed endpoint IS called with user message text."""
    blob = _embedding_to_blob(_EMBED_VEC)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=blob,
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        embed_route = respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    assert embed_route.called
    body = json.loads(embed_route.calls[0].request.content)
    assert body["input"] == "Hello"


async def test_send_message_experience_appears_in_system_prompt(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """Retrieved active experience statement appears in the character LLM system message."""
    blob = _embedding_to_blob(_EMBED_VEC)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=blob,
    )
    with respx.mock:
        chat_route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "Hello"})
    body = json.loads(chat_route.calls[1].request.content)
    system_content = body["messages"][0]["content"]
    assert "User lives in Chicago" in system_content


async def test_send_message_message_event_includes_active_experience_ids(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """SSE message event payload contains active_experience_ids key with a list."""
    blob = _embedding_to_blob(_EMBED_VEC)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=blob,
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert "active_experience_ids" in data
    assert isinstance(data["active_experience_ids"], list)


async def test_send_message_active_experience_ids_matches_retrieved(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """IDs in active_experience_ids correspond to experiences in the system prompt."""
    blob = _embedding_to_blob(_EMBED_VEC)
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=blob,
    )
    with respx.mock:
        chat_route = respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    active_ids = json.loads(msg_event["data"]).get("active_experience_ids", [])
    body = json.loads(chat_route.calls[1].request.content)
    system_content = body["messages"][0]["content"]
    # Experience was retrieved and injected into system prompt
    assert exp.id in active_ids
    assert "User lives in Chicago" in system_content


async def test_send_message_active_experience_ids_empty_when_no_experiences(
    client: AsyncClient, character: Character, session: Session
) -> None:
    """When no experiences exist in DB, active_experience_ids is []."""
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert data.get("active_experience_ids", []) == []


async def test_send_message_active_experience_ids_present_across_turns(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """Experiences that score above threshold appear in active_experience_ids on every turn."""
    blob = _embedding_to_blob(_EMBED_VEC)
    exp1 = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=blob,
    )
    exp2 = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User has a dog named Rex",
        source="told_by_user",
        embedding=blob,
    )
    # Both experiences use _EMBED_VEC so both score 1.0 and are retrieved every turn.
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("First reply."))
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "First message"})
    # Second turn
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("Second reply."))
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Second message"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    active_ids = json.loads(msg_event["data"]).get("active_experience_ids", [])
    assert exp1.id in active_ids or exp2.id in active_ids


async def test_send_message_message_event_includes_experience_scores(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """SSE message event payload contains experience_scores key with a list."""
    blob = _embedding_to_blob(_EMBED_VEC)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=blob,
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert "experience_scores" in data
    assert isinstance(data["experience_scores"], list)


async def test_send_message_experience_scores_covers_all_stored_experiences(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """experience_scores contains one entry per stored experience."""
    blob = _embedding_to_blob(_EMBED_VEC)
    exps = []
    for i in range(3):
        exp = await create_experience(
            db,
            character_id=character.id,
            session_id=session.id,
            statement=f"Experience {i}",
            source="told_by_user",
            embedding=blob,
        )
        exps.append(exp)
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    scores = json.loads(msg_event["data"]).get("experience_scores", [])
    score_ids = {s["id"] for s in scores}
    for exp in exps:
        assert exp.id in score_ids


async def test_send_message_experience_scores_empty_when_no_experiences(
    client: AsyncClient, character: Character, session: Session
) -> None:
    """When no experiences exist in DB, experience_scores is []."""
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    data = json.loads(msg_event["data"])
    assert data.get("experience_scores", []) == []


async def test_send_message_experience_scores_scores_are_floats(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    """Each entry in experience_scores has a numeric score value."""
    blob = _embedding_to_blob(_EMBED_VEC)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=blob,
    )
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        respx.post(_EMBED_URL).mock(return_value=_mock_embed_ok())
        response = await client.post(
            f"/api/sessions/{session.id}/messages", json={"content": "Hello"}
        )
    events = _parse_sse(response.text)
    msg_event = next(e for e in events if e.get("event") == "message")
    scores = json.loads(msg_event["data"]).get("experience_scores", [])
    for entry in scores:
        assert isinstance(entry["score"], int | float)


# ---------------------------------------------------------------------------


async def test_accept_implication_on_high_mutability_fact_preserves_mutability(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    # Create a high-mutability fact
    await create_fact(
        db, character_id=character.id, key="mood", value="cheerful", mutability="high"
    )

    _VIOLATION = {
        "type": "implication",
        "description": (
            "Mood appears to have shifted from 'cheerful' to 'anxious' (high-mutability fact)"
        ),
        "suggested_fact": {"key": "mood", "value": "anxious"},
    }

    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                _mock_world_builder(),
                httpx.Response(200, content=make_ollama_ndjson("I feel anxious today.")),
                httpx.Response(
                    200, content=make_evaluator_ndjson("implication", violations=[_VIOLATION])
                ),
            ]
        )
        await client.post(f"/api/sessions/{session.id}/messages", json={"content": "How are you?"})

    # Accept the implication (value changes, but we expect mutability to be preserved)
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn("I feel anxious today."))
        await client.post(
            f"/api/sessions/{session.id}/turns/1/accept-implication",
            json={"key": "mood", "value": "anxious", "regenerate": False},
        )

    facts = await get_fact_rows(db, character.id)
    mood_fact = next((f for f in facts if f.key == "mood"), None)
    assert mood_fact is not None
    assert mood_fact.mutability == "high"


# ---------------------------------------------------------------------------
# Step 4 additions — World Builder SSE events and DB writes
# ---------------------------------------------------------------------------


async def test_turn_emits_status_extracting_before_status_generating(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """SSE stream contains status(extracting) and it precedes status(generating)."""
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "Hello."},
        )

    events = _parse_sse(response.text)
    status_events = [e for e in events if e.get("event") == "status"]
    states = [json.loads(e["data"])["state"] for e in status_events]

    assert "extracting" in states
    extracting_idx = states.index("extracting")
    generating_idx = states.index("generating")
    assert extracting_idx < generating_idx


async def test_turn_on_world_builder_failure_still_delivers_response(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    """World Builder call fails to connect; SSE stream still delivers character message event."""
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                httpx.ConnectError("refused"),
                _mock_ok("I am fine, despite the world builder failure."),
                _mock_eval("pass"),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "Hello."},
        )

    events = _parse_sse(response.text)
    message_events = [e for e in events if e.get("event") == "message"]
    assert len(message_events) == 1
    assert "I am fine" in json.loads(message_events[0]["data"])["content"]


async def test_turn_world_builder_fact_written_to_character_facts_db(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.State-Of-Mind.Mood", "Anxious"),
                _mock_ok("I hear you."),
                _mock_eval("pass"),
            ]
        )
        await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "I'm feeling really anxious"},
        )

    facts = await get_facts(db, character.id)
    assert facts["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Anxious"


async def test_turn_world_builder_fact_appears_in_character_prompt(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        route = respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.State-Of-Mind.Mood", "Anxious"),
                _mock_ok("I hear you."),
                _mock_eval("pass"),
            ]
        )
        await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "I'm feeling really anxious"},
        )

    # calls[2] is the character LLM call (calls[0]/[1] are the World Builder's two rounds)
    char_body = json.loads(route.calls[2].request.content)
    system_content = char_body["messages"][0]["content"]
    assert "Anxious" in system_content


async def test_turn_emits_world_builder_applied_sidechannel(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.Identity.Name", "Beatrice"),
                _mock_ok("Hi Beatrice."),
                _mock_eval("pass"),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "My name is Beatrice"},
        )

    events = _parse_sse(response.text)
    sc_events = [
        e
        for e in events
        if e.get("event") == "sidechannel"
        and json.loads(e.get("data", "{}")).get("type") == "world_builder_applied"
    ]
    assert len(sc_events) == 1


async def test_turn_world_builder_applied_payload_has_path_and_value(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.Identity.Name", "Beatrice"),
                _mock_ok("Hi Beatrice."),
                _mock_eval("pass"),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "My name is Beatrice"},
        )

    events = _parse_sse(response.text)
    sc = next(
        e
        for e in events
        if e.get("event") == "sidechannel"
        and json.loads(e.get("data", "{}")).get("type") == "world_builder_applied"
    )
    payload = json.loads(sc["data"])
    assert payload["facts"] == [{"path": "Character.Identity.Name", "value": "Beatrice"}]


async def test_turn_no_world_builder_applied_sidechannel_when_no_facts_written(
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(side_effect=_mock_turn())
        response = await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "Hello."},
        )

    events = _parse_sse(response.text)
    sc_events = [
        e
        for e in events
        if e.get("event") == "sidechannel"
        and json.loads(e.get("data", "{}")).get("type") == "world_builder_applied"
    ]
    assert sc_events == []


async def test_turn_world_builder_unknown_path_does_not_write_but_turn_completes(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.Identity.NotARealPath", "x"),
                _mock_ok("Hi there."),
                _mock_eval("pass"),
            ]
        )
        response = await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "Hello."},
        )

    facts = await get_facts(db, character.id)
    assert "NotARealPath" not in facts.get("Character", {}).get("Identity", {})
    events = _parse_sse(response.text)
    message_events = [e for e in events if e.get("event") == "message"]
    assert len(message_events) == 1


async def test_turn_decisions_includes_world_builder_row_when_facts_written(
    db: aiosqlite.Connection,
    client: AsyncClient,
    character: Character,
    session: Session,
) -> None:
    with respx.mock:
        respx.post(_OLLAMA_CHAT_URL).mock(
            side_effect=[
                *_mock_world_builder_writes("Character.Identity.Name", "Beatrice"),
                _mock_ok("Hi Beatrice."),
                _mock_eval("pass"),
            ]
        )
        await client.post(
            f"/api/sessions/{session.id}/messages",
            json={"content": "My name is Beatrice"},
        )

    response = await client.get(f"/api/sessions/{session.id}/decisions")
    decisions = response.json()
    assert any(d["pass_name"] == "world_builder" for d in decisions)
