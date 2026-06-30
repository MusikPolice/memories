"""Unit tests for memories.services.evaluator."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import aiosqlite
import httpx
import pytest
import respx

from memories.database import create_character, create_session, get_facts, get_inferences, set_facts
from memories.models import Character, Experience, Inference
from memories.services.evaluator import (
    EvaluatorResult,
    Violation,
    build_evaluator_prompt,
    run_evaluator,
)
from memories.services.ollama_client import OllamaClient
from memories.services.sse_events import SSEEvent
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_multi_tool_call_response,
    make_tool_call_response,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_CHARACTER = Character(
    id=1,
    name="Alice",
    modelfile_base="qwen3:7b",
    current_model_name=None,
    created_at=__import__("datetime").datetime(2024, 1, 1),
)


@pytest.fixture(autouse=True)
async def _db_character(db: aiosqlite.Connection) -> None:
    """Ensure the test character and session exist so FK constraints pass."""
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    await create_session(db, character_id=char.id)


# Facts blob with some populated paths
_FACTS_BLOB: dict[str, Any] = {
    "Character": {
        "Identity": {"Occupation": {"Value": "surgeon"}},
        "Background": {"Hometown": {"Value": "Reykjavik"}},
    }
}

_USER_MSG = "Where are you from?"
_CHAR_RESPONSE = "I grew up in Reykjavik, actually."

# ---------------------------------------------------------------------------
# build_evaluator_prompt — basic content
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
    assert "surgeon" not in prompt
    assert "unset" in prompt.lower()


def test_evaluator_prompt_with_contradiction_hints_lists_them() -> None:
    hints = ["character said 'London' but hometown is Reykjavik"]
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, hints)
    assert hints[0] in prompt


# ---------------------------------------------------------------------------
# build_evaluator_prompt — inferences
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


# ---------------------------------------------------------------------------
# build_evaluator_prompt — experiences
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


# ---------------------------------------------------------------------------
# build_evaluator_prompt — schema and fact values sections
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


# ---------------------------------------------------------------------------
# build_evaluator_prompt — tool-calling task instructions (Step 5)
# ---------------------------------------------------------------------------


def test_evaluator_prompt_mentions_set_fact_tool() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "set_fact" in prompt


def test_evaluator_prompt_mentions_propose_inference_tool() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "propose_inference" in prompt


def test_evaluator_prompt_mentions_report_contradiction_tool() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "report_contradiction" in prompt


def test_evaluator_prompt_mentions_report_pass_tool() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "report_pass" in prompt


def test_evaluator_prompt_instructs_terminal_call() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "ALWAYS finish by calling exactly one" in prompt


def test_evaluator_prompt_no_json_output_instructions() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    assert "Return a JSON object" not in prompt
    assert '"verdict":' not in prompt


def test_evaluator_prompt_contains_immutable_contradiction_instruction() -> None:
    prompt = build_evaluator_prompt(_CHARACTER, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE)
    prompt_lower = prompt.lower()
    assert "immutable" in prompt_lower
    assert "report_contradiction" in prompt


# ---------------------------------------------------------------------------
# run_evaluator — request shape
# ---------------------------------------------------------------------------


@respx.mock
async def test_evaluator_request_sends_tools_list(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    body = json.loads(route.calls[0].request.content)
    tool_names = [t["function"]["name"] for t in body["tools"]]
    assert "set_fact" in tool_names
    assert "propose_inference" in tool_names
    assert "report_contradiction" in tool_names
    assert "report_pass" in tool_names


@respx.mock
async def test_evaluator_request_sends_stream_false(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("stream") is False


@respx.mock
async def test_evaluator_request_has_no_format_key(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    body = json.loads(route.calls[0].request.content)
    assert "format" not in body


# ---------------------------------------------------------------------------
# run_evaluator — terminal tool mapping
# ---------------------------------------------------------------------------


@respx.mock
async def test_run_evaluator_report_pass_returns_pass_verdict(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    result, _ = await run_evaluator(
        db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama
    )
    assert result.verdict == "pass"


@respx.mock
async def test_run_evaluator_report_contradiction_returns_contradiction_verdict(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_tool_call_response("report_contradiction", {"description": "wrong city"}),
        )
    )
    result, _ = await run_evaluator(
        db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama
    )
    assert result.verdict == "contradiction"
    assert result.violations[0].description == "wrong city"


@respx.mock
async def test_run_evaluator_report_contradiction_violation_type_is_contradiction(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_tool_call_response(
                "report_contradiction", {"description": "said London not Reykjavik"}
            ),
        )
    )
    result, _ = await run_evaluator(
        db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama
    )
    assert result.violations[0].type == "contradiction"


@respx.mock
async def test_run_evaluator_batched_set_fact_then_report_pass_in_one_round(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "set_fact",
                        {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    result, _ = await run_evaluator(
        db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama
    )
    assert result.verdict == "pass"
    assert route.call_count == 1


@respx.mock
async def test_run_evaluator_returns_evaluator_result_type(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    result, _ = await run_evaluator(
        db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama
    )
    assert isinstance(result, EvaluatorResult)


@respx.mock
async def test_run_evaluator_returns_updated_facts_blob(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    _, blob = await run_evaluator(
        db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama
    )
    assert isinstance(blob, dict)


# ---------------------------------------------------------------------------
# _handle_set_fact — fluid path
# ---------------------------------------------------------------------------


@respx.mock
async def test_set_fact_fluid_path_writes_to_db(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "set_fact",
                        {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)
    facts = await get_facts(db, _CHARACTER.id)
    assert facts["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Anxious"


@respx.mock
async def test_set_fact_fluid_path_enum_coerced_case_insensitively(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    ("set_fact", {"path": "Character.State-Of-Mind.Mood", "value": "calm"}),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)
    facts = await get_facts(db, _CHARACTER.id)
    assert facts["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Calm"


@respx.mock
async def test_set_fact_fluid_path_enum_invalid_value_returns_error_and_path_unwritten(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_multi_tool_call_response(
                    [
                        (
                            "set_fact",
                            {
                                "path": "Character.State-Of-Mind.Mood",
                                "value": "Apprehensive",
                            },
                        ),
                        ("report_pass", {}),
                    ]
                ),
            ),
        ]
    )
    _, blob = await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert route.call_count == 1
    assert "Character" not in blob or "State-Of-Mind" not in blob.get("Character", {})


@respx.mock
async def test_set_fact_fluid_path_logs_decision(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "set_fact",
                        {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    session_id = 1
    turn_id = 1
    await run_evaluator(db, _CHARACTER, session_id, turn_id, {}, _USER_MSG, _CHAR_RESPONSE, ollama)
    # get_decisions scoped to session

    # Verify a decision was logged with tool_name="set_fact"
    cursor = await db.execute(
        "SELECT tool_name, tool_args FROM decisions WHERE character_id = ?",
        (_CHARACTER.id,),
    )
    rows = await cursor.fetchall()
    tool_names = [r[0] for r in rows]
    assert "set_fact" in tool_names
    set_fact_row = next(r for r in rows if r[0] == "set_fact")
    args = json.loads(set_fact_row[1])
    assert args["path"] == "Character.State-Of-Mind.Mood"


@respx.mock
async def test_set_fact_fluid_path_emits_sidechannel_event(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    events: list[SSEEvent] = []

    async def on_event(ev: SSEEvent) -> None:
        events.append(ev)

    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "set_fact",
                        {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(
        db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama, on_event=on_event
    )
    sc_events = [e for e in events if e.event == "sidechannel"]
    assert any(e.data.get("type") == "fact_update_fluid" for e in sc_events)


@respx.mock
async def test_set_fact_unknown_path_returns_error(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_multi_tool_call_response(
                    [
                        ("set_fact", {"path": "Unknown.Path.Field", "value": "x"}),
                        ("report_pass", {}),
                    ]
                ),
            ),
        ]
    )
    _, blob = await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)
    # Tool result should be an error string — verify no blob write happened
    assert route.call_count == 1
    assert blob == {}


@respx.mock
async def test_set_fact_fluid_path_integer_coerced_from_string(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    """Integer value supplied as a string is coerced to int before being stored.

    The schema has no Fluid Integer leaf, so check_write_permitted is monkeypatched
    to return "Fluid" for Setting.Temporal.Current-Year (a real Integer leaf).
    """
    import unittest.mock

    _path = "Setting.Temporal.Current-Year"
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    with unittest.mock.patch(
        "memories.services.evaluator.check_write_permitted", return_value="Fluid"
    ):
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(
                200,
                content=make_multi_tool_call_response(
                    [
                        ("set_fact", {"path": _path, "value": "2024"}),
                        ("report_pass", {}),
                    ]
                ),
            )
        )
        _, blob = await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)

    node: Any = blob
    for part in _path.split("."):
        node = node[part]
    assert node["Value"] == 2024
    assert isinstance(node["Value"], int)


@respx.mock
async def test_set_fact_fluid_path_integer_invalid_string_returns_error_and_path_unwritten(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    """A non-integer string for an Integer leaf returns an error; blob is unchanged."""
    import unittest.mock

    _path = "Setting.Temporal.Current-Year"
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    with unittest.mock.patch(
        "memories.services.evaluator.check_write_permitted", return_value="Fluid"
    ):
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(
                200,
                content=make_multi_tool_call_response(
                    [
                        ("set_fact", {"path": _path, "value": "not-a-number"}),
                        ("report_pass", {}),
                    ]
                ),
            )
        )
        _, blob = await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)

    assert "Setting" not in blob


# ---------------------------------------------------------------------------
# _handle_set_fact — immutable path
# ---------------------------------------------------------------------------


@respx.mock
async def test_set_fact_immutable_unset_returns_stub_error(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    # Immutable path with no pre-seeded value
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_multi_tool_call_response(
                    [
                        ("set_fact", {"path": "Character.Identity.Name", "value": "Bob"}),
                        ("report_pass", {}),
                    ]
                ),
            ),
        ]
    )
    _, blob = await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)
    # Stub error returned; no value written
    assert "Character" not in blob or "Identity" not in blob.get("Character", {})


@respx.mock
async def test_set_fact_immutable_already_set_returns_contradiction_hint(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    """A set_fact call against an already-set Immutable path returns an error mentioning
    report_contradiction."""
    # Pre-seed with an Immutable value
    pre_blob = {"Character": {"Identity": {"Name": {"Value": "Alice"}}}}
    await set_facts(db, character_id=_CHARACTER.id, blob=pre_blob)

    tool_results: list[str] = []

    async def _capturing_chat_with_tools(self: Any, *args: Any, **kwargs: Any) -> Any:
        from memories.services.ollama_client import ToolCallResult

        # Intercept: manually invoke the set_fact handler with a conflicting value
        handlers = args[3] if len(args) > 3 else kwargs.get("tool_handlers", {})
        handler = handlers.get("set_fact")
        if handler:
            result = await handler({"path": "Character.Identity.Name", "value": "Bob"})
            tool_results.append(result)
        return ToolCallResult(
            content="",
            history=[],
            rounds=1,
            cap_reached=False,
            terminal_call={"name": "report_pass", "arguments": {}, "result": "Pass recorded."},
        )

    import unittest.mock

    with unittest.mock.patch.object(OllamaClient, "chat_with_tools", _capturing_chat_with_tools):
        await run_evaluator(db, _CHARACTER, 1, 1, pre_blob, _USER_MSG, _CHAR_RESPONSE, ollama)

    assert len(tool_results) == 1
    assert tool_results[0].startswith("Error:")
    assert "report_contradiction" in tool_results[0]


@respx.mock
async def test_set_fact_immutable_already_set_value_unchanged_in_returned_blob(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    pre_blob = {"Character": {"Identity": {"Name": {"Value": "Alice"}}}}
    await set_facts(db, character_id=_CHARACTER.id, blob=pre_blob)

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_multi_tool_call_response(
                    [
                        # Conflict: Immutable already set to "Alice", model tries "Bob"
                        ("set_fact", {"path": "Character.Identity.Name", "value": "Bob"}),
                        ("report_pass", {}),
                    ]
                ),
            ),
        ]
    )
    _, blob = await run_evaluator(db, _CHARACTER, 1, 1, pre_blob, _USER_MSG, _CHAR_RESPONSE, ollama)
    # Original value must be preserved
    assert blob["Character"]["Identity"]["Name"]["Value"] == "Alice"


# ---------------------------------------------------------------------------
# _handle_set_fact — mutable path
# ---------------------------------------------------------------------------


@respx.mock
async def test_set_fact_mutable_path_returns_stub_error_regardless_of_current_value(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    tool_results: list[str] = []

    import unittest.mock

    from memories.services.ollama_client import ToolCallResult

    async def _capturing(self: Any, *args: Any, **kwargs: Any) -> Any:
        handlers = args[3] if len(args) > 3 else kwargs.get("tool_handlers", {})
        handler = handlers.get("set_fact")
        if handler:
            result = await handler({"path": "Character.Identity.Occupation", "value": "engineer"})
            tool_results.append(result)
        return ToolCallResult(
            content="",
            history=[],
            rounds=1,
            cap_reached=False,
            terminal_call={"name": "report_pass", "arguments": {}, "result": "Pass recorded."},
        )

    with unittest.mock.patch.object(OllamaClient, "chat_with_tools", _capturing):
        await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama)

    assert len(tool_results) == 1
    assert "not implemented" in tool_results[0].lower() or tool_results[0].startswith("Error:")


# ---------------------------------------------------------------------------
# _handle_propose_inference
# ---------------------------------------------------------------------------


@respx.mock
async def test_propose_inference_writes_inference_row(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "propose_inference",
                        {
                            "statement": "Alice likely works nights",
                            "derivation": "occupation=surgeon",
                        },
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    inferences = await get_inferences(db, _CHARACTER.id)
    assert any(i.statement == "Alice likely works nights" for i in inferences)


@respx.mock
async def test_propose_inference_stored_with_depth_one(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "propose_inference",
                        {"statement": "Alice is skilled", "derivation": "years of training"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    inferences = await get_inferences(db, _CHARACTER.id)
    inf = next(i for i in inferences if "Alice is skilled" in i.statement)
    assert inf.depth == 1


@respx.mock
async def test_propose_inference_stores_source_paths(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "propose_inference",
                        {
                            "statement": "Alice has expertise",
                            "derivation": "from occupation",
                            "source_paths": ["Character.Identity.Occupation"],
                        },
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    inferences = await get_inferences(db, _CHARACTER.id)
    inf = next(i for i in inferences if "Alice has expertise" in i.statement)
    assert "Character.Identity.Occupation" in inf.source_fact_paths


@respx.mock
async def test_propose_inference_missing_source_paths_defaults_to_empty_list(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "propose_inference",
                        {"statement": "Alice is dedicated", "derivation": "general pattern"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    inferences = await get_inferences(db, _CHARACTER.id)
    inf = next(i for i in inferences if "Alice is dedicated" in i.statement)
    assert inf.source_fact_paths == []


@respx.mock
async def test_propose_inference_logs_decision(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "propose_inference",
                        {"statement": "Alice is resilient", "derivation": "from background"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    cursor = await db.execute(
        "SELECT tool_name FROM decisions WHERE character_id = ?", (_CHARACTER.id,)
    )
    rows = await cursor.fetchall()
    tool_names = [r[0] for r in rows]
    assert "propose_inference" in tool_names


@respx.mock
async def test_propose_inference_emits_sidechannel_event(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def on_event(ev: SSEEvent) -> None:
        events.append(ev)

    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "propose_inference",
                        {"statement": "Alice is methodical", "derivation": "observed pattern"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(
        db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama, on_event=on_event
    )
    sc = [e for e in events if e.event == "sidechannel"]
    assert any(e.data.get("type") == "inference_proposed" for e in sc)
    infer_event = next(e for e in sc if e.data.get("type") == "inference_proposed")
    assert infer_event.data["inference"]["statement"] == "Alice is methodical"


@respx.mock
async def test_propose_inference_multiple_calls_in_one_round_all_written(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [
                    (
                        "propose_inference",
                        {"statement": "Inference one", "derivation": "reason one"},
                    ),
                    (
                        "propose_inference",
                        {"statement": "Inference two", "derivation": "reason two"},
                    ),
                    ("report_pass", {}),
                ]
            ),
        )
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    inferences = await get_inferences(db, _CHARACTER.id)
    stmts = [i.statement for i in inferences]
    assert "Inference one" in stmts
    assert "Inference two" in stmts


# ---------------------------------------------------------------------------
# Cap and nudge behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_run_evaluator_nudge_round_fires_when_cap_reached_without_terminal_call(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            # Round 1: only set_fact, no terminal
            httpx.Response(
                200,
                content=make_tool_call_response(
                    "set_fact", {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
                ),
            ),
            # Nudge round: report_pass
            httpx.Response(200, content=make_tool_call_response("report_pass", {})),
        ]
    )
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    result, _ = await run_evaluator(
        db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama, max_rounds=1
    )
    assert result.verdict == "pass"
    assert route.call_count == 2


@respx.mock
async def test_run_evaluator_nudge_message_instructs_terminal_tool(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response(
                    "set_fact", {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
                ),
            ),
            httpx.Response(200, content=make_tool_call_response("report_pass", {})),
        ]
    )
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    await run_evaluator(db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama, max_rounds=1)
    # The nudge round's request body should contain the nudge system message
    nudge_body = json.loads(route.calls[1].request.content)
    all_content = " ".join(
        m.get("content", "") for m in nudge_body["messages"] if isinstance(m.get("content"), str)
    )
    assert "report_pass" in all_content or "report_contradiction" in all_content


@respx.mock
async def test_run_evaluator_falls_back_to_pass_when_nudge_also_fails(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            # Original round: only set_fact
            httpx.Response(
                200,
                content=make_tool_call_response(
                    "set_fact", {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
                ),
            ),
            # Nudge round: also only set_fact (still no terminal)
            httpx.Response(
                200,
                content=make_tool_call_response(
                    "set_fact", {"path": "Character.State-Of-Mind.Energy", "value": "Alert"}
                ),
            ),
        ]
    )
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    result, _ = await run_evaluator(
        db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama, max_rounds=1
    )
    assert result.verdict == "pass"
    assert "cap reached" in result.decision_log


@respx.mock
async def test_run_evaluator_logs_warning_on_cap_fallback(
    db: aiosqlite.Connection, ollama: OllamaClient, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response(
                    "set_fact", {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
                ),
            ),
            httpx.Response(
                200,
                content=make_tool_call_response(
                    "set_fact", {"path": "Character.State-Of-Mind.Energy", "value": "Alert"}
                ),
            ),
        ]
    )
    await set_facts(db, character_id=_CHARACTER.id, blob={})
    import logging

    with caplog.at_level(logging.WARNING):
        await run_evaluator(
            db, _CHARACTER, 1, 1, {}, _USER_MSG, _CHAR_RESPONSE, ollama, max_rounds=1
        )
    assert any("tool-call cap reached" in r.message for r in caplog.records)


@respx.mock
async def test_run_evaluator_no_nudge_round_when_terminal_call_made_within_budget(
    db: aiosqlite.Connection, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    await run_evaluator(db, _CHARACTER, 1, 1, _FACTS_BLOB, _USER_MSG, _CHAR_RESPONSE, ollama)
    assert route.call_count == 1


# ---------------------------------------------------------------------------
# Retained: violation model shape
# ---------------------------------------------------------------------------


def test_violation_has_no_suggested_fact_field() -> None:
    """Violation no longer carries a suggested_fact field — removed in Step 3."""
    v = Violation(type="contradiction", description="immutable fact was contradicted")
    assert not hasattr(v, "suggested_fact")
