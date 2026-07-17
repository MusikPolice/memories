"""Unit tests for OllamaClient.chat_with_tools()."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from memories.services.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_multi_tool_call_response,
    make_plain_tool_response,
    make_tool_call_response,
    make_tool_call_response_with_thinking,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_SET_FACT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_fact",
        "description": "Set the value of a fact at the given schema path",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["path", "value"],
        },
    },
}

_TOOLS = [_SET_FACT_TOOL]
_MESSAGES = [{"role": "user", "content": "Set the mood to Anxious"}]
_DEFAULT_ARGS: dict[str, Any] = {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}


async def _ok_handler(args: dict[str, Any]) -> str:
    return "OK"


# ---------------------------------------------------------------------------
# Request shape tests
# ---------------------------------------------------------------------------


@respx.mock
async def test_tools_list_sent_in_request_body(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    body = json.loads(route.calls[0].request.content)
    assert "tools" in body
    assert body["tools"] == _TOOLS


@respx.mock
async def test_stream_false_used_for_tool_calls(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    body = json.loads(route.calls[0].request.content)
    assert body.get("stream") is False


@respx.mock
async def test_tools_and_stream_sent_in_all_rounds(ollama: OllamaClient) -> None:
    """tools and stream:false must be present in every round, not just the first."""
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    assert len(route.calls) == 2
    for call in route.calls:
        body = json.loads(call.request.content)
        assert body["tools"] == _TOOLS
        assert body["stream"] is False


# ---------------------------------------------------------------------------
# Return value and loop behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_no_tool_calls_returns_plain_content_directly(ollama: OllamaClient) -> None:
    call_log: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> str:
        call_log.append(args)
        return "OK"

    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_plain_tool_response("Hello!"))
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": handler})

    assert result.content == "Hello!"
    assert result.rounds == 1
    assert result.cap_reached is False
    assert len(call_log) == 0


@respx.mock
async def test_single_tool_call_then_plain_content(ollama: OllamaClient) -> None:
    call_log: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> str:
        call_log.append(args)
        return "OK"

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done setting mood.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": handler})

    assert len(call_log) == 1
    assert call_log[0] == _DEFAULT_ARGS
    assert result.content == "Done setting mood."
    assert result.rounds == 2
    assert result.cap_reached is False


@respx.mock
async def test_multiple_sequential_tool_calls(ollama: OllamaClient) -> None:
    call_log: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> str:
        call_log.append(args)
        return "OK"

    args1: dict[str, Any] = {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
    args2: dict[str, Any] = {"path": "Character.State-Of-Mind.Energy", "value": "Tired"}
    args3: dict[str, Any] = {"path": "Character.Identity.Name", "value": "Sarah"}

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", args1)),
            httpx.Response(200, content=make_tool_call_response("set_fact", args2)),
            httpx.Response(200, content=make_tool_call_response("set_fact", args3)),
            httpx.Response(200, content=make_plain_tool_response("All done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": handler})

    assert len(call_log) == 3
    assert call_log[0] == args1
    assert call_log[1] == args2
    assert call_log[2] == args3
    assert result.rounds == 4
    assert result.cap_reached is False


@respx.mock
async def test_multiple_tool_calls_in_single_response(ollama: OllamaClient) -> None:
    call_log: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> str:
        call_log.append(args)
        return "OK"

    args1: dict[str, Any] = {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
    args2: dict[str, Any] = {"path": "Character.State-Of-Mind.Energy", "value": "Tired"}

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_multi_tool_call_response([("set_fact", args1), ("set_fact", args2)]),
            ),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": handler})

    assert len(call_log) == 2
    assert call_log[0] == args1
    assert call_log[1] == args2
    tool_msgs = [m for m in result.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert result.rounds == 2


# ---------------------------------------------------------------------------
# History structure
# ---------------------------------------------------------------------------


@respx.mock
async def test_history_contains_all_messages(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    assert len(result.history) == 4
    # 1. Original user message
    assert result.history[0] == _MESSAGES[0]
    # 2. Assistant tool-call message
    assert result.history[1]["role"] == "assistant"
    assert result.history[1]["tool_calls"] is not None
    # 3. Tool result message
    assert result.history[2]["role"] == "tool"
    assert result.history[2]["content"] == "OK"
    # 4. Final assistant message
    assert result.history[3]["role"] == "assistant"
    assert result.history[3]["content"] == "Done."


@respx.mock
async def test_input_messages_not_mutated(ollama: OllamaClient) -> None:
    """chat_with_tools must not modify the caller's messages list."""
    original = [{"role": "user", "content": "Set the mood to Anxious"}]
    messages_copy = list(original)

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    await ollama.chat_with_tools("qwen3:7b", messages_copy, _TOOLS, {"set_fact": _ok_handler})

    assert messages_copy == original


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@respx.mock
async def test_handler_exception_becomes_error_string(ollama: OllamaClient) -> None:
    call_count = [0]

    async def handler(args: dict[str, Any]) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("'Apprehensive' is not a valid value. Valid values are: Calm, Anxious")
        return "OK"

    args2: dict[str, Any] = {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", args2)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": handler})

    tool_msgs = [m for m in result.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    error_content = tool_msgs[0]["content"]
    assert error_content.startswith("Error:")
    assert "'Apprehensive' is not a valid value" in error_content
    assert "Traceback" not in error_content
    assert call_count[0] == 2
    assert result.rounds == 3
    assert result.content == "Done."


@respx.mock
async def test_unknown_tool_name_returns_error_to_model(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200, content=make_tool_call_response("nonexistent_tool", {"foo": "bar"})
            ),
            httpx.Response(200, content=make_plain_tool_response("I'll try differently.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    tool_msgs = [m for m in result.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Unknown tool" in tool_msgs[0]["content"]
    assert "nonexistent_tool" in tool_msgs[0]["content"]
    assert result.content == "I'll try differently."


# ---------------------------------------------------------------------------
# Cap behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_cap_reached_returns_cap_reached_true(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
        ]
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, max_rounds=3
    )

    assert result.cap_reached is True
    assert result.rounds == 3
    assert len(route.calls) == 3


@respx.mock
async def test_cap_reached_content_is_empty_string(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
        ]
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, max_rounds=3
    )

    assert result.content == ""


@respx.mock
async def test_cap_reached_history_contains_all_rounds(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
        ]
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, max_rounds=2
    )

    # original + 2 rounds x (tool-call msg + tool-result msg) = 1 + 4 = 5
    assert len(result.history) == 5
    assert result.history[0] == _MESSAGES[0]
    assert result.history[1]["role"] == "assistant"  # round 1 tool-call
    assert result.history[2]["role"] == "tool"  # round 1 tool-result
    assert result.history[3]["role"] == "assistant"  # round 2 tool-call
    assert result.history[4]["role"] == "tool"  # round 2 tool-result


# ---------------------------------------------------------------------------
# Infrastructure errors
# ---------------------------------------------------------------------------


@respx.mock
async def test_connection_error_propagates(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
    with pytest.raises(OllamaConnectionError):
        await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {})


@respx.mock
async def test_non_200_response_propagates(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(500, content=b"Internal Server Error"))
    with pytest.raises(OllamaResponseError):
        await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {})


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@respx.mock
async def test_tool_call_with_none_tool_calls_field(ollama: OllamaClient) -> None:
    call_log: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> str:
        call_log.append(args)
        return "OK"

    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_plain_tool_response("Hello."))
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": handler})

    assert result.rounds == 1
    assert result.cap_reached is False
    assert len(call_log) == 0


@respx.mock
async def test_tool_call_id_propagated_to_result_message(ollama: OllamaClient) -> None:
    """When the model assigns an id to a tool call, it must appear as tool_call_id
    in the corresponding tool result message.  Without this, models that rely on
    ID correlation (e.g. qwen3) loop indefinitely because they never learn which
    calls were resolved.
    """
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response("set_fact", _DEFAULT_ARGS, call_id="call_abc123"),
            ),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    tool_msgs = [m for m in result.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].get("tool_call_id") == "call_abc123"
    assert result.cap_reached is False


@respx.mock
async def test_multi_tool_call_ids_propagated_to_result_messages(ollama: OllamaClient) -> None:
    """Each result message must carry the tool_call_id that matches its call,
    so the model can correlate results to calls when several fire in one round.
    """
    args1: dict[str, Any] = {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
    args2: dict[str, Any] = {"path": "Character.State-Of-Mind.Energy", "value": "Tired"}

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_multi_tool_call_response(
                    [("set_fact", args1), ("set_fact", args2)],
                    call_ids=["call_001", "call_002"],
                ),
            ),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    tool_msgs = [m for m in result.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0].get("tool_call_id") == "call_001"
    assert tool_msgs[1].get("tool_call_id") == "call_002"
    assert result.cap_reached is False


@respx.mock
async def test_tool_result_has_no_id_when_call_has_no_id(ollama: OllamaClient) -> None:
    """When the model omits the id field (single-call models), no tool_call_id
    should be added to the result — an unexpected key would confuse those models.
    """
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response("set_fact", _DEFAULT_ARGS),  # no call_id
            ),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    tool_msgs = [m for m in result.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "tool_call_id" not in tool_msgs[0]


@respx.mock
async def test_thinking_stripped_from_history(ollama: OllamaClient) -> None:
    """Thinking tokens must not be fed back to the model as conversation context.

    The qwen3 family loops indefinitely when its own thinking tokens appear in
    history — they cause the next turn's generation to keep emitting tool calls
    rather than producing a plain-text response.
    """
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response_with_thinking(
                    "set_fact", _DEFAULT_ARGS, thinking="I should record the mood."
                ),
            ),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    assert all("thinking" not in m for m in result.history)
    assert result.cap_reached is False
    assert result.content == "Done."


@respx.mock
async def test_handler_is_callable_with_arguments_dict(ollama: OllamaClient) -> None:
    received: list[Any] = []

    async def handler(args: dict[str, Any]) -> str:
        received.append(args)
        return "OK"

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": handler})

    assert len(received) == 1
    assert isinstance(received[0], dict)


# ---------------------------------------------------------------------------
# Terminal tool behaviour
# ---------------------------------------------------------------------------

_REPORT_PASS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {"name": "report_pass", "parameters": {"type": "object", "properties": {}}},
}

_REPORT_CONTRADICTION_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "report_contradiction",
        "parameters": {"type": "object", "properties": {}},
    },
}

_TERMINAL_TOOLS_BOTH = frozenset({"report_pass", "report_contradiction"})


@respx.mock
async def test_terminal_tools_none_preserves_existing_behaviour(
    ollama: OllamaClient,
) -> None:
    """When terminal_tools is None, plain-content response ends loop with terminal_call=None."""
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_plain_tool_response("Hello!"))
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, terminal_tools=None
    )

    assert result.terminal_call is None
    assert result.cap_reached is False
    assert result.content == "Hello!"


@respx.mock
async def test_terminal_call_set_when_terminal_tool_invoked(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_tool_call_response("report_pass", {}))
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b",
        _MESSAGES,
        [_SET_FACT_TOOL, _REPORT_PASS_TOOL],
        {"set_fact": _ok_handler, "report_pass": _ok_handler},
        terminal_tools=frozenset({"report_pass"}),
    )

    assert result.terminal_call is not None
    assert result.terminal_call["name"] == "report_pass"
    assert result.terminal_call["arguments"] == {}
    assert result.terminal_call["result"] == "OK"
    assert result.cap_reached is False


@respx.mock
async def test_terminal_call_stops_loop_without_further_http_calls(
    ollama: OllamaClient,
) -> None:
    """After a terminal tool fires, no additional HTTP round-trip should occur."""
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("report_pass", {})),
        ]
    )
    await ollama.chat_with_tools(
        "qwen3:7b",
        _MESSAGES,
        [_REPORT_PASS_TOOL],
        {"report_pass": _ok_handler},
        terminal_tools=frozenset({"report_pass"}),
    )

    assert route.call_count == 1


@respx.mock
async def test_terminal_call_processes_other_calls_in_same_round_first(
    ollama: OllamaClient,
) -> None:
    """Non-terminal calls in the same batch are executed before the loop stops."""
    set_fact_log: list[dict[str, Any]] = []

    async def _log_handler(args: dict[str, Any]) -> str:
        set_fact_log.append(args)
        return "OK"

    sf_args: dict[str, Any] = {"path": "Character.State-Of-Mind.Mood", "value": "Anxious"}
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response([("set_fact", sf_args), ("report_pass", {})]),
        )
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b",
        _MESSAGES,
        [_SET_FACT_TOOL, _REPORT_PASS_TOOL],
        {"set_fact": _log_handler, "report_pass": _ok_handler},
        terminal_tools=frozenset({"report_pass"}),
    )

    assert len(set_fact_log) == 1
    assert set_fact_log[0] == sf_args
    assert result.terminal_call is not None
    assert result.terminal_call["name"] == "report_pass"


@respx.mock
async def test_terminal_call_takes_first_match_when_multiple_in_one_round(
    ollama: OllamaClient,
) -> None:
    """When multiple terminal tools fire in one round, the first in call order wins."""
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_multi_tool_call_response(
                [("report_pass", {}), ("report_contradiction", {})]
            ),
        )
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b",
        _MESSAGES,
        [_REPORT_PASS_TOOL, _REPORT_CONTRADICTION_TOOL_SCHEMA],
        {"report_pass": _ok_handler, "report_contradiction": _ok_handler},
        terminal_tools=_TERMINAL_TOOLS_BOTH,
    )

    assert result.terminal_call is not None
    assert result.terminal_call["name"] == "report_pass"


@respx.mock
async def test_non_terminal_tool_calls_continue_looping_as_before(
    ollama: OllamaClient,
) -> None:
    """`terminal_tools` given but model only calls set_fact → loops until plain content."""
    call_log: list[dict[str, Any]] = []

    async def handler(args: dict[str, Any]) -> str:
        call_log.append(args)
        return "OK"

    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b",
        _MESSAGES,
        [_SET_FACT_TOOL, _REPORT_PASS_TOOL],
        {"set_fact": handler, "report_pass": _ok_handler},
        terminal_tools=frozenset({"report_pass"}),
    )

    assert result.terminal_call is None
    assert result.content == "Done."
    assert len(call_log) == 1


@respx.mock
async def test_cap_reached_with_terminal_tools_and_no_terminal_call(
    ollama: OllamaClient,
) -> None:
    """`max_rounds` exhausted without a terminal tool → cap_reached=True, terminal_call=None."""
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
        ]
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b",
        _MESSAGES,
        [_SET_FACT_TOOL, _REPORT_PASS_TOOL],
        {"set_fact": _ok_handler, "report_pass": _ok_handler},
        max_rounds=2,
        terminal_tools=frozenset({"report_pass"}),
    )

    assert result.cap_reached is True
    assert result.terminal_call is None


# ---------------------------------------------------------------------------
# think / thinking behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_think_false_sent_by_default(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_plain_tool_response("Hello!"))
    )
    await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    body = json.loads(route.calls[0].request.content)
    assert body["think"] is False


@respx.mock
async def test_think_true_sent_when_requested(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_plain_tool_response("Hello!"))
    )
    await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, think=True
    )

    body = json.loads(route.calls[0].request.content)
    assert body["think"] is True


@respx.mock
async def test_think_sent_in_every_round(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, think=True
    )

    assert len(route.calls) == 2
    for call in route.calls:
        body = json.loads(call.request.content)
        assert body["think"] is True


@respx.mock
async def test_thinking_empty_by_default(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_plain_tool_response("Hello!"))
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    assert result.thinking == ""


@respx.mock
async def test_thinking_captured_from_plain_content_response(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_plain_tool_response("Hi.", thinking="Considering...")
        )
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, think=True
    )

    assert result.thinking == "Considering..."
    assert result.content == "Hi."


@respx.mock
async def test_thinking_accumulated_across_tool_and_final_rounds(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response_with_thinking(
                    "set_fact", _DEFAULT_ARGS, thinking="Step one."
                ),
            ),
            httpx.Response(200, content=make_plain_tool_response("Done.", thinking="Step two.")),
        ]
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler}, think=True
    )

    assert result.thinking == "Step one.Step two."
    assert result.content == "Done."


@respx.mock
async def test_thinking_still_stripped_from_history(ollama: OllamaClient) -> None:
    """Thinking is captured on ToolCallResult but not fed back as conversation context."""
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                content=make_tool_call_response_with_thinking(
                    "set_fact", _DEFAULT_ARGS, thinking="I should record the mood."
                ),
            ),
            httpx.Response(200, content=make_plain_tool_response("Done.")),
        ]
    )
    result = await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": _ok_handler})

    assert all("thinking" not in m for m in result.history)
    assert result.thinking != ""
