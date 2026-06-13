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
    await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": lambda a: "OK"})

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
    await ollama.chat_with_tools("qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": lambda a: "OK"})

    body = json.loads(route.calls[0].request.content)
    assert body.get("stream") is False


# ---------------------------------------------------------------------------
# Return value and loop behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_no_tool_calls_returns_plain_content_directly(ollama: OllamaClient) -> None:
    call_log: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> str:
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

    def handler(args: dict[str, Any]) -> str:
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

    def handler(args: dict[str, Any]) -> str:
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
    assert result.rounds == 4
    assert result.cap_reached is False


@respx.mock
async def test_multiple_tool_calls_in_single_response(ollama: OllamaClient) -> None:
    call_log: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> str:
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
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": lambda a: "OK"}
    )

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


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@respx.mock
async def test_handler_exception_becomes_error_string(ollama: OllamaClient) -> None:
    call_count = [0]

    def handler(args: dict[str, Any]) -> str:
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
    assert tool_msgs[0]["content"].startswith("Error:")
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
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": lambda a: "OK"}
    )

    tool_msgs = [m for m in result.history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "Unknown tool" in tool_msgs[0]["content"]
    assert result.content == "I'll try differently."


# ---------------------------------------------------------------------------
# Cap behaviour
# ---------------------------------------------------------------------------


@respx.mock
async def test_cap_reached_returns_cap_reached_true(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        side_effect=[
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
            httpx.Response(200, content=make_tool_call_response("set_fact", _DEFAULT_ARGS)),
        ]
    )
    result = await ollama.chat_with_tools(
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": lambda a: "OK"}, max_rounds=3
    )

    assert result.cap_reached is True
    assert result.rounds == 3


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
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": lambda a: "OK"}, max_rounds=3
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
        "qwen3:7b", _MESSAGES, _TOOLS, {"set_fact": lambda a: "OK"}, max_rounds=2
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

    def handler(args: dict[str, Any]) -> str:
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
async def test_handler_is_callable_with_arguments_dict(ollama: OllamaClient) -> None:
    received: list[Any] = []

    def handler(args: dict[str, Any]) -> str:
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
