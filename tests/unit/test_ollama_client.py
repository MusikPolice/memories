"""Unit tests for memories.services.ollama_client.OllamaClient."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from memories.services.ollama_client import (
    OllamaClient,
    OllamaConnectionError,
    OllamaResponseError,
)
from tests.unit.conftest import OLLAMA_BASE_URL, make_ollama_ndjson

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
_SAMPLE_MESSAGES = [{"role": "user", "content": "Hello"}]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@respx.mock
async def test_request_sends_model_and_messages(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("Hi"))
    )
    await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)

    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:7b"
    assert body["messages"] == _SAMPLE_MESSAGES


@respx.mock
async def test_request_uses_stream_true(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("Hi"))
    )
    await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)

    body = json.loads(route.calls[0].request.content)
    assert body.get("stream") is True


@respx.mock
async def test_chunks_are_concatenated(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("Hello", " world", "!"))
    )
    content, _ = await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)
    assert content == "Hello world!"


@respx.mock
async def test_special_tokens_truncate_content(ollama: OllamaClient) -> None:
    """Content after the first special token (and the token itself) is dropped."""
    raw = "Good answer.<|endoftext|><|im_start|>user Repeated text"
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=make_ollama_ndjson(raw)))
    content, _ = await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)
    assert content == "Good answer."


@respx.mock
async def test_thinking_returned_in_metadata(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_ollama_ndjson("The answer is 42.", thinking="Let me reason through this."),
        )
    )
    content, metadata = await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)
    assert content == "The answer is 42."
    assert metadata.get("thinking") == "Let me reason through this."


@respx.mock
async def test_thinking_empty_string_when_absent(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("Hello."))
    )
    _, metadata = await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)
    assert metadata.get("thinking") == ""


@respx.mock
async def test_returns_final_chunk_metadata(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200,
            content=make_ollama_ndjson("Hello", " there", prompt_eval_count=42, eval_count=17),
        )
    )
    _, metadata = await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)

    assert metadata.get("prompt_eval_count") == 42
    assert metadata.get("eval_count") == 17


@respx.mock
async def test_raises_ollama_connection_error_on_network_failure(
    ollama: OllamaClient,
) -> None:
    respx.post(_CHAT_URL).mock(side_effect=httpx.ConnectError("Connection refused"))
    with pytest.raises(OllamaConnectionError):
        await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)


@respx.mock
async def test_raises_ollama_response_error_on_non_200(
    ollama: OllamaClient,
) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(500, content=b"Internal Server Error"))
    with pytest.raises(OllamaResponseError):
        await ollama.chat("qwen3:7b", _SAMPLE_MESSAGES)


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


@respx.mock
async def test_warmup_posts_to_generate_endpoint(ollama: OllamaClient) -> None:
    route = respx.post(_GENERATE_URL).mock(return_value=httpx.Response(200, json={"done": True}))
    await ollama.warmup("qwen3:7b")
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "qwen3:7b"


@respx.mock
async def test_warmup_raises_connection_error(ollama: OllamaClient) -> None:
    respx.post(_GENERATE_URL).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(OllamaConnectionError):
        await ollama.warmup("qwen3:7b")


@respx.mock
async def test_warmup_raises_response_error_on_non_200(ollama: OllamaClient) -> None:
    respx.post(_GENERATE_URL).mock(return_value=httpx.Response(404, content=b"Not Found"))
    with pytest.raises(OllamaResponseError):
        await ollama.warmup("qwen3:7b")
