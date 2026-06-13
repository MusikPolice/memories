"""Live tests for OllamaClient.chat_with_tools() against a real Ollama server.

Run with:
    LIVE_OLLAMA=1 uv run pytest tests/live/ -v

All tests are skipped unless LIVE_OLLAMA=1 is set in the environment.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest

from memories.services.ollama_client import OllamaClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.getenv("LIVE_OLLAMA"),
        reason="Set LIVE_OLLAMA=1 to run against a real Ollama server",
    ),
]

_SYSTEM_PROMPT = (
    "You are a fact-extraction assistant. When the user's message implies a fact about "
    "the character, call set_fact to record it. Do not describe what you would do — "
    "call the tool. Call it once per distinct fact implied."
)


async def test_model_calls_tool_not_prose(
    live_ollama: OllamaClient,
    live_model: str,
    set_fact_tool: dict[str, Any],
) -> None:
    call_log: list[tuple[str, str]] = []

    def handler(args: dict[str, Any]) -> str:
        call_log.append((args.get("path", ""), args.get("value", "")))
        return "OK"

    result = await live_ollama.chat_with_tools(
        model=live_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "Her mood seems really anxious right now."},
        ],
        tools=[set_fact_tool],
        tool_handlers={"set_fact": handler},
    )

    assert len(call_log) >= 1, "Model did not make any tool calls — check model/prompt"
    assert result.rounds >= 2
    assert result.cap_reached is False


async def test_model_extracts_multiple_facts(
    live_ollama: OllamaClient,
    live_model: str,
    set_fact_tool: dict[str, Any],
) -> None:
    call_log: list[tuple[str, str]] = []

    def handler(args: dict[str, Any]) -> str:
        call_log.append((args.get("path", ""), args.get("value", "")))
        return "OK"

    result = await live_ollama.chat_with_tools(
        model=live_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "She's wearing a blue surgical scrub top and her mood is calm.",
            },
        ],
        tools=[set_fact_tool],
        tool_handlers={"set_fact": handler},
    )

    assert len(call_log) >= 2, "Expected at least two tool calls (outfit + mood)"
    paths = [path.lower() for path, _ in call_log]
    assert any("outfit" in p or "top" in p for p in paths), "No outfit path found in calls"
    assert any("mood" in p for p in paths), "No mood path found in calls"
    assert result.cap_reached is False


async def test_model_retries_after_tool_error(
    live_ollama: OllamaClient,
    live_model: str,
    set_fact_tool: dict[str, Any],
) -> None:
    call_count = [0]

    def handler(args: dict[str, Any]) -> str:
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Validation failed on first attempt — please retry")
        return "OK"

    result = await live_ollama.chat_with_tools(
        model=live_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "She's feeling really anxious."},
        ],
        tools=[set_fact_tool],
        tool_handlers={"set_fact": handler},
    )

    assert call_count[0] >= 2, "Model did not retry after error"
    assert result.content, "Model produced no final response"
    assert result.cap_reached is False


async def test_latency_within_acceptable_range(
    live_ollama: OllamaClient,
    live_model: str,
    set_fact_tool: dict[str, Any],
) -> None:
    def handler(args: dict[str, Any]) -> str:
        return "OK"

    start = time.monotonic()
    result = await live_ollama.chat_with_tools(
        model=live_model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "She seems a bit tired and is wearing a white lab coat.",
            },
        ],
        tools=[set_fact_tool],
        tool_handlers={"set_fact": handler},
    )
    elapsed = time.monotonic() - start

    assert (
        elapsed < 60
    ), f"chat_with_tools took {elapsed:.1f}s — investigate for hang or infinite loop"
    assert result.cap_reached is False
