"""Fixtures for live Ollama integration tests."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from typing import Any

import pytest

from memories.services.ollama_client import OllamaClient

LIVE_MODEL = os.getenv("LIVE_OLLAMA_MODEL", "jaahas/qwen3.5-uncensored:latest")


@pytest.fixture
async def live_ollama() -> AsyncGenerator[OllamaClient, None]:
    client = OllamaClient()
    yield client
    await client.aclose()


@pytest.fixture
def live_model() -> str:
    return LIVE_MODEL


@pytest.fixture
def set_fact_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "set_fact",
            "description": "Record a fact about the character or setting at the given schema path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Dot-notation path, e.g. 'Character.State-Of-Mind.Mood'",
                    },
                    "value": {"type": "string"},
                },
                "required": ["path", "value"],
            },
        },
    }
