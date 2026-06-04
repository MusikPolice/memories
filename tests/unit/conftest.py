"""Unit-test fixtures: pre-populated DB objects and Ollama mock helpers."""

from __future__ import annotations

import json

import aiosqlite
import pytest

from memories.database import create_character, create_fact, create_session
from memories.models import Character, Fact, Session
from memories.services.ollama_client import OllamaClient

# Base URL used by all unit tests that mock the Ollama HTTP layer.
OLLAMA_BASE_URL = "http://test-ollama:11434"


def make_ollama_ndjson(*chunks: str, prompt_eval_count: int = 10, eval_count: int = 5) -> bytes:
    """Build a minimal Ollama NDJSON streaming body from content *chunks*.

    The last chunk carries ``done: true`` and the token-count metadata fields.
    """
    lines: list[str] = []
    for i, text in enumerate(chunks):
        is_last = i == len(chunks) - 1
        obj: dict[str, object] = {
            "message": {"role": "assistant", "content": text},
            "done": is_last,
        }
        if is_last:
            obj["prompt_eval_count"] = prompt_eval_count
            obj["eval_count"] = eval_count
        lines.append(json.dumps(obj))
    return ("\n".join(lines) + "\n").encode()


@pytest.fixture
async def character(db: aiosqlite.Connection) -> Character:
    return await create_character(db, name="Alice", modelfile_base="qwen3:7b")


@pytest.fixture
async def session(db: aiosqlite.Connection, character: Character) -> Session:
    return await create_session(db, character_id=character.id)


@pytest.fixture
async def fact(db: aiosqlite.Connection, character: Character) -> Fact:
    return await create_fact(db, character_id=character.id, key="occupation", value="surgeon")


@pytest.fixture
def ollama() -> OllamaClient:
    return OllamaClient(base_url=OLLAMA_BASE_URL)
