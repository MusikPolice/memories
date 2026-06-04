"""Integration-test fixtures: overrides the DB and Ollama dependencies."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

from memories.database import create_character, create_fact, create_session
from memories.deps import get_db, get_ollama
from memories.main import app
from memories.models import Character, Fact, Session
from memories.services.ollama_client import OllamaClient

OLLAMA_BASE_URL = "http://test-ollama-integration:11434"


@pytest.fixture
async def client(db: aiosqlite.Connection) -> AsyncGenerator[AsyncClient, None]:
    async def _get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
        yield db

    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_ollama] = lambda: OllamaClient(base_url=OLLAMA_BASE_URL)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
async def character(db: aiosqlite.Connection) -> Character:
    return await create_character(db, name="Alice", modelfile_base="qwen3:7b")


@pytest.fixture
async def session(db: aiosqlite.Connection, character: Character) -> Session:
    return await create_session(db, character_id=character.id)


@pytest.fixture
async def fact(db: aiosqlite.Connection, character: Character) -> Fact:
    return await create_fact(db, character_id=character.id, key="occupation", value="surgeon")
