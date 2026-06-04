"""FastAPI dependency providers for DB and Ollama."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import aiosqlite

from memories.services.ollama_client import OllamaClient

_db: aiosqlite.Connection | None = None


def set_db(conn: aiosqlite.Connection | None) -> None:
    global _db
    _db = conn


async def get_db() -> AsyncGenerator[aiosqlite.Connection, None]:
    if _db is None:
        raise RuntimeError("Database not initialized")
    yield _db


def get_ollama() -> OllamaClient:
    return OllamaClient()
