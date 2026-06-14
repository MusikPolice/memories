from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

import memories.services.tool_gate as _tool_gate
from memories.database import init_db
from memories.main import app


@pytest.fixture(autouse=True)
def _clear_gates() -> Any:
    _tool_gate._pending.clear()
    yield
    _tool_gate._pending.clear()


@pytest.fixture
async def db() -> AsyncGenerator[aiosqlite.Connection, None]:
    """Fresh in-memory SQLite database with all tables created."""
    async with aiosqlite.connect(":memory:") as conn:
        await init_db(conn)
        yield conn


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
