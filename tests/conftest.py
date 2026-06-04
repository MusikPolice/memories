from collections.abc import AsyncGenerator

import aiosqlite
import pytest
from httpx import ASGITransport, AsyncClient

from memories.database import init_db
from memories.main import app


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
