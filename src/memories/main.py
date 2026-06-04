from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from memories import deps
from memories.database import init_db
from memories.routers import characters, chat, facts, sessions


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_path = os.getenv("MEMORIES_DB_PATH", "memories.db")
    async with aiosqlite.connect(db_path) as conn:
        await init_db(conn)
        deps.set_db(conn)
        yield
    deps.set_db(None)


app = FastAPI(title="Memories", lifespan=lifespan)

app.include_router(characters.router, prefix="/api/characters", tags=["characters"])
app.include_router(facts.router, prefix="/api/characters", tags=["facts"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(chat.router, prefix="/api/sessions", tags=["chat"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


_FRONTEND = Path(__file__).parent / "frontend"
if _FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
