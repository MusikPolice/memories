from __future__ import annotations

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from memories import deps
from memories.database import init_db, list_characters
from memories.routers import characters, chat, facts, sessions
from memories.services.ollama_client import OllamaClient, OllamaConnectionError, OllamaResponseError

_log = logging.getLogger(__name__)


async def _warmup_models(db: aiosqlite.Connection) -> None:
    """Preload every model referenced by an existing character into Ollama's memory."""
    chars = await list_characters(db)
    models = {c.current_model_name or c.modelfile_base for c in chars}
    if not models:
        return
    ollama = OllamaClient()
    for model in models:
        _log.info("warming up model: %s", model)
        try:
            await ollama.warmup(model)
            _log.info("model ready: %s", model)
        except (OllamaConnectionError, OllamaResponseError) as exc:
            _log.warning("could not warm up %s (%s) — will load on first request", model, exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    db_path = os.getenv("MEMORIES_DB_PATH", "memories.db")
    async with aiosqlite.connect(db_path) as conn:
        await init_db(conn)
        deps.set_db(conn)
        await _warmup_models(conn)
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
