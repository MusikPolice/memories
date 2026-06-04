"""Chat SSE endpoint."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from memories.database import get_session, next_turn_id
from memories.deps import get_db, get_ollama
from memories.services.chat_service import run_turn
from memories.services.ollama_client import OllamaClient

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _SendBody(BaseModel):
    content: str


@router.post("/{session_id}/messages")
async def send_message(
    session_id: int, body: _SendBody, db: _DB, ollama: _Ollama
) -> StreamingResponse:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at is not None:
        raise HTTPException(status_code=409, detail="Session has ended")

    turn_id = await next_turn_id(db, session_id)

    async def _stream() -> AsyncGenerator[str, None]:
        yield 'event: status\ndata: {"state": "generating"}\n\n'
        content = await run_turn(db, session_id, body.content, ollama)
        data = json.dumps({"role": "assistant", "content": content, "turn_id": turn_id})
        yield f"event: message\ndata: {data}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
