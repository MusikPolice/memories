"""Sessions API router."""

from __future__ import annotations

from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memories.database import create_session, end_session, get_character, get_messages, get_session
from memories.deps import get_db
from memories.exceptions import NotFoundError
from memories.models import Message, Session

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]


class _CreateBody(BaseModel):
    character_id: int


@router.post("/", status_code=201, response_model=Session)
async def create_session_endpoint(body: _CreateBody, db: _DB) -> Session:
    character = await get_character(db, body.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return await create_session(db, character_id=body.character_id)


@router.post("/{session_id}/end", response_model=Session)
async def end_session_endpoint(session_id: int, db: _DB) -> Session:
    try:
        return await end_session(db, session_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc


@router.get("/{session_id}/messages", response_model=list[Message])
async def get_session_messages_endpoint(session_id: int, db: _DB) -> list[Message]:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await get_messages(db, session_id)
