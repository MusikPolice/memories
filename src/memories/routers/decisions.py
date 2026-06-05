"""Decisions audit log endpoint."""

from __future__ import annotations

from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException

from memories.database import get_decisions, get_session
from memories.deps import get_db
from memories.models import Decision

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]


@router.get("/{session_id}/decisions")
async def list_decisions(session_id: int, db: _DB) -> list[Decision]:
    """Return all decisions for a session, most recent turn first."""
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await get_decisions(db, session_id)
