"""Experiences API router."""

from __future__ import annotations

from typing import Annotated, Literal

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memories.database import (
    delete_experience,
    get_character,
    get_experience,
    get_experiences,
    get_session,
)
from memories.deps import get_db, get_ollama
from memories.exceptions import NotFoundError
from memories.models import Experience
from memories.services.experience_service import (
    embed_and_store,
    remove_experience_from_all_sessions,
)
from memories.services.ollama_client import OllamaClient

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _CreateBody(BaseModel):
    session_id: int
    statement: str
    source: Literal["told_by_user", "observed"]


@router.post("/{character_id}/experiences", status_code=201, response_model=Experience)
async def create_experience_endpoint(
    character_id: int, body: _CreateBody, db: _DB, ollama: _Ollama
) -> Experience:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    session = await get_session(db, body.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.character_id != character_id:
        raise HTTPException(status_code=404, detail="Session not found")
    return await embed_and_store(
        db,
        character_id=character_id,
        session_id=body.session_id,
        statement=body.statement,
        source=body.source,
        ollama=ollama,
    )


@router.get("/{character_id}/experiences", response_model=list[Experience])
async def list_experiences_endpoint(character_id: int, db: _DB) -> list[Experience]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return await get_experiences(db, character_id)


@router.delete("/{character_id}/experiences/{experience_id}", status_code=204)
async def delete_experience_endpoint(character_id: int, experience_id: int, db: _DB) -> None:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    experience = await get_experience(db, experience_id)
    if experience is None or experience.character_id != character_id:
        raise HTTPException(status_code=404, detail="Experience not found")
    try:
        await delete_experience(db, experience_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Experience not found") from exc
    remove_experience_from_all_sessions(experience_id)
