"""Characters API router."""

from __future__ import annotations

from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memories.database import create_character, get_character, list_characters
from memories.deps import get_db
from memories.models import Character

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]


class _CreateBody(BaseModel):
    name: str
    modelfile_base: str


@router.post("/", status_code=201, response_model=Character)
async def create_character_endpoint(body: _CreateBody, db: _DB) -> Character:
    return await create_character(db, name=body.name, modelfile_base=body.modelfile_base)


@router.get("/", response_model=list[Character])
async def list_characters_endpoint(db: _DB) -> list[Character]:
    return await list_characters(db)


@router.get("/{character_id}", response_model=Character)
async def get_character_endpoint(character_id: int, db: _DB) -> Character:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return character
