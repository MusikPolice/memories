"""Facts API router (mounted under /api/characters)."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Literal

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from memories.database import (
    create_fact,
    delete_fact,
    get_character,
    get_facts,
    get_inferences,
    patch_fact,
    update_fact,
)
from memories.deps import get_db
from memories.exceptions import NotFoundError
from memories.models import Fact, Inference

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]


class _CreateBody(BaseModel):
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    mutability: Literal["immutable", "low", "high"] = "immutable"


class _UpdateBody(BaseModel):
    value: str
    category: Literal["user", "character", "setting"] | None = None
    mutability: Literal["immutable", "low", "high"] | None = None


class _PatchBody(BaseModel):
    category: Literal["user", "character", "setting"] | None = None
    mutability: Literal["immutable", "low", "high"] | None = None


@router.get("/{character_id}/facts", response_model=list[Fact])
async def list_facts_endpoint(character_id: int, db: _DB) -> list[Fact]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return await get_facts(db, character_id)


@router.get("/{character_id}/inferences", response_model=list[Inference])
async def list_inferences_endpoint(
    character_id: int,
    db: _DB,
    status: str = Query(default="active"),
) -> list[Inference]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    return await get_inferences(db, character_id, status=status)


@router.post("/{character_id}/facts", status_code=201, response_model=Fact)
async def create_fact_endpoint(character_id: int, body: _CreateBody, db: _DB) -> Fact:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    try:
        return await create_fact(
            db,
            character_id=character_id,
            key=body.key,
            value=body.value,
            category=body.category,
            mutability=body.mutability,
        )
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail=f"Fact '{body.key}' already exists") from exc


@router.put("/{character_id}/facts/{key_or_id}", response_model=Fact)
async def update_fact_endpoint(
    character_id: int, key_or_id: str, body: _UpdateBody, db: _DB
) -> Fact:
    # If path segment is an integer, use id-based lookup with character ownership check.
    try:
        fact_id = int(key_or_id)
        row = await (
            await db.execute(
                "SELECT id FROM facts WHERE id = ? AND character_id = ?",
                (fact_id, character_id),
            )
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Fact not found")
        return await update_fact(
            db,
            fact_id=fact_id,
            value=body.value,
            category=body.category,
            mutability=body.mutability,
        )
    except ValueError:
        # Legacy key-based lookup
        key = key_or_id
        try:
            return await update_fact(db, character_id=character_id, key=key, value=body.value)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Fact '{key}' not found") from exc


@router.patch("/{character_id}/facts/{fact_id}", response_model=Fact)
async def patch_fact_endpoint(character_id: int, fact_id: int, body: _PatchBody, db: _DB) -> Fact:
    if body.category is None and body.mutability is None:
        raise HTTPException(
            status_code=422, detail="At least one of category or mutability must be provided"
        )
    # Verify fact belongs to this character
    row = await (
        await db.execute(
            "SELECT id FROM facts WHERE id = ? AND character_id = ?",
            (fact_id, character_id),
        )
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Fact not found")
    try:
        return await patch_fact(
            db, fact_id=fact_id, category=body.category, mutability=body.mutability
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Fact not found") from exc


@router.delete("/{character_id}/facts/{key_or_id}")
async def delete_fact_endpoint(character_id: int, key_or_id: str, db: _DB) -> dict[str, object]:
    from memories.services.inference_service import cascade_on_fact_delete

    # If path segment is an integer, delete by fact_id with ownership check.
    try:
        fact_id = int(key_or_id)
        row = await (
            await db.execute(
                "SELECT id FROM facts WHERE id = ? AND character_id = ?",
                (fact_id, character_id),
            )
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")
        await delete_fact(db, fact_id=fact_id)
    except ValueError:
        # Legacy key-based deletion
        key = key_or_id
        row = await (
            await db.execute(
                "SELECT id FROM facts WHERE character_id = ? AND key = ?",
                (character_id, key),
            )
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"Fact '{key}' not found") from None
        fact_id = row[0]
        try:
            await delete_fact(db, character_id=character_id, key=key)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Fact '{key}' not found") from exc

    invalidated = await cascade_on_fact_delete(db, character_id, fact_id)
    return {"invalidated_inferences": [inf.model_dump() for inf in invalidated]}
