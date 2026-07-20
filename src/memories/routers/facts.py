"""Facts API router (mounted under /api/characters)."""

from __future__ import annotations

from typing import Annotated, Any

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from memories.database import (
    get_character,
    get_facts,
    get_inferences,
    patch_fact,
)
from memories.deps import get_db
from memories.models import Inference
from memories.schema_loader import _collect_leaves, check_write_permitted, load_schema

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]


@router.get("/{character_id}/facts")
async def get_facts_endpoint(character_id: int, db: _DB) -> dict[str, Any]:
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


class _SetFactBody(BaseModel):
    path: str
    value: str


@router.put("/{character_id}/facts")
async def set_fact_value_endpoint(character_id: int, body: _SetFactBody, db: _DB) -> dict[str, Any]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    schema = load_schema()
    try:
        check_write_permitted(body.path, schema)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    leaf = dict(_collect_leaves(schema))[body.path]
    coerced: str | int = body.value
    if leaf["Type"] == "Enum":
        match = next((c for c in leaf["Constraint"] if c.lower() == body.value.lower()), None)
        if match is None:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"{body.value!r} is not a valid value for {body.path}. "
                    f"Valid values: {', '.join(leaf['Constraint'])}"
                ),
            )
        coerced = match
    elif leaf["Type"] == "Integer":
        try:
            coerced = int(body.value)
        except ValueError as exc:
            raise HTTPException(
                status_code=422, detail=f"{body.value!r} is not a valid integer"
            ) from exc

    await patch_fact(db, character_id, tuple(body.path.split(".")), coerced)
    return await get_facts(db, character_id)
