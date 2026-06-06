"""Inference management API router (mounted under /api/characters)."""

from __future__ import annotations

from typing import Annotated, Literal

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memories.database import (
    _parse_inference,
    _row,
    delete_inference,
    get_character,
    get_facts,
    get_inferences,
    update_inference_status,
)
from memories.deps import get_db, get_ollama
from memories.exceptions import NotFoundError
from memories.models import Fact, Inference
from memories.services.inference_service import InferenceParseError
from memories.services.ollama_client import OllamaClient

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _RevalidateBody(BaseModel):
    changed_fact_id: int


class _PatchBody(BaseModel):
    status: str


class _PromoteBody(BaseModel):
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    mutability: Literal["immutable", "low", "high"] = "immutable"


async def _fact_exists(db: aiosqlite.Connection, fact_id: int) -> bool:
    row = await (await db.execute("SELECT id FROM facts WHERE id = ?", (fact_id,))).fetchone()
    return row is not None


@router.post("/{character_id}/inferences/generate")
async def generate_inferences_endpoint(
    character_id: int,
    db: _DB,
    ollama: _Ollama,
) -> dict[str, object]:
    from memories.services.inference_service import run_eager_pass

    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    facts = await get_facts(db, character_id)
    existing_inferences = await get_inferences(db, character_id, status="all")

    try:
        new_inferences = await run_eager_pass(db, character, facts, existing_inferences, ollama)
        return {"new_inferences": [inf.model_dump() for inf in new_inferences]}
    except InferenceParseError:
        return {
            "new_inferences": [],
            "warning": "Inference pass could not be parsed; try again.",
        }


@router.post("/{character_id}/inferences/revalidate")
async def revalidate_inferences_endpoint(
    character_id: int,
    body: _RevalidateBody,
    db: _DB,
    ollama: _Ollama,
) -> dict[str, object]:
    from memories.services.inference_service import cascade_on_fact_edit

    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    if not await _fact_exists(db, body.changed_fact_id):
        raise HTTPException(status_code=404, detail="Fact not found")

    stale = await cascade_on_fact_edit(db, character_id, body.changed_fact_id, ollama)
    return {"stale_inferences": [inf.model_dump() for inf in stale]}


@router.delete("/{character_id}/inferences/{inference_id}", status_code=204)
async def delete_inference_endpoint(
    character_id: int,
    inference_id: int,
    db: _DB,
) -> None:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    inferences = await get_inferences(db, character_id, status="all")
    if not any(inf.id == inference_id for inf in inferences):
        raise HTTPException(status_code=404, detail="Inference not found")

    try:
        await delete_inference(db, inference_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Inference not found") from exc


@router.patch("/{character_id}/inferences/{inference_id}", response_model=Inference)
async def patch_inference_endpoint(
    character_id: int,
    inference_id: int,
    body: _PatchBody,
    db: _DB,
) -> Inference:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    inferences = await get_inferences(db, character_id, status="all")
    if not any(inf.id == inference_id for inf in inferences):
        raise HTTPException(status_code=404, detail="Inference not found")

    try:
        return await update_inference_status(db, inference_id, body.status)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Inference not found") from exc


@router.post("/{character_id}/inferences/{inference_id}/promote", status_code=201)
async def promote_inference_endpoint(
    character_id: int,
    inference_id: int,
    body: _PromoteBody,
    db: _DB,
) -> dict[str, object]:
    character = await get_character(db, character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")

    inferences = await get_inferences(db, character_id, status="all")
    if not any(inf.id == inference_id for inf in inferences):
        raise HTTPException(status_code=404, detail="Inference not found")

    # Compute which inferences become stale (BFS) before opening the transaction.
    stale_inf_ids: list[int] = []
    visited: set[int] = {inference_id}
    queue: list[int] = [inference_id]
    while queue:
        current_id = queue.pop(0)
        for inf in inferences:
            if inf.id in visited:
                continue
            if current_id in inf.source_inference_ids:
                visited.add(inf.id)
                stale_inf_ids.append(inf.id)
                queue.append(inf.id)

    # All three writes (create fact, mark stale, delete inference) in one transaction.
    await db.execute("BEGIN IMMEDIATE")
    try:
        sql = (
            "INSERT INTO facts (character_id, key, value, category, mutability)"
            " VALUES (?, ?, ?, ?, ?)"
        )
        cursor = await db.execute(
            sql,
            (character_id, body.key, body.value, body.category, body.mutability),
        )
        fact_id = cursor.lastrowid
        assert fact_id is not None

        for inf_id in stale_inf_ids:
            await db.execute(
                "UPDATE inferences SET status = 'stale' WHERE id = ?",
                (inf_id,),
            )

        await db.execute("DELETE FROM inferences WHERE id = ?", (inference_id,))
        await db.commit()
    except aiosqlite.IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409,
            detail=f"A {body.category} Fact with key '{body.key}' already exists",
        ) from exc
    except Exception:
        await db.rollback()
        raise

    # Fetch the written fact and stale inference records to return in the response.
    fact_row = await (await db.execute("SELECT * FROM facts WHERE id = ?", (fact_id,))).fetchone()
    assert fact_row is not None
    fact = Fact.model_validate(_row(fact_row))

    stale: list[Inference] = []
    for inf_id in stale_inf_ids:
        inf_row = await (
            await db.execute("SELECT * FROM inferences WHERE id = ?", (inf_id,))
        ).fetchone()
        if inf_row is not None:
            stale.append(_parse_inference(inf_row))

    return {
        "fact": fact.model_dump(),
        "stale_inferences": [i.model_dump() for i in stale],
    }
