"""Inference management API router (mounted under /api/characters)."""

from __future__ import annotations

from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memories.database import (
    delete_inference,
    get_character,
    get_facts,
    get_inferences,
)
from memories.deps import get_db, get_ollama
from memories.exceptions import NotFoundError
from memories.schema_loader import check_write_permitted, load_schema
from memories.services.inference_service import InferenceParseError
from memories.services.ollama_client import OllamaClient

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _RevalidateBody(BaseModel):
    changed_path: str


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

    facts_blob = await get_facts(db, character_id)
    existing_inferences = await get_inferences(db, character_id, status="all")

    try:
        new_inferences = await run_eager_pass(
            db, character, facts_blob, existing_inferences, ollama
        )
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

    try:
        check_write_permitted(body.changed_path, load_schema())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    stale = await cascade_on_fact_edit(db, character_id, body.changed_path, ollama)
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
