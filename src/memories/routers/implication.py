"""Endpoints for accepting or ignoring implications and inferences surfaced by the evaluator."""

from __future__ import annotations

from typing import Annotated, Any, Literal

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from memories.database import (
    create_fact,
    create_inference,
    get_character,
    get_fact,
    get_fact_by_category_key,
    get_facts,
    get_inferences,
    get_messages,
    get_session,
    replace_message_content,
    store_decision,
    update_fact,
)
from memories.deps import get_db, get_ollama
from memories.exceptions import NotFoundError
from memories.models import Fact, Session
from memories.services.chat_service import MAX_CONTRADICTION_RETRIES, run_contradiction_loop
from memories.services.inference_service import (
    MAX_INFERENCE_DEPTH,
    cascade_on_fact_edit,
    compute_depth,
)
from memories.services.ollama_client import OllamaClient
from memories.services.prompt_builder import build_system_prompt

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _AcceptImplicationBody(BaseModel):
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    regenerate: bool = True


class _AcceptInferenceBody(BaseModel):
    statement: str
    derivation: str
    source_fact_ids: list[int] = []
    source_inference_ids: list[int] = []
    inference_type: str = "probabilistic"


@router.post("/{session_id}/turns/{turn_id}/accept-implication")
async def accept_implication(
    session_id: int,
    turn_id: int,
    body: _AcceptImplicationBody,
    db: _DB,
    ollama: _Ollama,
) -> dict[str, Any]:
    """Create a Fact from an implied value and regenerate the assistant response."""
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    # Find the ungrounded assistant message
    messages = await get_messages(db, session_id)
    assistant_msg = next(
        (m for m in messages if m.role == "assistant" and m.turn_id == turn_id),
        None,
    )
    if assistant_msg is None:
        raise HTTPException(status_code=404, detail="Turn not found")

    # Create the fact; update if the (category, key) tuple already exists.
    try:
        await create_fact(
            db,
            character_id=session.character_id,
            key=body.key,
            value=body.value,
            category=body.category,
        )
    except aiosqlite.IntegrityError:
        existing = await get_fact_by_category_key(
            db,
            character_id=session.character_id,
            category=body.category,
            key=body.key,
        )
        if existing is None:
            raise
        # Update value only — preserve existing category and mutability.
        await update_fact(db, fact_id=existing.id, value=body.value)

    # When the user accepted the value exactly as suggested, no regeneration is needed —
    # the character's existing response already reflects the fact correctly.
    if not body.regenerate:
        try:
            await replace_message_content(
                db, session_id=session_id, turn_id=turn_id, new_content=assistant_msg.content
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail="Turn not found") from exc
        return {"content": assistant_msg.content, "turn_id": turn_id}

    # Reload facts and rebuild context for regeneration
    character = await get_character(db, session.character_id)
    assert character is not None
    facts = await get_facts(db, session.character_id)
    inferences = await get_inferences(db, session.character_id)
    system_prompt = build_system_prompt(character, facts, inferences)

    # Reconstruct conversation history up to (but not including) this turn's assistant message
    history_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.role == "assistant" and msg.turn_id == turn_id:
            break
        history_messages.append({"role": msg.role, "content": msg.content})

    model = character.current_model_name or character.modelfile_base

    # Find the user message for this turn (evaluator context)
    user_msg = next((m for m in messages if m.role == "user" and m.turn_id == turn_id), None)
    user_text = user_msg.content if user_msg else ""

    new_content, _, ev = await run_contradiction_loop(
        model,
        history_messages,
        character,
        facts,
        user_text,
        ollama,
        max_retries=MAX_CONTRADICTION_RETRIES,
        inferences=inferences,
    )

    # Replace the stored message and clear the ungrounded flag
    try:
        await replace_message_content(
            db, session_id=session_id, turn_id=turn_id, new_content=new_content
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Turn not found") from exc

    # Log a new decision using the actual verdict from the regenerated response
    violations_for_log = [v.model_dump() for v in ev.violations] if ev.violations else None
    await store_decision(
        db,
        character_id=session.character_id,
        session_id=session_id,
        turn_id=turn_id,
        reasoning=ev.decision_log or "Response regenerated after implication accepted.",
        verdict=ev.verdict,
        violations=violations_for_log,
    )

    response: dict[str, Any] = {"content": new_content, "turn_id": turn_id}
    # If the regenerated response is itself ungrounded, surface that to the caller
    if ev.verdict in ("implication", "new_inference_probabilistic"):
        response["ungrounded"] = True
        response["violations"] = [v.model_dump() for v in ev.violations]
        response["new_inferences"] = [i.model_dump() for i in ev.new_inferences]
    return response


@router.post("/{session_id}/turns/{turn_id}/ignore-implication", status_code=204)
async def ignore_implication(
    session_id: int,
    turn_id: int,
    db: _DB,
) -> None:
    """Dismiss an implication notification without creating a Fact."""
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    # No DB change needed — the message stays tagged; the client just dismisses the notification


@router.post("/{session_id}/turns/{turn_id}/accept-inference", status_code=201)
async def accept_inference(
    session_id: int,
    turn_id: int,
    body: _AcceptInferenceBody,
    db: _DB,
) -> dict[str, Any]:
    """Store a user-approved probabilistic inference."""
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    existing = await get_inferences(db, session.character_id)
    depth = compute_depth(body.source_inference_ids, existing)
    if depth > MAX_INFERENCE_DEPTH:
        raise HTTPException(
            status_code=422,
            detail=f"Inference depth {depth} exceeds cap {MAX_INFERENCE_DEPTH}",
        )

    inference = await create_inference(
        db,
        character_id=session.character_id,
        statement=body.statement,
        derivation=body.derivation,
        source_fact_ids=body.source_fact_ids,
        source_inference_ids=body.source_inference_ids,
        inference_type=body.inference_type,
        depth=depth,
    )
    return inference.model_dump()


@router.post("/{session_id}/turns/{turn_id}/ignore-inference", status_code=204)
async def ignore_inference(
    session_id: int,
    turn_id: int,
    db: _DB,
) -> None:
    """Dismiss a probabilistic inference notification without storing it."""
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")


# ---------------------------------------------------------------------------
# Phase 6 endpoints — extraction resolution
# ---------------------------------------------------------------------------


class _UndoFactBody(BaseModel):
    fact_id: int
    restore_value: str


class _AcceptImplicitBody(BaseModel):
    key: str
    value: str
    category: Literal["user", "character", "setting"]
    mutability: Literal["immutable", "low", "high"]
    existing_fact_id: int | None = None


class _IgnoreImplicitBody(BaseModel):
    key: str


async def _get_active_session(db: aiosqlite.Connection, session_id: int) -> Session:
    """Return session or raise 404/409."""
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    if session.ended_at is not None:
        raise HTTPException(status_code=409, detail="Session has ended")
    return session


async def _find_owned_fact(db: aiosqlite.Connection, character_id: int, fact_id: int) -> Fact:
    """Return fact or raise 404 if not found or not owned by character."""
    fact = await get_fact(db, character_id, fact_id)
    if fact is None:
        raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")
    return fact


@router.post("/{session_id}/turns/{turn_id}/undo-user-fact")
async def undo_user_fact(
    session_id: int,
    turn_id: int,
    body: _UndoFactBody,
    db: _DB,
    ollama: _Ollama,
) -> dict[str, Any]:
    """Restore a Tier 2 auto-applied fact to a previous value."""
    session = await _get_active_session(db, session_id)
    fact = await _find_owned_fact(db, session.character_id, body.fact_id)

    await update_fact(db, fact_id=fact.id, value=body.restore_value)

    stale = await cascade_on_fact_edit(db, session.character_id, fact.id, ollama)

    # Re-fetch updated fact for the response
    updated_facts = await get_facts(db, session.character_id)
    updated_fact = next(f for f in updated_facts if f.id == fact.id)

    return {
        "fact": updated_fact.model_dump(mode="json"),
        "stale_inferences": [i.model_dump(mode="json") for i in stale],
    }


@router.post("/{session_id}/turns/{turn_id}/accept-implicit-fact")
async def accept_implicit_fact(
    session_id: int,
    turn_id: int,
    body: _AcceptImplicitBody,
    db: _DB,
    ollama: _Ollama,
) -> JSONResponse:
    """Accept a Tier 3 or Tier 4 implicit fact proposal."""
    session = await _get_active_session(db, session_id)

    if body.existing_fact_id is not None:
        # Tier 4: update existing fact
        fact = await _find_owned_fact(db, session.character_id, body.existing_fact_id)
        await update_fact(db, fact_id=fact.id, value=body.value)

        stale = await cascade_on_fact_edit(db, session.character_id, fact.id, ollama)

        updated_facts = await get_facts(db, session.character_id)
        updated_fact = next(f for f in updated_facts if f.id == fact.id)

        return JSONResponse(
            status_code=200,
            content={
                "fact": updated_fact.model_dump(mode="json"),
                "stale_inferences": [i.model_dump(mode="json") for i in stale],
            },
        )
    else:
        # Tier 3: create new fact; fall back to update if the key already exists.
        try:
            new_fact = await create_fact(
                db,
                character_id=session.character_id,
                key=body.key,
                value=body.value,
                category=body.category,
                mutability=body.mutability,
            )
        except aiosqlite.IntegrityError:
            existing = await get_fact_by_category_key(
                db,
                character_id=session.character_id,
                category=body.category,
                key=body.key,
            )
            if existing is None:
                raise
            await update_fact(db, fact_id=existing.id, value=body.value)
            stale = await cascade_on_fact_edit(db, session.character_id, existing.id, ollama)
            updated_facts = await get_facts(db, session.character_id)
            updated_fact = next(f for f in updated_facts if f.id == existing.id)
            return JSONResponse(
                status_code=200,
                content={
                    "fact": updated_fact.model_dump(mode="json"),
                    "stale_inferences": [i.model_dump(mode="json") for i in stale],
                },
            )

        return JSONResponse(
            status_code=201,
            content={
                "fact": new_fact.model_dump(mode="json"),
                "stale_inferences": [],
            },
        )


@router.post("/{session_id}/turns/{turn_id}/ignore-implicit-fact", status_code=204)
async def ignore_implicit_fact(
    session_id: int,
    turn_id: int,
    body: _IgnoreImplicitBody,
    db: _DB,
) -> None:
    """Dismiss a Tier 3/4 implicit fact proposal without writing to DB."""
    await _get_active_session(db, session_id)
