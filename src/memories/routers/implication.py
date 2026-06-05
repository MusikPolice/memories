"""Endpoints for accepting or ignoring implications and inferences surfaced by the evaluator."""

from __future__ import annotations

from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memories.database import (
    create_fact,
    create_inference,
    get_character,
    get_facts,
    get_messages,
    get_session,
    replace_message_content,
)
from memories.deps import get_db, get_ollama
from memories.exceptions import NotFoundError
from memories.services.chat_service import MAX_CONTRADICTION_RETRIES
from memories.services.evaluator import run_evaluator
from memories.services.ollama_client import OllamaClient
from memories.services.prompt_builder import build_system_prompt

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _AcceptImplicationBody(BaseModel):
    key: str
    value: str


class _AcceptInferenceBody(BaseModel):
    statement: str
    derivation: str
    source_fact_ids: list[int] = []
    inference_type: str = "probabilistic"


@router.post("/{session_id}/turns/{turn_id}/accept-implication")
async def accept_implication(
    session_id: int,
    turn_id: int,
    body: _AcceptImplicationBody,
    db: _DB,
    ollama: _Ollama,
) -> dict[str, object]:
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
    if assistant_msg.ungrounded_implications is None:
        raise HTTPException(status_code=422, detail="Turn has no ungrounded implications")

    # Create the fact
    try:
        await create_fact(db, character_id=session.character_id, key=body.key, value=body.value)
    except Exception:
        # Fact may already exist (e.g. duplicate key) — update it instead
        from memories.database import update_fact

        await update_fact(db, character_id=session.character_id, key=body.key, value=body.value)

    # Reload facts and rebuild context for regeneration
    character = await get_character(db, session.character_id)
    assert character is not None
    facts = await get_facts(db, session.character_id)
    system_prompt = build_system_prompt(character, facts)

    # Reconstruct conversation history up to (but not including) this turn's assistant message
    history_messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if msg.role == "assistant" and msg.turn_id == turn_id:
            break
        history_messages.append({"role": msg.role, "content": msg.content})

    model = character.current_model_name or character.modelfile_base

    # Regenerate with contradiction loop
    contradiction_hints: list[str] = []
    new_content = ""
    for attempt in range(MAX_CONTRADICTION_RETRIES + 1):
        regen_messages = list(history_messages)
        if contradiction_hints:
            note = (
                "[SYSTEM NOTE: Your previous response contained a contradiction. "
                + "; ".join(contradiction_hints)
                + ". Please revise.]"
            )
            regen_messages.append({"role": "user", "content": note})

        raw_content, _ = await ollama.chat(model, regen_messages)
        new_content = raw_content

        # Find the user message for this turn (for evaluator context)
        user_msg = next((m for m in messages if m.role == "user" and m.turn_id == turn_id), None)
        user_text = user_msg.content if user_msg else ""

        ev = await run_evaluator(
            character,
            facts,
            user_text,
            new_content,
            ollama,
            contradiction_hints=contradiction_hints or None,
        )
        if ev.verdict != "contradiction":
            break
        for v in ev.violations:
            if v.type == "contradiction":
                contradiction_hints.append(v.description)
        if attempt == MAX_CONTRADICTION_RETRIES:
            break

    # Replace the stored message and clear the ungrounded flag
    try:
        await replace_message_content(
            db, session_id=session_id, turn_id=turn_id, new_content=new_content
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Turn not found") from exc

    # Log a new decision for the regenerated response
    from memories.database import store_decision

    await store_decision(
        db,
        character_id=session.character_id,
        session_id=session_id,
        turn_id=turn_id,
        reasoning="Response regenerated after implication accepted.",
        verdict="pass",
    )

    return {"content": new_content, "turn_id": turn_id}


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
) -> dict[str, object]:
    """Store a user-approved probabilistic inference."""
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    inference = await create_inference(
        db,
        character_id=session.character_id,
        statement=body.statement,
        derivation=body.derivation,
        source_fact_ids=body.source_fact_ids,
        inference_type=body.inference_type,
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
