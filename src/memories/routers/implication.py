"""Endpoints for accepting or ignoring implications and inferences surfaced by the evaluator."""

from __future__ import annotations

from typing import Annotated, Any

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
    store_decision,
    update_fact,
)
from memories.deps import get_db, get_ollama
from memories.exceptions import NotFoundError
from memories.services.chat_service import MAX_CONTRADICTION_RETRIES, run_contradiction_loop
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
    if assistant_msg.ungrounded_implications is None:
        raise HTTPException(status_code=422, detail="Turn has no ungrounded implications")

    # Create the fact; update if the key already exists
    try:
        await create_fact(db, character_id=session.character_id, key=body.key, value=body.value)
    except aiosqlite.IntegrityError:
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
