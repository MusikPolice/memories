"""Sessions API router."""

from __future__ import annotations

import logging
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from memories.database import (
    create_session,
    end_session,
    get_character,
    get_facts,
    get_inferences,
    get_messages,
    get_previous_session,
    get_session,
    update_session_closing_journal,
)
from memories.deps import get_db, get_ollama
from memories.models import Message, Session
from memories.services.experience_service import (
    ProposedExperience,
    SessionEndParseError,
    SessionEndResult,
    clear_active_experiences,
    run_session_end_evaluator,
)
from memories.services.ollama_client import OllamaClient, OllamaConnectionError, OllamaResponseError

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]

_log = logging.getLogger(__name__)


class _CreateBody(BaseModel):
    character_id: int


class _CreateSessionResponse(BaseModel):
    session: Session
    previous_journal: str | None


class _EndSessionResponse(BaseModel):
    session: Session
    closing_journal: str
    proposed_experiences: list[ProposedExperience]


@router.post("/", status_code=201, response_model=_CreateSessionResponse)
async def create_session_endpoint(body: _CreateBody, db: _DB) -> _CreateSessionResponse:
    character = await get_character(db, body.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail="Character not found")
    session = await create_session(db, character_id=body.character_id)
    prev = await get_previous_session(db, body.character_id, before_session_id=session.id)
    return _CreateSessionResponse(
        session=session,
        previous_journal=prev.closing_journal if prev else None,
    )


@router.post("/{session_id}/end", response_model=_EndSessionResponse)
async def end_session_endpoint(session_id: int, db: _DB, ollama: _Ollama) -> _EndSessionResponse:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at is not None:
        raise HTTPException(status_code=409, detail="Session has already ended")

    character = await get_character(db, session.character_id)
    assert character is not None
    facts = await get_facts(db, session.character_id)
    inferences = await get_inferences(db, session.character_id)
    messages = await get_messages(db, session_id)

    # Mark the session ended BEFORE the LLM call to prevent concurrent end requests.
    session = await end_session(db, session_id)

    try:
        result = await run_session_end_evaluator(character, facts, inferences, messages, ollama)
    except (
        SessionEndParseError,
        NotImplementedError,
        OllamaConnectionError,
        OllamaResponseError,
    ) as exc:
        _log.warning("session-end evaluator failed for session %d: %s", session_id, exc)
        result = SessionEndResult(closing_journal="", proposed_experiences=[])

    if result.closing_journal:
        session = await update_session_closing_journal(db, session_id, result.closing_journal)

    clear_active_experiences(session_id)

    return _EndSessionResponse(
        session=session,
        closing_journal=result.closing_journal,
        proposed_experiences=result.proposed_experiences,
    )


@router.get("/{session_id}/messages", response_model=list[Message])
async def get_session_messages_endpoint(session_id: int, db: _DB) -> list[Message]:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return await get_messages(db, session_id)
