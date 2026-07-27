"""Accept, edit, reject, or dismiss endpoint for set_fact approval cards.

The client POSTs here when the user acts on a fact_update_mutable or
fact_update_immutable_unset blocking card surfaced by the Character Evaluator's
_handle_set_fact handler (see evaluator.run_evaluator._handle_set_fact).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from memories.services.tool_gate import resolve_gate

router = APIRouter()

_log = logging.getLogger(__name__)


class _ApprovalBody(BaseModel):
    action: Literal["accept", "edit", "reject", "dismiss"]
    value: str | None = None


@router.post("/{session_id}/turns/{turn_id}/set-fact/respond")
async def respond_to_set_fact(
    session_id: int,
    turn_id: int,
    body: _ApprovalBody,
) -> dict[str, str]:
    """Resolve an active set_fact approval gate.

    Returns immediately without waiting for the SSE stream to resume.
    """
    payload = json.dumps({"action": body.action, "value": body.value})
    try:
        resolve_gate(session_id, turn_id, payload)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No pending set_fact for session {session_id} turn {turn_id}",
        ) from exc
    except asyncio.QueueFull as exc:
        raise HTTPException(status_code=409, detail="Already resolved") from exc

    _log.info(
        "set-fact resolved session=%d turn=%d action=%s",
        session_id,
        turn_id,
        body.action,
    )
    return {"status": "ok"}
