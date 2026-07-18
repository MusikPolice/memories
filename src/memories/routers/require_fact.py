"""Accept/dismiss endpoint for require_fact blocking cards.

The client POSTs here when the user fills in (or dismisses) a require_fact card
surfaced by the Character LLM's require_fact tool handler (see
chat_service.run_contradiction_loop._handle_require_fact).
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from memories.services.tool_gate import resolve_gate

router = APIRouter()


class _RespondBody(BaseModel):
    value: str | None = None


@router.post("/{session_id}/turns/{turn_id}/require-fact/respond")
async def respond_to_require_fact(
    session_id: int,
    turn_id: int,
    body: _RespondBody = _RespondBody(),
) -> dict[str, str]:
    """Resolve (or dismiss) an active require_fact gate.

    Returns immediately without waiting for the SSE stream to resume.
    """
    try:
        resolve_gate(session_id, turn_id, body.value)
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"No pending require_fact for session {session_id} turn {turn_id}",
        ) from exc
    except asyncio.QueueFull as exc:
        raise HTTPException(status_code=409, detail="Already resolved") from exc

    return {"status": "ok"}
