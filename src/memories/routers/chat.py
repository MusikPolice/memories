"""Chat SSE endpoint."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from memories.database import get_session
from memories.deps import get_db, get_ollama
from memories.services.chat_service import run_turn
from memories.services.experience_service import get_active_experiences
from memories.services.ollama_client import OllamaClient

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]
_Ollama = Annotated[OllamaClient, Depends(get_ollama)]


class _SendBody(BaseModel):
    content: str
    think: bool = False


@router.post("/{session_id}/messages")
async def send_message(
    session_id: int, body: _SendBody, db: _DB, ollama: _Ollama
) -> StreamingResponse:
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.ended_at is not None:
        raise HTTPException(status_code=409, detail="Session has ended")

    async def _stream() -> AsyncGenerator[str, None]:
        _q: asyncio.Queue[str] = asyncio.Queue()

        async def _on_status(state: str) -> None:
            await _q.put(state)

        yield 'event: status\ndata: {"state": "extracting"}\n\n'

        _task = asyncio.ensure_future(
            run_turn(db, session_id, body.content, ollama, think=body.think, on_status=_on_status)
        )

        while not _task.done():
            try:
                state = _q.get_nowait()
                yield f'event: status\ndata: {{"state": "{state}"}}\n\n'
            except asyncio.QueueEmpty:
                await asyncio.sleep(0)

        while not _q.empty():
            yield f'event: status\ndata: {{"state": "{_q.get_nowait()}"}}\n\n'

        exc = _task.exception()
        if exc is not None:
            raise exc

        (
            content,
            thinking,
            turn_id,
            eval_result,
            experience_scores,
            extraction_result,
        ) = _task.result()

        # Emit contradiction loop events (Option B: after run_turn returns).
        # Each notification represents one contradiction found during the loop;
        # the regenerating+reviewing pair shows what happened on that retry.
        for notif in eval_result.contradiction_notifications:
            sc_data = json.dumps(
                {
                    "type": "contradiction",
                    "iteration": notif.iteration,
                    "description": notif.description,
                }
            )
            yield f"event: sidechannel\ndata: {sc_data}\n\n"
            yield 'event: status\ndata: {"state": "regenerating"}\n\n'
            yield 'event: status\ndata: {"state": "reviewing"}\n\n'

        if thinking:
            yield f"event: thinking\ndata: {json.dumps({'content': thinking})}\n\n"

        active_ids = [e.id for e in get_active_experiences(session_id)]
        scores_list = [{"id": eid, "score": score} for eid, score in experience_scores.items()]

        msg_data: dict[str, object] = {
            "role": "assistant",
            "content": content,
            "turn_id": turn_id,
            "active_experience_ids": active_ids,
            "experience_scores": scores_list,
        }
        if eval_result.verdict in ("implication", "new_inference_probabilistic"):
            msg_data["ungrounded"] = True
        if eval_result.max_retries_exceeded:
            msg_data["contradiction_exhausted"] = True

        yield f"event: message\ndata: {json.dumps(msg_data)}\n\n"

        # Emit sidechannel for non-contradiction violations / probabilistic inferences
        if eval_result.verdict in ("implication", "new_inference_probabilistic"):
            sc_payload: dict[str, object] = {
                "type": eval_result.verdict,
                "turn_id": turn_id,
                "violations": [v.model_dump() for v in eval_result.violations],
                "new_inferences": [i.model_dump() for i in eval_result.new_inferences],
            }
            yield f"event: sidechannel\ndata: {json.dumps(sc_payload)}\n\n"

        # Emit sidechannel for experience_update verdict
        if eval_result.verdict == "experience_update":
            exp_sc_payload: dict[str, object] = {
                "type": "experience_update",
                "turn_id": turn_id,
                "experience_updates": [u.model_dump() for u in eval_result.experience_updates],
            }
            yield f"event: sidechannel\ndata: {json.dumps(exp_sc_payload)}\n\n"

        # Emit extraction_applied sidechannel for Tier 1/2 auto-applied facts
        if extraction_result.new_facts or extraction_result.fact_updates:
            ext_payload: dict[str, object] = {
                "type": "extraction_applied",
                "turn_id": turn_id,
                "added": [
                    {
                        "fact_id": f.fact_id,
                        "key": f.key,
                        "value": f.value,
                        "category": f.category,
                        "mutability": f.mutability,
                        "source_quote": f.source_quote,
                    }
                    for f in extraction_result.new_facts
                    if f.fact_id is not None
                ],
                "updated": [
                    {
                        "fact_id": u.fact_id,
                        "key": u.key,
                        "old_value": u.old_value,
                        "new_value": u.new_value,
                        "source_quote": u.source_quote,
                    }
                    for u in extraction_result.fact_updates
                ],
            }
            yield f"event: sidechannel\ndata: {json.dumps(ext_payload)}\n\n"

        # Emit implicit_fact_proposed sidechannel for Tier 3/4 proposals
        if extraction_result.implicit_proposals:
            new_proposals = [
                p.model_dump()
                for p in extraction_result.implicit_proposals
                if p.existing_fact_id is None
            ]
            update_proposals = [
                p.model_dump()
                for p in extraction_result.implicit_proposals
                if p.existing_fact_id is not None
            ]
            impl_payload: dict[str, object] = {
                "type": "implicit_fact_proposed",
                "turn_id": turn_id,
                "new_proposals": new_proposals,
                "update_proposals": update_proposals,
            }
            yield f"event: sidechannel\ndata: {json.dumps(impl_payload)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")
