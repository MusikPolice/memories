# PoC-only SSE endpoint for require_fact suspension testing (Step 0b).
# This router is scaffolding that will be removed in Step 6 when require_fact
# is integrated into the real run_turn() orchestration.

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Annotated

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from memories.database import get_session
from memories.deps import get_db
from memories.services.chat_service import SSEEvent
from memories.services.tool_gate import (
    await_gate,
    cleanup_gate,
    create_gate,
)

router = APIRouter()

_DB = Annotated[aiosqlite.Connection, Depends(get_db)]


@router.post("/{session_id}/test-require-fact-poc")
async def test_require_fact_poc(
    session_id: int,
    turn_id: int,
    db: _DB,
    delay_ms: int = 0,
    fail: bool = False,
) -> StreamingResponse:
    """PoC SSE endpoint that simulates a require_fact suspension cycle.

    Validates the session, creates a gate, emits a sidechannel event, suspends
    on await_gate(), then emits the resolved value and done.  The ``fail`` query
    param injects an artificial error after the sidechannel to test gate cleanup.
    """
    session = await get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        create_gate(session_id, turn_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail=f"Gate for session {session_id} turn {turn_id} already exists",
        ) from exc

    async def _stream() -> AsyncGenerator[str, None]:
        _q: asyncio.Queue[SSEEvent] = asyncio.Queue()

        async def _on_event(event: SSEEvent) -> None:
            await _q.put(event)

        async def _work() -> None:
            if delay_ms:
                await asyncio.sleep(delay_ms / 1000)

            await _on_event(SSEEvent(event="status", data={"state": "generating"}))
            await _on_event(
                SSEEvent(
                    event="sidechannel",
                    data={
                        "type": "require_fact",
                        "turn_id": turn_id,
                        "path": "Character.Identity.Name",
                        "reason": "Integration test",
                        "suggested_value": "Alice",
                    },
                )
            )

            if fail:
                raise RuntimeError("Injected test failure")

            value = await await_gate(session_id, turn_id)
            content = value if value is not None else "(dismissed)"
            await _on_event(
                SSEEvent(event="message", data={"role": "assistant", "content": content})
            )
            await _on_event(SSEEvent(event="done", data={}))

        task = asyncio.ensure_future(_work())

        try:
            while not task.done():
                try:
                    ev = await asyncio.wait_for(_q.get(), timeout=0.05)
                    yield ev.to_sse()
                except TimeoutError:
                    yield ": ping\n\n"

            while not _q.empty():
                yield _q.get_nowait().to_sse()

            exc = task.exception()
            if exc is not None:
                raise exc

        finally:
            cleanup_gate(session_id, turn_id)

    return StreamingResponse(_stream(), media_type="text/event-stream")
