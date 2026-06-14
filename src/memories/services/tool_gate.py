"""Per-turn asyncio.Queue gate for blocking tool calls that require user input.

Any tool handler that must suspend the tool-call loop and wait for an HTTP
response from the user (e.g. require_fact, mutable-fact approval) uses this
module.  The SSE generator creates a gate at turn start, the tool handler
awaits it, and the user-response endpoint resolves it.
"""

from __future__ import annotations

import asyncio

_pending: dict[tuple[int, int], asyncio.Queue[str | None]] = {}


def create_gate(session_id: int, turn_id: int) -> None:
    """Register a new one-shot gate keyed by (session_id, turn_id).

    Raises ValueError if a gate for this key already exists, making accidental
    double-registration visible rather than silently stranding the first awaiter.
    """
    key = (session_id, turn_id)
    if key in _pending:
        raise ValueError(f"Gate for ({session_id}, {turn_id}) already exists")
    _pending[key] = asyncio.Queue(maxsize=1)


async def await_gate(session_id: int, turn_id: int) -> str | None:
    """Suspend until resolve_gate puts a value into this gate's queue."""
    return await _pending[(session_id, turn_id)].get()


def resolve_gate(session_id: int, turn_id: int, value: str | None) -> None:
    """Deliver a user response to the awaiting tool handler.

    Raises asyncio.QueueFull if called a second time for the same gate,
    making double-resolve bugs visible rather than silently queuing a second value.
    """
    _pending[(session_id, turn_id)].put_nowait(value)


def cleanup_gate(session_id: int, turn_id: int) -> None:
    """Remove the gate from _pending; no-op if it does not exist."""
    _pending.pop((session_id, turn_id), None)
