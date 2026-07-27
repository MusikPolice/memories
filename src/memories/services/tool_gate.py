"""Per-turn asyncio.Queue gate for blocking tool calls that require user input.

Any tool handler that must suspend the tool-call loop and wait for an HTTP
response from the user (e.g. require_fact, mutable-fact approval) uses this
module.  The SSE generator creates a gate at turn start, the tool handler
awaits it, and the user-response endpoint resolves it.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict

_log = logging.getLogger(__name__)

# Upper bound on how many resolved-gate keys we retain to detect late double-resolves.
# Without a bound this set would grow by one entry per resolved gate for the lifetime of
# the process (a slow leak on a long-running server).  A turn's double-resolve check only
# needs its key to survive from the first resolve until any immediate retry, so evicting
# the oldest keys once the cap is exceeded never interferes with that.
_MAX_RESOLVED_KEYS = 1024

_pending: dict[tuple[int, int], asyncio.Queue[str | None]] = {}
_resolved: OrderedDict[tuple[int, int], None] = OrderedDict()


def create_gate(session_id: int, turn_id: int) -> None:
    """Register a new one-shot gate keyed by (session_id, turn_id).

    Raises ValueError if a gate for this key already exists, making accidental
    double-registration visible rather than silently stranding the first awaiter.
    """
    key = (session_id, turn_id)
    if key in _pending:
        raise ValueError(f"Gate for ({session_id}, {turn_id}) already exists")
    _pending[key] = asyncio.Queue(maxsize=1)
    _resolved.pop(key, None)
    _log.debug("gate created session=%d turn=%d", session_id, turn_id)


async def await_gate(session_id: int, turn_id: int) -> str | None:
    """Suspend until resolve_gate puts a value into this gate's queue."""
    key = (session_id, turn_id)
    queue = _pending[key]
    if queue.empty():
        _resolved.pop(key, None)
    return await queue.get()


def resolve_gate(session_id: int, turn_id: int, value: str | None) -> None:
    """Deliver a user response to the awaiting tool handler.

    Raises asyncio.QueueFull if called a second time for the same gate (including
    after cleanup_gate has removed the gate), preventing late double-resolves.
    Raises KeyError if no gate was ever created for this key.
    """
    key = (session_id, turn_id)
    if key in _resolved:
        raise asyncio.QueueFull()
    _pending[key].put_nowait(value)
    _resolved[key] = None
    _log.debug("gate resolved session=%d turn=%d", session_id, turn_id)
    # Bound memory: drop the oldest resolved keys once we exceed the cap.
    while len(_resolved) > _MAX_RESOLVED_KEYS:
        _resolved.popitem(last=False)


def cleanup_gate(session_id: int, turn_id: int) -> None:
    """Remove the gate from _pending; no-op if it does not exist.

    _resolved is intentionally not cleared here: entries persist (bounded by
    _MAX_RESOLVED_KEYS) to catch any double-resolve attempt that races with or
    arrives after cleanup.
    """
    _pending.pop((session_id, turn_id), None)
    _log.debug("gate cleanup session=%d turn=%d", session_id, turn_id)
