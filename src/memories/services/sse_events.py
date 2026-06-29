"""Shared SSE event type for tool handlers that need to emit live notifications."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass
class SSEEvent:
    """A typed SSE event that tool handlers can emit via EventCallback."""

    event: str
    data: dict[str, object] = field(default_factory=dict)

    def to_sse(self) -> str:
        """Serialise to SSE wire format: two lines followed by a blank line."""
        return f"event: {self.event}\ndata: {json.dumps(self.data)}\n\n"


EventCallback = Callable[[SSEEvent], Awaitable[None]] | None
