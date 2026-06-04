"""Async HTTP client for the Ollama REST API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx


class OllamaConnectionError(Exception):
    """Raised when a network-level connection to Ollama fails."""


class OllamaResponseError(Exception):
    """Raised when Ollama returns a non-2xx HTTP response."""


class OllamaClient:
    """Thin async wrapper around POST /api/chat.

    Uses ``stream: true`` and buffers the full response before returning so
    that the caller (the chat service) can insert the evaluator between
    'buffering complete' and 'deliver to client' in Phase 2.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
    ) -> tuple[str, dict[str, Any]]:
        """Send a chat request and return ``(assembled_content, final_chunk)``.

        *final_chunk* is the last NDJSON object from the stream, which
        contains ``prompt_eval_count`` and ``eval_count`` for token tracking.
        """
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"think": False},
        }
        try:
            async with (
                httpx.AsyncClient() as http,
                http.stream("POST", url, json=payload) as response,
            ):
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")
                parts: list[str] = []
                last_chunk: dict[str, Any] = {}
                async for line in response.aiter_lines():
                    if line.strip():
                        chunk: dict[str, Any] = json.loads(line)
                        msg = chunk.get("message", {})
                        if isinstance(msg, dict):
                            token: str = msg.get("content", "") or ""
                            if token:
                                parts.append(token)
                        last_chunk = chunk
                return "".join(parts), last_chunk
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(str(exc)) from exc
