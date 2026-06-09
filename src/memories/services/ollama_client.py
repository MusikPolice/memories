"""Async HTTP client for the Ollama REST API."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx

# Matches chat-template control tokens that some models emit past their
# natural stop point (e.g. qwen3 emitting <|endoftext|> then repeating).
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>|</s>")


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
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        think: bool = False,
        format: dict[str, Any] | str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Send a chat request and return ``(assembled_content, final_chunk)``.

        *final_chunk* is the last NDJSON object from the stream, which
        contains ``prompt_eval_count`` and ``eval_count`` for token tracking.
        ``think`` must be at the top level of the request body — NOT inside
        ``options`` — which Ollama silently ignores.
        ``format`` is passed verbatim to Ollama; use ``"json"`` or a JSON
        schema dict to constrain the output format (e.g. for the evaluator).
        """
        url = f"{self.base_url}/api/chat"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
        }
        if format is not None:
            payload["format"] = format
        try:
            async with self._http.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")
                parts: list[str] = []
                thinking_parts: list[str] = []
                last_chunk: dict[str, Any] = {}
                async for line in response.aiter_lines():
                    if line.strip():
                        chunk: dict[str, Any] = json.loads(line)
                        msg = chunk.get("message", {})
                        if isinstance(msg, dict):
                            thought: str = msg.get("thinking", "") or ""
                            if thought:
                                thinking_parts.append(thought)
                            token: str = msg.get("content", "") or ""
                            if token:
                                parts.append(token)
                        last_chunk = chunk
                raw = "".join(parts)
                m = _SPECIAL_TOKEN_RE.search(raw)
                last_chunk["thinking"] = "".join(thinking_parts)
                return (raw[: m.start()].rstrip() if m else raw), last_chunk
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(str(exc)) from exc

    async def embed(self, model: str, text: str) -> list[float]:
        """Embed *text* using *model* via POST /api/embed.

        Returns the first embedding from the response.
        Raises OllamaConnectionError or OllamaResponseError on failure.
        """
        url = f"{self.base_url}/api/embed"
        payload = {"model": model, "input": text}
        try:
            response = await self._http.post(url, json=payload)
            if response.status_code != 200:
                raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")
            data = response.json()
            embeddings = data.get("embeddings", [])
            if not embeddings:
                raise OllamaResponseError("Ollama embed response contained no embeddings")
            vec: list[float] = embeddings[0]
            return vec
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(str(exc)) from exc

    async def warmup(self, model: str) -> None:
        """Load *model* into Ollama's memory without generating any output.

        Sends POST /api/generate with no prompt, which is Ollama's documented
        way to preload a model.  Call this at startup so the first real chat
        request doesn't pay the model-loading latency cost.
        """
        url = f"{self.base_url}/api/generate"
        try:
            response = await self._http.post(url, json={"model": model, "keep_alive": "10m"})
            if response.status_code != 200:
                raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(str(exc)) from exc

    async def warmup_embed(self, model: str) -> None:
        """Load an embedding model into Ollama's memory.

        Uses POST /api/embed (not /api/generate) because embedding models
        do not support the generate endpoint.
        """
        url = f"{self.base_url}/api/embed"
        try:
            response = await self._http.post(
                url, json={"model": model, "input": "", "keep_alive": "10m"}
            )
            if response.status_code != 200:
                raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(str(exc)) from exc
