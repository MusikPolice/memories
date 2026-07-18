"""Async HTTP client for the Ollama REST API."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Matches chat-template control tokens that some models emit past their
# natural stop point (e.g. qwen3 emitting <|endoftext|> then repeating).
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>|</s>")

MAX_TOOL_CALL_ROUNDS = int(os.getenv("MAX_TOOL_CALL_ROUNDS", "10"))

# Async callable invoked by chat_with_tools for each tool call the model makes.
# Receives the arguments dict from the tool_call; must return a result string.
# May raise any exception — str(exc) is sent back to the model as the error.
# Must be async so that handlers can suspend the loop (e.g. to await a turn gate).
ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


@dataclass
class ToolCallResult:
    content: str  # final plain-text response from the model
    history: list[dict[str, Any]]  # full message history including all tool turns
    rounds: int  # number of HTTP round-trips made
    cap_reached: bool  # True if the loop was cut short by max_rounds
    # Set when a round contains a call to a tool named in chat_with_tools()'s
    # terminal_tools argument.  {"name": str, "arguments": dict, "result": str}
    # for the first such call; None if terminal_tools was not passed or no
    # terminal tool was called before the loop ended.
    terminal_call: dict[str, Any] | None = None
    # Accumulated thinking text across every round, joined with no separator —
    # mirrors chat()'s "".join(thinking_parts) behaviour. "" if think=False.
    thinking: str = ""


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

    async def chat_with_tools(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_handlers: dict[str, ToolHandler],
        max_rounds: int = MAX_TOOL_CALL_ROUNDS,
        terminal_tools: frozenset[str] | None = None,
        think: bool = False,
    ) -> ToolCallResult:
        """Drive a tool-calling loop against POST /api/chat with ``stream: false``.

        Re-invokes the model after each batch of tool calls until it returns
        plain content or ``max_rounds`` is exhausted.  Returns a
        ``ToolCallResult`` whose ``cap_reached`` flag signals whether the loop
        was cut short.  If ``terminal_tools`` is given, the loop returns as soon
        as a round contains a call to one of those tool names —
        ``ToolCallResult.terminal_call`` carries that call's name, arguments,
        and result.  Pass ``None`` (the default) to preserve the original
        behaviour of looping until the model returns plain content or
        ``max_rounds`` is exhausted.  ``OllamaConnectionError`` and
        ``OllamaResponseError`` propagate unchanged.
        ``think`` is forwarded to Ollama on every round exactly as it is for
        ``chat()``; ``ToolCallResult.thinking`` accumulates every round's thinking
        text, stripped from history before the next round is sent, for the same
        reason ``chat()`` strips it from the conversation it returns.
        """
        url = f"{self.base_url}/api/chat"
        history: list[dict[str, Any]] = list(messages)
        content = ""
        rounds = 0
        thinking_parts: list[str] = []

        log.debug(
            "chat_with_tools start model=%s max_rounds=%d tools=%s",
            model,
            max_rounds,
            [t["function"]["name"] for t in tools if "function" in t],
        )

        while rounds < max_rounds:
            payload: dict[str, Any] = {
                "model": model,
                "messages": history,
                "tools": tools,
                "stream": False,
                "think": think,
            }
            try:
                response = await self._http.post(url, json=payload)
            except httpx.ConnectError as exc:
                raise OllamaConnectionError(str(exc)) from exc

            if response.status_code != 200:
                raise OllamaResponseError(f"Ollama returned HTTP {response.status_code}")

            data: dict[str, Any] = response.json()
            rounds += 1
            msg: dict[str, Any] = data["message"]
            thought: str = msg.get("thinking", "") or ""
            if thought:
                thinking_parts.append(thought)
            # Strip thinking before appending to history. The qwen3 family (and
            # others) loop indefinitely when their own thinking tokens are fed
            # back as conversation context — they were never meant to be.
            history.append({k: v for k, v in msg.items() if k != "thinking"})

            tool_calls: list[dict[str, Any]] = msg.get("tool_calls") or []
            if not tool_calls:
                content = msg.get("content", "") or ""
                log.debug("chat_with_tools done rounds=%d content=%r", rounds, content[:120])
                return ToolCallResult(
                    content=content,
                    history=history,
                    rounds=rounds,
                    cap_reached=False,
                    thinking="".join(thinking_parts),
                )

            log.debug(
                "chat_with_tools round=%d tool_calls=%s",
                rounds,
                [
                    {
                        "id": c.get("id"),
                        "name": c["function"]["name"],
                        "args": c["function"]["arguments"],
                    }
                    for c in tool_calls
                ],
            )

            terminal_call: dict[str, Any] | None = None
            for call in tool_calls:
                fn: dict[str, Any] = call["function"]
                name: str = fn["name"]
                args: dict[str, Any] = fn["arguments"]
                call_id: str | None = call.get("id")
                handler = tool_handlers.get(name)
                if handler is None:
                    result = f"Unknown tool: {name}"
                else:
                    try:
                        result = await handler(args)
                    except Exception as exc:
                        result = f"Error: {exc}"
                log.debug(
                    "chat_with_tools tool_result id=%s name=%s result=%r", call_id, name, result
                )
                tool_msg: dict[str, Any] = {"role": "tool", "content": result}
                if call_id is not None:
                    # Required for models that use IDs (e.g. qwen3 multi-call
                    # responses): without tool_call_id the model can't correlate
                    # results to calls and loops indefinitely.
                    tool_msg["tool_call_id"] = call_id
                history.append(tool_msg)

                if terminal_tools is not None and terminal_call is None and name in terminal_tools:
                    terminal_call = {"name": name, "arguments": args, "result": result}

            if terminal_call is not None:
                log.debug(
                    "chat_with_tools terminal_call=%s rounds=%d", terminal_call["name"], rounds
                )
                return ToolCallResult(
                    content=content,
                    history=history,
                    rounds=rounds,
                    cap_reached=False,
                    terminal_call=terminal_call,
                    thinking="".join(thinking_parts),
                )

        log.debug("chat_with_tools cap_reached rounds=%d", rounds)
        return ToolCallResult(
            content=content,
            history=history,
            rounds=rounds,
            cap_reached=True,
            thinking="".join(thinking_parts),
        )

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
