"""Chat turn orchestration.

Stub — raises NotImplementedError. Tests will fail until implemented.
"""

from __future__ import annotations

import aiosqlite

from memories.services.ollama_client import OllamaClient


async def run_turn(
    db: aiosqlite.Connection,
    session_id: int,
    user_content: str,
    ollama: OllamaClient,
) -> str:
    """Execute one conversation turn and return the assistant response.

    Full sequence (Phase 1):
    1. Load session + character + facts from DB.
    2. Build system prompt.
    3. Load conversation history.
    4. Store user message.
    5. Call Ollama; buffer response.
    6. Store assistant message.
    7. Return response string.
    """
    raise NotImplementedError
