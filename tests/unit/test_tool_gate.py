"""Unit tests for memories.services.tool_gate."""

from __future__ import annotations

import asyncio
from collections.abc import Generator

import pytest

import memories.services.tool_gate as tool_gate_module
from memories.services.tool_gate import (
    await_gate,
    cleanup_gate,
    create_gate,
    resolve_gate,
)


@pytest.fixture(autouse=True)
def _clear_gates() -> Generator[None, None, None]:
    tool_gate_module._pending.clear()
    yield
    tool_gate_module._pending.clear()


async def test_create_and_resolve_gate() -> None:
    create_gate(session_id=1, turn_id=42)
    resolve_gate(session_id=1, turn_id=42, value="Sarah")
    result = await await_gate(session_id=1, turn_id=42)
    assert result == "Sarah"


async def test_none_value_resolves_gate() -> None:
    create_gate(session_id=1, turn_id=1)
    resolve_gate(session_id=1, turn_id=1, value=None)
    result = await await_gate(session_id=1, turn_id=1)
    assert result is None


async def test_multiple_concurrent_gates() -> None:
    create_gate(1, 1)
    create_gate(1, 2)
    create_gate(2, 1)
    resolve_gate(1, 1, value="alpha")
    resolve_gate(1, 2, value="beta")
    resolve_gate(2, 1, value="gamma")
    assert await await_gate(1, 1) == "alpha"
    assert await await_gate(1, 2) == "beta"
    assert await await_gate(2, 1) == "gamma"


async def test_cleanup_removes_gate() -> None:
    create_gate(1, 42)
    cleanup_gate(1, 42)
    with pytest.raises(KeyError):
        await await_gate(1, 42)


async def test_resolve_before_await() -> None:
    create_gate(1, 99)
    resolve_gate(1, 99, value="preloaded")
    result = await await_gate(1, 99)
    assert result == "preloaded"


async def test_double_resolve_raises() -> None:
    create_gate(1, 10)
    resolve_gate(1, 10, value="first")
    with pytest.raises(asyncio.QueueFull):
        resolve_gate(1, 10, value="second")


async def test_cleanup_idempotent() -> None:
    cleanup_gate(99, 99)  # gate does not exist — should not raise
