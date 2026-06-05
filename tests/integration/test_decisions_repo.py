"""Integration tests for the decisions repository."""

from __future__ import annotations

import aiosqlite

from memories.database import (
    create_character,
    create_session,
    get_decisions,
    store_decision,
)


async def test_store_decision_returns_with_id(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    decision = await store_decision(
        db,
        character_id=char.id,
        session_id=session.id,
        turn_id=1,
        reasoning="Response looked clean.",
        verdict="pass",
    )
    assert decision.id > 0


async def test_store_decision_stores_verdict_and_reasoning(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    decision = await store_decision(
        db,
        character_id=char.id,
        session_id=session.id,
        turn_id=1,
        reasoning="Character mentioned a sibling.",
        verdict="implication",
    )
    assert decision.verdict == "implication"
    assert decision.reasoning == "Character mentioned a sibling."


async def test_store_decision_with_violations(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    violations = [
        {"type": "implication", "description": "implied a sibling", "suggested_fact": None}
    ]
    decision = await store_decision(
        db,
        character_id=char.id,
        session_id=session.id,
        turn_id=1,
        reasoning="See violations.",
        verdict="implication",
        violations=violations,
    )
    assert isinstance(decision.violations, list)
    assert decision.violations[0]["type"] == "implication"


async def test_store_decision_without_violations(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    decision = await store_decision(
        db,
        character_id=char.id,
        session_id=session.id,
        turn_id=1,
        reasoning="Clean.",
        verdict="pass",
        violations=None,
    )
    assert decision.violations is None


async def test_get_decisions_returns_all_for_session(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    await store_decision(
        db, character_id=char.id, session_id=session.id, turn_id=1, reasoning="t1", verdict="pass"
    )
    await store_decision(
        db, character_id=char.id, session_id=session.id, turn_id=2, reasoning="t2", verdict="pass"
    )
    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 2


async def test_get_decisions_ordered_by_turn_id_desc(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session = await create_session(db, character_id=char.id)
    await store_decision(
        db,
        character_id=char.id,
        session_id=session.id,
        turn_id=1,
        reasoning="first",
        verdict="pass",
    )
    await store_decision(
        db,
        character_id=char.id,
        session_id=session.id,
        turn_id=2,
        reasoning="second",
        verdict="implication",
    )
    decisions = await get_decisions(db, session.id)
    assert decisions[0].turn_id == 2
    assert decisions[1].turn_id == 1


async def test_get_decisions_isolated_per_session(db: aiosqlite.Connection) -> None:
    char = await create_character(db, name="Alice", modelfile_base="qwen3:7b")
    session_a = await create_session(db, character_id=char.id)
    session_b = await create_session(db, character_id=char.id)
    await store_decision(
        db, character_id=char.id, session_id=session_a.id, turn_id=1, reasoning="A", verdict="pass"
    )
    decisions_b = await get_decisions(db, session_b.id)
    assert decisions_b == []
