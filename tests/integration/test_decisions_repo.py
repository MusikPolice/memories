"""Integration tests for the decisions repository."""

from __future__ import annotations

import aiosqlite

from memories.database import (
    create_session,
    get_decisions,
    store_decision,
)
from memories.models import Character, Session


async def test_store_decision_with_tool_call_data(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    decision = await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=1,
        pass_name="character_evaluator",
        tool_name="report_pass",
        tool_args={},
        user_input=None,
    )
    assert decision.id > 0
    assert decision.pass_name == "character_evaluator"
    assert decision.tool_name == "report_pass"
    assert decision.tool_args == {}
    assert decision.user_input is None
    assert not hasattr(decision, "reasoning")
    assert not hasattr(decision, "verdict")


async def test_store_decision_tool_args_round_trips_as_dict(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=1,
        pass_name="character_evaluator",
        tool_name="set_fact",
        tool_args={"path": "Character.Identity.Name", "value": "Sarah"},
        user_input=None,
    )
    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 1
    assert isinstance(decisions[0].tool_args, dict)
    assert decisions[0].tool_args["path"] == "Character.Identity.Name"
    assert decisions[0].tool_args["value"] == "Sarah"


async def test_store_decision_with_user_input(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=1,
        pass_name="character_evaluator",
        tool_name="set_fact",
        tool_args={"path": "Character.Identity.Name", "value": "Sarah"},
        user_input={"action": "accept", "value": "Sarah"},
    )
    decisions = await get_decisions(db, session.id)
    assert isinstance(decisions[0].user_input, dict)
    assert decisions[0].user_input["action"] == "accept"
    assert decisions[0].user_input["value"] == "Sarah"


async def test_store_decision_user_input_null(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=1,
        pass_name="character_evaluator",
        tool_name="report_pass",
        tool_args={},
        user_input=None,
    )
    decisions = await get_decisions(db, session.id)
    assert decisions[0].user_input is None


async def test_get_decisions_returns_all_for_session(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=1,
        pass_name="character_evaluator",
        tool_name="report_pass",
        tool_args={},
        user_input=None,
    )
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=2,
        pass_name="character_evaluator",
        tool_name="report_pass",
        tool_args={},
        user_input=None,
    )
    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 2


async def test_get_decisions_returns_in_reverse_chronological_order(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=1,
        pass_name="character_evaluator",
        tool_name="report_pass",
        tool_args={},
        user_input=None,
    )
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=2,
        pass_name="character_evaluator",
        tool_name="set_fact",
        tool_args={"path": "Character.State-Of-Mind.Mood", "value": "Calm"},
        user_input=None,
    )
    decisions = await get_decisions(db, session.id)
    assert decisions[0].turn_id == 2
    assert decisions[1].turn_id == 1


async def test_get_decisions_isolated_per_session(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    session_b = await create_session(db, character_id=character.id)
    await store_decision(
        db,
        character_id=character.id,
        session_id=session.id,
        turn_id=1,
        pass_name="character_evaluator",
        tool_name="report_pass",
        tool_args={},
        user_input=None,
    )
    decisions_b = await get_decisions(db, session_b.id)
    assert decisions_b == []
