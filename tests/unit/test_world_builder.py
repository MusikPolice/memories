"""Unit tests for memories.services.world_builder.

Covers build_world_builder_prompt (prompt content) and run_world_builder
(tool-call mechanics, decision logging, sidechannel notification, and
cap behaviour) using the ollama fixture and the make_*_response helpers
from tests/unit/conftest.py.
"""

from __future__ import annotations

import logging
from typing import Any

import aiosqlite
import httpx
import pytest
import respx

from memories.database import get_decisions
from memories.models import Character, Inference, Session
from memories.services.ollama_client import OllamaClient
from memories.services.sse_events import SSEEvent
from memories.services.world_builder import build_world_builder_prompt, run_world_builder
from tests.unit.conftest import (
    OLLAMA_BASE_URL,
    make_multi_tool_call_response,
    make_plain_tool_response,
    make_tool_call_response,
)

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"


def _inference(
    id: int = 1,
    statement: str = "Alice is left-handed",
    derivation: str = "observed",
) -> Inference:
    return Inference(
        id=id,
        character_id=1,
        statement=statement,
        derivation=derivation,
        created_at="2024-01-01T00:00:00",  # type: ignore[arg-type]
    )


def _one_round(tool_name: str, arguments: dict[str, Any]) -> list[httpx.Response]:
    """side_effect for a single author_set_facts call followed by a terminal plain reply."""
    return [
        httpx.Response(200, content=make_tool_call_response(tool_name, arguments)),
        httpx.Response(200, content=make_plain_tool_response("Done.")),
    ]


# ---------------------------------------------------------------------------
# build_world_builder_prompt — prompt content
# ---------------------------------------------------------------------------


def test_world_builder_prompt_includes_user_message(character: Character) -> None:
    prompt = build_world_builder_prompt("I am freezing, it must be snowing out", character, {})
    assert "I am freezing, it must be snowing out" in prompt


def test_world_builder_prompt_includes_fact_schema_section(character: Character) -> None:
    prompt = build_world_builder_prompt("Hello", character, {})
    assert "## Fact Schema" in prompt


def test_world_builder_prompt_includes_current_fact_values_section(
    character: Character,
) -> None:
    prompt = build_world_builder_prompt("Hello", character, {})
    assert "## Current Fact Values" in prompt


def test_world_builder_prompt_includes_populated_fact_value(character: Character) -> None:
    blob = {"Character": {"Identity": {"Occupation": {"Value": "surgeon"}}}}
    prompt = build_world_builder_prompt("Hello", character, blob)
    assert "surgeon" in prompt


def test_world_builder_prompt_includes_inferences(character: Character) -> None:
    inferences = [_inference(statement="Alice likely works weekends")]
    prompt = build_world_builder_prompt("Hello", character, {}, inferences)
    assert "Alice likely works weekends" in prompt


def test_world_builder_prompt_omits_inferences_section_when_none(character: Character) -> None:
    prompt = build_world_builder_prompt("Hello", character, {}, [])
    assert "(no inferences established yet)" in prompt


def test_world_builder_prompt_instructs_single_batched_call(character: Character) -> None:
    prompt = build_world_builder_prompt("Hello", character, {})
    assert "author_set_facts" in prompt
    assert "SINGLE call" in prompt


def test_world_builder_prompt_instructs_no_new_paths(character: Character) -> None:
    prompt = build_world_builder_prompt("Hello", character, {})
    assert "NOT invent new" in prompt


# ---------------------------------------------------------------------------
# run_world_builder — tool-call mechanics
# ---------------------------------------------------------------------------


async def test_run_world_builder_no_tool_call_returns_blob_unchanged(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    facts_blob = {"Character": {"Identity": {"Occupation": {"Value": "surgeon"}}}}
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(
                200, content=make_plain_tool_response("Nothing to extract.")
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="Hello",
            facts_blob=facts_blob,
            inferences=[],
            ollama=ollama,
        )
    assert result == facts_blob
    decisions = await get_decisions(db, session.id)
    assert decisions == []


async def test_run_world_builder_no_event_called_when_no_tool_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(
                200, content=make_plain_tool_response("Nothing to extract.")
            )
        )
        await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="Hello",
            facts_blob={},
            inferences=[],
            ollama=ollama,
            on_event=_on_event,
        )
    assert events == []


async def test_run_world_builder_single_fact_written_to_blob(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Occupation", "value": "Surgeon"}]},
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="I'm a surgeon",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Occupation"]["Value"] == "Surgeon"


async def test_run_world_builder_multiple_facts_in_one_call_all_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {
                    "facts": [
                        {"path": "Character.Identity.Occupation", "value": "Surgeon"},
                        {"path": "Character.Identity.Name", "value": "Alice"},
                        {"path": "Character.State-Of-Mind.Mood", "value": "Calm"},
                    ]
                },
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Occupation"]["Value"] == "Surgeon"
    assert result["Character"]["Identity"]["Name"]["Value"] == "Alice"
    assert result["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Calm"


async def test_run_world_builder_preserves_existing_unrelated_facts(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    facts_blob = {"Character": {"Identity": {"Name": {"Value": "Alice"}}}}
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Occupation", "value": "Surgeon"}]},
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="I'm a surgeon",
            facts_blob=facts_blob,
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Name"]["Value"] == "Alice"
    assert result["Character"]["Identity"]["Occupation"]["Value"] == "Surgeon"


async def test_run_world_builder_overwrites_existing_value_regardless_of_mutability(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    facts_blob = {"Character": {"Identity": {"Name": {"Value": "Alice"}}}}
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Name", "value": "Beatrice"}]},
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="My name is Beatrice now",
            facts_blob=facts_blob,
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Name"]["Value"] == "Beatrice"


async def test_run_world_builder_unknown_path_returns_per_entry_error_other_entries_still_applied(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {
                    "facts": [
                        {"path": "Character.Identity.NotARealPath", "value": "x"},
                        {"path": "Character.Identity.Occupation", "value": "Surgeon"},
                    ]
                },
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Occupation"]["Value"] == "Surgeon"
    assert "NotARealPath" not in result.get("Character", {}).get("Identity", {})


async def test_run_world_builder_enum_value_coerced_case_insensitively(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.State-Of-Mind.Mood", "value": "calm"}]},
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["State-Of-Mind"]["Mood"]["Value"] == "Calm"


async def test_run_world_builder_enum_invalid_value_not_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.State-Of-Mind.Mood", "value": "Apprehensive"}]},
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert "Mood" not in result.get("Character", {}).get("State-Of-Mind", {})


async def test_run_world_builder_integer_value_coerced_from_string(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Age", "value": "33"}]},
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Age"]["Value"] == 33


async def test_run_world_builder_integer_invalid_value_not_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Age", "value": "not a number"}]},
            )
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert "Age" not in result.get("Character", {}).get("Identity", {})


async def test_run_world_builder_multiple_sequential_calls_accumulate(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "author_set_facts",
                        {"facts": [{"path": "Character.Identity.Name", "value": "Alice"}]},
                    ),
                ),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "author_set_facts",
                        {"facts": [{"path": "Character.Identity.Occupation", "value": "Surgeon"}]},
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Done.")),
            ]
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Name"]["Value"] == "Alice"
    assert result["Character"]["Identity"]["Occupation"]["Value"] == "Surgeon"


# ---------------------------------------------------------------------------
# Decision logging
# ---------------------------------------------------------------------------


async def test_run_world_builder_logs_decision_when_facts_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Name", "value": "Alice"}]},
            )
        )
        await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    decisions = await get_decisions(db, session.id)
    assert len(decisions) == 1
    assert decisions[0].pass_name == "world_builder"
    assert decisions[0].tool_name == "author_set_facts"


async def test_run_world_builder_decision_tool_args_contains_facts_list(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Name", "value": "Alice"}]},
            )
        )
        await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    decisions = await get_decisions(db, session.id)
    assert "facts" in decisions[0].tool_args
    assert decisions[0].tool_args["facts"][0]["path"] == "Character.Identity.Name"


async def test_run_world_builder_no_decision_logged_when_no_tool_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            return_value=httpx.Response(
                200, content=make_plain_tool_response("Nothing to extract.")
            )
        )
        await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="Hello",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    decisions = await get_decisions(db, session.id)
    assert decisions == []


# ---------------------------------------------------------------------------
# Sidechannel notification
# ---------------------------------------------------------------------------


async def test_run_world_builder_emits_sidechannel_event_per_call(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Name", "value": "Alice"}]},
            )
        )
        await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
            on_event=_on_event,
        )
    assert len(events) == 1
    assert events[0].event == "sidechannel"
    assert events[0].data["type"] == "world_builder_applied"
    facts = events[0].data["facts"]
    assert facts == [{"path": "Character.Identity.Name", "value": "Alice"}]


async def test_run_world_builder_no_event_emitted_when_no_facts_written(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    events: list[SSEEvent] = []

    async def _on_event(ev: SSEEvent) -> None:
        events.append(ev)

    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.NotARealPath", "value": "x"}]},
            )
        )
        await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
            on_event=_on_event,
        )
    assert events == []


async def test_run_world_builder_on_event_none_does_not_raise(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=_one_round(
                "author_set_facts",
                {"facts": [{"path": "Character.Identity.Name", "value": "Alice"}]},
            )
        )
        await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
            on_event=None,
        )


# ---------------------------------------------------------------------------
# Cap behaviour
# ---------------------------------------------------------------------------


async def test_run_world_builder_cap_reached_keeps_partial_writes(
    db: aiosqlite.Connection,
    character: Character,
    session: Session,
    ollama: OllamaClient,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "author_set_facts",
                        {"facts": [{"path": "Character.Identity.Name", "value": "Alice"}]},
                    ),
                ),
                httpx.Response(
                    200,
                    content=make_tool_call_response(
                        "author_set_facts",
                        {"facts": [{"path": "Character.Identity.Occupation", "value": "Surgeon"}]},
                    ),
                ),
            ]
        )
        with caplog.at_level(logging.WARNING):
            result = await run_world_builder(
                db,
                character,
                session_id=session.id,
                turn_id=1,
                user_message="...",
                facts_blob={},
                inferences=[],
                ollama=ollama,
                max_rounds=2,
            )
    assert result["Character"]["Identity"]["Name"]["Value"] == "Alice"
    assert result["Character"]["Identity"]["Occupation"]["Value"] == "Surgeon"
    assert any("cap reached" in r.message for r in caplog.records)


async def test_run_world_builder_multi_tool_call_response_supported(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    """Sanity check: a single round with multiple tool_calls also works (not the prompted path)."""
    with respx.mock:
        respx.post(_CHAT_URL).mock(
            side_effect=[
                httpx.Response(
                    200,
                    content=make_multi_tool_call_response(
                        [
                            (
                                "author_set_facts",
                                {"facts": [{"path": "Character.Identity.Name", "value": "Alice"}]},
                            ),
                        ]
                    ),
                ),
                httpx.Response(200, content=make_plain_tool_response("Done.")),
            ]
        )
        result = await run_world_builder(
            db,
            character,
            session_id=session.id,
            turn_id=1,
            user_message="...",
            facts_blob={},
            inferences=[],
            ollama=ollama,
        )
    assert result["Character"]["Identity"]["Name"]["Value"] == "Alice"
