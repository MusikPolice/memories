"""Unit tests for memories.services.inference_service."""

from __future__ import annotations

import json
from datetime import datetime

import aiosqlite
import httpx
import pytest
import respx

from memories.database import create_inference, get_inferences
from memories.models import Character, Fact, Inference
from memories.services.inference_service import (
    MAX_INFERENCE_DEPTH,
    InferenceParseError,
    build_eager_pass_prompt,
    build_revalidation_prompt,
    cascade_on_fact_delete,
    cascade_on_fact_edit,
    compute_depth,
    revalidate_single_inference,
    run_eager_pass,
)
from memories.services.ollama_client import OllamaClient
from tests.unit.conftest import OLLAMA_BASE_URL, make_ollama_ndjson

_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
_NOW = datetime(2026, 1, 1, 12, 0, 0)

_CHARACTER = Character(
    id=1,
    name="Alice",
    modelfile_base="qwen3:7b",
    created_at=_NOW,
)

_FACTS = [
    Fact(id=1, character_id=1, key="age", value="33", created_at=_NOW),
    Fact(id=2, character_id=1, key="current_year", value="2026", created_at=_NOW),
]

_EXISTING_INFERENCES = [
    Inference(
        id=10,
        character_id=1,
        statement="Alice was born in 1993",
        derivation="age=33, current_year=2026",
        source_fact_ids=[1, 2],
        source_inference_ids=[],
        depth=1,
        inference_type="logical",
        status="active",
        created_at=_NOW,
    )
]

_EAGER_PASS_ITEM = {
    "inference_type": "logical",
    "statement": "Alice was born in 1993",
    "derivation": "age=33, current_year=2026",
    "source_fact_ids": [1, 2],
    "source_inference_ids": [],
}


# ---------------------------------------------------------------------------
# compute_depth — pure function, no fixtures needed
# ---------------------------------------------------------------------------


def test_compute_depth_returns_one_for_empty_source_inference_ids() -> None:
    assert compute_depth([], _EXISTING_INFERENCES) == 1


def test_compute_depth_returns_source_depth_plus_one() -> None:
    sources = [
        Inference(
            id=5,
            character_id=1,
            statement="S",
            derivation="d",
            depth=3,
            status="active",
            created_at=_NOW,
        )
    ]
    assert compute_depth([5], sources) == 4


def test_compute_depth_takes_max_of_multiple_sources() -> None:
    sources = [
        Inference(
            id=5,
            character_id=1,
            statement="S1",
            derivation="d",
            depth=2,
            status="active",
            created_at=_NOW,
        ),
        Inference(
            id=6,
            character_id=1,
            statement="S2",
            derivation="d",
            depth=4,
            status="active",
            created_at=_NOW,
        ),
    ]
    assert compute_depth([5, 6], sources) == 5


def test_compute_depth_skips_unknown_source_ids() -> None:
    sources = [
        Inference(
            id=5,
            character_id=1,
            statement="S",
            derivation="d",
            depth=3,
            status="active",
            created_at=_NOW,
        )
    ]
    # id 99 doesn't exist; id 5 has depth 3
    assert compute_depth([5, 99], sources) == 4


def test_compute_depth_returns_one_when_all_sources_unknown() -> None:
    sources = [
        Inference(
            id=5,
            character_id=1,
            statement="S",
            derivation="d",
            depth=3,
            status="active",
            created_at=_NOW,
        )
    ]
    assert compute_depth([99, 100], sources) == 1


# ---------------------------------------------------------------------------
# build_eager_pass_prompt — pure function
# ---------------------------------------------------------------------------


def test_eager_pass_prompt_includes_all_facts() -> None:
    prompt = build_eager_pass_prompt(_CHARACTER, _FACTS, [], 5)
    assert "age: 33" in prompt or ("age" in prompt and "33" in prompt)
    assert "current_year: 2026" in prompt or ("current_year" in prompt and "2026" in prompt)


def test_eager_pass_prompt_includes_character_name() -> None:
    prompt = build_eager_pass_prompt(_CHARACTER, _FACTS, [], 5)
    assert "Alice" in prompt


def test_eager_pass_prompt_lists_existing_inferences() -> None:
    prompt = build_eager_pass_prompt(_CHARACTER, _FACTS, _EXISTING_INFERENCES, 5)
    assert "Alice was born in 1993" in prompt


def test_eager_pass_prompt_no_existing_inferences_uses_fallback() -> None:
    prompt = build_eager_pass_prompt(_CHARACTER, _FACTS, [], 5)
    assert "(none established yet)" in prompt


def test_eager_pass_prompt_instructs_max_breadth() -> None:
    prompt = build_eager_pass_prompt(_CHARACTER, _FACTS, [], 10)
    assert "10" in prompt


# ---------------------------------------------------------------------------
# run_eager_pass — requires db, ollama, respx mocks
# ---------------------------------------------------------------------------


@respx.mock
async def test_eager_pass_parses_returned_inferences(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    item = dict(_EAGER_PASS_ITEM, source_fact_ids=[fact.id])
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([item])))
    )
    result = await run_eager_pass(db, character, [fact], [], ollama)
    assert len(result) == 1
    assert result[0].statement == "Alice was born in 1993"


@respx.mock
async def test_eager_pass_stores_inferences_to_db(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    item = dict(_EAGER_PASS_ITEM, source_fact_ids=[fact.id])
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([item])))
    )
    await run_eager_pass(db, character, [fact], [], ollama)
    stored = await get_inferences(db, character.id)
    assert len(stored) == 1
    assert stored[0].statement == "Alice was born in 1993"


@respx.mock
async def test_eager_pass_returns_empty_list_on_empty_json_array(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([])))
    )
    result = await run_eager_pass(db, character, [fact], [], ollama)
    assert result == []
    stored = await get_inferences(db, character.id)
    assert stored == []


@respx.mock
async def test_eager_pass_discards_inference_exceeding_max_depth(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    # source_inference_ids references an existing inference at depth=MAX_INFERENCE_DEPTH
    # so the new one would be depth=MAX+1, which exceeds the cap
    existing = await create_inference(
        db,
        character_id=character.id,
        statement="Deep inference",
        derivation="d",
        depth=MAX_INFERENCE_DEPTH,
    )
    item = {
        "inference_type": "logical",
        "statement": "Too deep",
        "derivation": "from deep source",
        "source_fact_ids": [],
        "source_inference_ids": [existing.id],
    }
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([item])))
    )
    result = await run_eager_pass(
        db, character, [fact], [existing], ollama, max_depth=MAX_INFERENCE_DEPTH
    )
    assert result == []
    stored = await get_inferences(db, character.id)
    # Only the pre-existing one, not the too-deep one
    assert all(s.statement != "Too deep" for s in stored)


@respx.mock
async def test_eager_pass_applies_breadth_cap(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    items = [
        {
            "inference_type": "logical",
            "statement": f"Inference {i}",
            "derivation": "d",
            "source_fact_ids": [fact.id],
            "source_inference_ids": [],
        }
        for i in range(7)
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps(items)))
    )
    result = await run_eager_pass(db, character, [fact], [], ollama, max_breadth=5)
    assert len(result) == 5


@respx.mock
async def test_eager_pass_rejects_same_pass_cross_reference(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    # source_inference_ids references id=999 which is NOT in existing_inferences
    item = {
        "inference_type": "logical",
        "statement": "Cross-referenced",
        "derivation": "d",
        "source_fact_ids": [fact.id],
        "source_inference_ids": [999],
    }
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([item])))
    )
    result = await run_eager_pass(db, character, [fact], [], ollama)
    assert result == []


@respx.mock
async def test_eager_pass_computes_depth_one_for_fact_only_source(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    item = {
        "inference_type": "logical",
        "statement": "Fact-only derived",
        "derivation": "from fact",
        "source_fact_ids": [fact.id],
        "source_inference_ids": [],
    }
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([item])))
    )
    result = await run_eager_pass(db, character, [fact], [], ollama)
    assert result[0].depth == 1


@respx.mock
async def test_eager_pass_computes_depth_from_source_inference(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    existing = await create_inference(
        db,
        character_id=character.id,
        statement="Existing at depth 2",
        derivation="d",
        depth=2,
    )
    item = {
        "inference_type": "logical",
        "statement": "Derived from depth-2",
        "derivation": "from existing",
        "source_fact_ids": [],
        "source_inference_ids": [existing.id],
    }
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([item])))
    )
    result = await run_eager_pass(db, character, [fact], [existing], ollama)
    assert result[0].depth == 3


@respx.mock
async def test_eager_pass_raises_on_non_json_response(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("This is not JSON at all."))
    )
    with pytest.raises(InferenceParseError):
        await run_eager_pass(db, character, [fact], [], ollama)


@respx.mock
async def test_eager_pass_request_sends_format_json(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([])))
    )
    await run_eager_pass(db, character, [fact], [], ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("format") == "json"


@respx.mock
async def test_eager_pass_request_sends_think_false(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps([])))
    )
    await run_eager_pass(db, character, [fact], [], ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("think") is False


# ---------------------------------------------------------------------------
# build_revalidation_prompt — pure function
# ---------------------------------------------------------------------------

_REVALIDATION_INFERENCE = Inference(
    id=1,
    character_id=1,
    statement="Alice was born in 1993",
    derivation="age=33, current_year=2026",
    source_fact_ids=[1, 2],
    source_inference_ids=[],
    depth=1,
    inference_type="logical",
    status="active",
    created_at=_NOW,
)

_OTHER_INFERENCE = Inference(
    id=2,
    character_id=1,
    statement="Alice works long hours",
    derivation="occupation=surgeon",
    source_fact_ids=[3],
    source_inference_ids=[],
    depth=1,
    inference_type="probabilistic",
    status="active",
    created_at=_NOW,
)


def test_revalidation_prompt_includes_inference_statement() -> None:
    prompt = build_revalidation_prompt(_REVALIDATION_INFERENCE, _FACTS, [_OTHER_INFERENCE])
    assert "Alice was born in 1993" in prompt


def test_revalidation_prompt_includes_inference_derivation() -> None:
    prompt = build_revalidation_prompt(_REVALIDATION_INFERENCE, _FACTS, [_OTHER_INFERENCE])
    assert "age=33, current_year=2026" in prompt


def test_revalidation_prompt_includes_current_facts() -> None:
    prompt = build_revalidation_prompt(_REVALIDATION_INFERENCE, _FACTS, [])
    assert "age" in prompt
    assert "33" in prompt
    assert "current_year" in prompt
    assert "2026" in prompt


def test_revalidation_prompt_includes_other_active_inferences() -> None:
    prompt = build_revalidation_prompt(_REVALIDATION_INFERENCE, _FACTS, [_OTHER_INFERENCE])
    assert "Alice works long hours" in prompt


# ---------------------------------------------------------------------------
# revalidate_single_inference — requires ollama mock
# ---------------------------------------------------------------------------


def _revalidation_ndjson(holds: bool) -> bytes:
    return make_ollama_ndjson(json.dumps({"holds": holds, "reason": "test"}))


@respx.mock
async def test_revalidate_returns_true_when_inference_holds(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=_revalidation_ndjson(True)))
    result = await revalidate_single_inference(_REVALIDATION_INFERENCE, _FACTS, [], ollama)
    assert result is True


@respx.mock
async def test_revalidate_returns_false_when_inference_does_not_hold(
    ollama: OllamaClient,
) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_revalidation_ndjson(False))
    )
    result = await revalidate_single_inference(_REVALIDATION_INFERENCE, _FACTS, [], ollama)
    assert result is False


@respx.mock
async def test_revalidate_defaults_to_true_on_parse_error(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("this is not json"))
    )
    result = await revalidate_single_inference(_REVALIDATION_INFERENCE, _FACTS, [], ollama)
    assert result is True


# ---------------------------------------------------------------------------
# cascade_on_fact_edit — requires db, character, fact, ollama
# ---------------------------------------------------------------------------


@respx.mock
async def test_cascade_edit_marks_directly_dependent_inference_stale(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    inf = await create_inference(
        db,
        character_id=character.id,
        statement="Alice works in medicine",
        derivation="occupation=surgeon",
        source_fact_ids=[fact.id],
    )
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"holds": False, "reason": "changed"}))
        )
    )
    stale = await cascade_on_fact_edit(db, character.id, fact.id, ollama)
    assert any(s.id == inf.id for s in stale)
    refreshed = await get_inferences(db, character.id, status="stale")
    assert any(s.id == inf.id for s in refreshed)


@respx.mock
async def test_cascade_edit_leaves_unrelated_inference_active(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    unrelated = await create_inference(
        db,
        character_id=character.id,
        statement="Unrelated inference",
        derivation="from other fact",
        source_fact_ids=[999],  # different fact id
    )
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"holds": False, "reason": "changed"}))
        )
    )
    await cascade_on_fact_edit(db, character.id, fact.id, ollama)
    active = await get_inferences(db, character.id, status="active")
    assert any(a.id == unrelated.id for a in active)


@respx.mock
async def test_cascade_edit_transitively_marks_chained_inference_stale(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    inf_a = await create_inference(
        db,
        character_id=character.id,
        statement="Inference A",
        derivation="from fact",
        source_fact_ids=[fact.id],
        depth=1,
    )
    inf_b = await create_inference(
        db,
        character_id=character.id,
        statement="Inference B",
        derivation="from Inf A",
        source_fact_ids=[],
        source_inference_ids=[inf_a.id],
        depth=2,
    )
    # Both revalidation calls return holds=False
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"holds": False, "reason": "changed"}))
        )
    )
    stale = await cascade_on_fact_edit(db, character.id, fact.id, ollama)
    stale_ids = {s.id for s in stale}
    assert inf_a.id in stale_ids
    assert inf_b.id in stale_ids


@respx.mock
async def test_cascade_edit_returns_stale_inferences(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    await create_inference(
        db,
        character_id=character.id,
        statement="Dep inf",
        derivation="from fact",
        source_fact_ids=[fact.id],
    )
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"holds": False, "reason": "no"}))
        )
    )
    result = await cascade_on_fact_edit(db, character.id, fact.id, ollama)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(r.status == "stale" for r in result)


@respx.mock
async def test_cascade_edit_skips_already_stale_inferences(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    # Insert a stale inference that depends on the fact
    cols = (
        "character_id, statement, derivation, "
        "source_fact_ids, source_inference_ids, depth, inference_type, status"
    )
    await db.execute(
        f"INSERT INTO inferences ({cols}) VALUES (?, ?, ?, ?, ?, 1, 'logical', 'stale')",
        (character.id, "Already stale", "from fact", json.dumps([fact.id]), json.dumps([])),
    )
    await db.commit()
    # No Ollama call should be made for already-stale inferences
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"holds": False, "reason": "no"}))
        )
    )
    await cascade_on_fact_edit(db, character.id, fact.id, ollama)
    # The already-stale inference should not trigger an Ollama call
    assert route.call_count == 0


@respx.mock
async def test_cascade_edit_does_not_mark_if_revalidation_returns_true(
    db: aiosqlite.Connection, character: Character, fact: Fact, ollama: OllamaClient
) -> None:
    inf = await create_inference(
        db,
        character_id=character.id,
        statement="Still valid inference",
        derivation="from fact",
        source_fact_ids=[fact.id],
    )
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=make_ollama_ndjson(json.dumps({"holds": True, "reason": "still ok"}))
        )
    )
    stale = await cascade_on_fact_edit(db, character.id, fact.id, ollama)
    assert not any(s.id == inf.id for s in stale)
    active = await get_inferences(db, character.id, status="active")
    assert any(a.id == inf.id for a in active)


# ---------------------------------------------------------------------------
# cascade_on_fact_delete — requires db, character, fact; NO ollama
# ---------------------------------------------------------------------------


async def test_cascade_delete_marks_directly_dependent_inference_invalidated(
    db: aiosqlite.Connection, character: Character, fact: Fact
) -> None:
    inf = await create_inference(
        db,
        character_id=character.id,
        statement="Depends on deleted fact",
        derivation="from fact",
        source_fact_ids=[fact.id],
    )
    invalidated = await cascade_on_fact_delete(db, character.id, fact.id)
    assert any(i.id == inf.id for i in invalidated)
    invalidated_db = await get_inferences(db, character.id, status="invalidated")
    assert any(i.id == inf.id for i in invalidated_db)


async def test_cascade_delete_leaves_unrelated_inference_active(
    db: aiosqlite.Connection, character: Character, fact: Fact
) -> None:
    unrelated = await create_inference(
        db,
        character_id=character.id,
        statement="Unrelated",
        derivation="d",
        source_fact_ids=[999],
    )
    await cascade_on_fact_delete(db, character.id, fact.id)
    active = await get_inferences(db, character.id, status="active")
    assert any(a.id == unrelated.id for a in active)


async def test_cascade_delete_transitively_marks_chained_inference_invalidated(
    db: aiosqlite.Connection, character: Character, fact: Fact
) -> None:
    inf_a = await create_inference(
        db,
        character_id=character.id,
        statement="Inf A",
        derivation="from fact",
        source_fact_ids=[fact.id],
        depth=1,
    )
    inf_b = await create_inference(
        db,
        character_id=character.id,
        statement="Inf B",
        derivation="from Inf A",
        source_fact_ids=[],
        source_inference_ids=[inf_a.id],
        depth=2,
    )
    invalidated = await cascade_on_fact_delete(db, character.id, fact.id)
    inv_ids = {i.id for i in invalidated}
    assert inf_a.id in inv_ids
    assert inf_b.id in inv_ids


async def test_cascade_delete_returns_all_invalidated_inferences(
    db: aiosqlite.Connection, character: Character, fact: Fact
) -> None:
    await create_inference(
        db,
        character_id=character.id,
        statement="Dep 1",
        derivation="d",
        source_fact_ids=[fact.id],
    )
    await create_inference(
        db,
        character_id=character.id,
        statement="Dep 2",
        derivation="d",
        source_fact_ids=[fact.id],
    )
    result = await cascade_on_fact_delete(db, character.id, fact.id)
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(r.status == "invalidated" for r in result)


@respx.mock
async def test_cascade_delete_no_llm_call_made(
    db: aiosqlite.Connection, character: Character, fact: Fact
) -> None:
    await create_inference(
        db,
        character_id=character.id,
        statement="Will be invalidated",
        derivation="d",
        source_fact_ids=[fact.id],
    )
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("{}"))
    )
    await cascade_on_fact_delete(db, character.id, fact.id)
    assert route.call_count == 0
