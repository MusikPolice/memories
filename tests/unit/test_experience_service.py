"""Unit tests for memories.services.experience_service."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import aiosqlite
import httpx
import pytest
import respx

from memories.database import _blob_to_embedding, _embedding_to_blob, create_experience
from memories.models import Character, Experience, Fact, Inference, Message, Session
from memories.services.experience_service import (
    EMBED_MODEL,
    SessionEndParseError,
    SessionEndResult,
    _dot,
    add_active_experiences,
    build_session_end_prompt,
    clear_active_experiences,
    cold_start_retrieve,
    get_active_experiences,
    remove_active_experience,
    remove_experience_from_all_sessions,
    retrieve_experiences,
    retrieve_top_k,
    run_session_end_evaluator,
)
from memories.services.ollama_client import OllamaClient
from tests.unit.conftest import OLLAMA_BASE_URL, make_embed_response, make_ollama_ndjson

_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embed"
_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"

_NOW = datetime(2026, 1, 1)

_CHARACTER = Character(id=1, name="Alice", modelfile_base="qwen3:7b", created_at=_NOW)

_FACTS = [
    Fact(id=1, character_id=1, key="occupation", value="surgeon", created_at=_NOW),
]

_INFERENCES = [
    Inference(
        id=1,
        character_id=1,
        statement="Alice works long hours",
        derivation="occupation=surgeon",
        source_fact_ids=[1],
        source_inference_ids=[],
        depth=1,
        inference_type="probabilistic",
        status="active",
        created_at=_NOW,
    )
]


def _make_experience(
    exp_id: int = 1,
    statement: str = "Test statement",
    source: str = "told_by_user",
    character_id: int = 1,
    session_id: int = 1,
) -> Experience:
    return Experience(
        id=exp_id,
        character_id=character_id,
        session_id=session_id,
        statement=statement,
        source=source,  # type: ignore[arg-type]
        approved_at=_NOW,
        created_at=_NOW,
    )


def _make_message(
    msg_id: int = 1,
    role: str = "user",
    content: str = "Hello",
    turn_id: int = 1,
    session_id: int = 1,
    character_id: int = 1,
    segment_id: int = 1,
) -> Message:
    return Message(
        id=msg_id,
        character_id=character_id,
        session_id=session_id,
        segment_id=segment_id,
        role=role,
        content=content,
        turn_id=turn_id,
        created_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Embedding helpers (in database.py, tested here for convenience)
# ---------------------------------------------------------------------------


def test_embedding_round_trip() -> None:
    vec = [1.0, 2.0, 3.0]
    assert _blob_to_embedding(_embedding_to_blob(vec)) == vec


def test_embedding_to_blob_returns_bytes() -> None:
    result = _embedding_to_blob([0.1, 0.2])
    assert isinstance(result, bytes)


def test_blob_to_embedding_returns_list_of_floats() -> None:
    blob = _embedding_to_blob([0.5, 0.6, 0.7])
    result = _blob_to_embedding(blob)
    assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# Dot product / similarity
# ---------------------------------------------------------------------------


def test_dot_product_of_identical_unit_vectors_is_one() -> None:
    assert _dot([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_dot_product_of_orthogonal_vectors_is_zero() -> None:
    assert _dot([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_dot_product_of_opposite_unit_vectors_is_minus_one() -> None:
    assert _dot([1.0, 0.0], [-1.0, 0.0]) == -1.0


def test_dot_raises_value_error_on_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        _dot([1.0, 0.0], [1.0])


# ---------------------------------------------------------------------------
# retrieve_top_k
# ---------------------------------------------------------------------------


def _candidates(n: int, scores: list[float] | None = None) -> list[tuple[Experience, list[float]]]:
    """Create n candidate (Experience, vec) pairs with unit vectors as embeddings."""
    pairs = []
    for i in range(n):
        exp = _make_experience(exp_id=i + 1, statement=f"Statement {i + 1}")
        # Simple unit-ish embedding: score with [1.0] query = values[i]
        score = scores[i] if scores else 1.0 / (i + 1)
        vec = [score, 0.0]
        pairs.append((exp, vec))
    return pairs


def test_retrieve_top_k_returns_most_similar() -> None:
    candidates = _candidates(3, scores=[0.9, 0.5, 0.3])
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=1)
    assert len(result) == 1
    assert result[0].id == 1  # highest score


def test_retrieve_top_k_respects_k_limit() -> None:
    candidates = _candidates(5)
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=2)
    assert len(result) == 2


def test_retrieve_top_k_excludes_active_ids() -> None:
    candidates = _candidates(3, scores=[0.9, 0.5, 0.3])
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=1, exclude_ids={1})
    assert len(result) == 1
    assert result[0].id == 2  # most similar after excluding id=1


def test_retrieve_top_k_returns_empty_when_all_excluded() -> None:
    candidates = _candidates(3)
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=3, exclude_ids={1, 2, 3})
    assert result == []


def test_retrieve_top_k_returns_empty_for_empty_candidates() -> None:
    result = retrieve_top_k([1.0, 0.0], [], k=5)
    assert result == []


def test_retrieve_top_k_handles_k_greater_than_candidates() -> None:
    candidates = _candidates(3)
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=10)
    assert len(result) == 3


def test_retrieve_top_k_excludes_candidates_below_min_score() -> None:
    candidates = _candidates(3, scores=[0.9, 0.5, 0.1])
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=3, min_score=0.3)
    ids = {e.id for e in result}
    assert 1 in ids and 2 in ids  # scores 0.9 and 0.5 pass
    assert 3 not in ids  # score 0.1 is below threshold


def test_retrieve_top_k_returns_empty_when_all_below_min_score() -> None:
    candidates = _candidates(3, scores=[0.1, 0.2, 0.05])
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=3, min_score=0.5)
    assert result == []


def test_retrieve_top_k_min_score_zero_includes_zero_scoring_candidates() -> None:
    candidates = _candidates(2, scores=[0.5, 0.0])
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=2, min_score=0.0)
    assert len(result) == 2  # score=0.0 passes the >= 0.0 threshold


def test_retrieve_top_k_min_score_excludes_negative_scoring_candidates() -> None:
    candidates = _candidates(2, scores=[0.5, -0.3])
    query = [1.0, 0.0]
    result = retrieve_top_k(query, candidates, k=2, min_score=0.0)
    assert len(result) == 1
    assert result[0].id == 1  # only the 0.5-scorer passes


# ---------------------------------------------------------------------------
# Active experience tracking
# ---------------------------------------------------------------------------


def test_get_active_experiences_returns_empty_for_unknown_session() -> None:
    assert get_active_experiences(99) == []


def test_add_active_experiences_adds_to_session() -> None:
    exp = _make_experience()
    add_active_experiences(1, [exp])
    result = get_active_experiences(1)
    assert exp in result


def test_add_active_experiences_deduplicates_by_id() -> None:
    exp = _make_experience()
    add_active_experiences(1, [exp])
    add_active_experiences(1, [exp])
    result = get_active_experiences(1)
    assert result.count(exp) == 1


def test_add_active_experiences_does_not_affect_other_sessions() -> None:
    exp = _make_experience()
    add_active_experiences(1, [exp])
    assert get_active_experiences(2) == []


def test_remove_active_experience_removes_by_id() -> None:
    exp = _make_experience()
    add_active_experiences(1, [exp])
    remove_active_experience(1, exp.id)
    assert get_active_experiences(1) == []


def test_remove_active_experience_no_op_for_unknown_id() -> None:
    remove_active_experience(1, 9999)  # should not raise


def test_clear_active_experiences_empties_session() -> None:
    exp = _make_experience()
    add_active_experiences(1, [exp])
    clear_active_experiences(1)
    assert get_active_experiences(1) == []


def test_clear_active_experiences_no_op_for_unknown_session() -> None:
    clear_active_experiences(99)  # should not raise


def test_remove_experience_from_all_sessions_clears_across_sessions() -> None:
    exp1 = _make_experience(exp_id=1)
    exp2 = _make_experience(exp_id=2)
    add_active_experiences(10, [exp1, exp2])
    add_active_experiences(20, [exp1])
    remove_experience_from_all_sessions(1)
    assert exp1 not in get_active_experiences(10)
    assert exp2 in get_active_experiences(10)
    assert exp1 not in get_active_experiences(20)


def test_remove_experience_from_all_sessions_no_op_when_not_present() -> None:
    exp = _make_experience(exp_id=5)
    add_active_experiences(10, [exp])
    remove_experience_from_all_sessions(999)  # unknown id — should not raise
    assert exp in get_active_experiences(10)


def test_remove_experience_from_all_sessions_no_op_when_no_sessions() -> None:
    remove_experience_from_all_sessions(1)  # empty dict — should not raise


# ---------------------------------------------------------------------------
# retrieve_experiences (mocked Ollama + real in-memory DB)
# ---------------------------------------------------------------------------


async def _insert_experiences_with_embeddings(
    db: aiosqlite.Connection, character_id: int, session_id: int, n: int
) -> list[Experience]:
    exps = []
    for i in range(n):
        vec = [float(i + 1), 0.0, 0.0, 0.0]
        blob = _embedding_to_blob(vec)
        exp = await create_experience(
            db,
            character_id=character_id,
            session_id=session_id,
            statement=f"Experience {i + 1}",
            source="told_by_user",
            embedding=blob,
        )
        exps.append(exp)
    return exps


@respx.mock
async def test_retrieve_experiences_calls_embed_endpoint(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experiences_with_embeddings(db, character.id, session.id, 1)
    route = respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    await retrieve_experiences(db, character.id, "test query", ollama)
    assert route.called


@respx.mock
async def test_retrieve_experiences_sends_embed_model_name(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experiences_with_embeddings(db, character.id, session.id, 1)
    route = respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    await retrieve_experiences(db, character.id, "test query", ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("model") == EMBED_MODEL


@respx.mock
async def test_retrieve_experiences_returns_top_k_new_experiences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experiences_with_embeddings(db, character.id, session.id, 5)
    respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    new_exps, _ = await retrieve_experiences(db, character.id, "test", ollama, top_k=2)
    assert len(new_exps) == 2


@respx.mock
async def test_retrieve_experiences_skips_excluded_ids_in_new_experiences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    exps = await _insert_experiences_with_embeddings(db, character.id, session.id, 3)
    respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    exclude = {exps[0].id}
    new_exps, _ = await retrieve_experiences(
        db, character.id, "test", ollama, top_k=3, exclude_ids=exclude
    )
    ids = {e.id for e in new_exps}
    assert exps[0].id not in ids


@respx.mock
async def test_retrieve_experiences_excluded_id_still_appears_in_scores(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    exps = await _insert_experiences_with_embeddings(db, character.id, session.id, 3)
    respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    exclude = {exps[0].id}
    _, scores = await retrieve_experiences(
        db, character.id, "test", ollama, top_k=2, exclude_ids=exclude
    )
    assert exps[0].id in scores


@respx.mock
async def test_retrieve_experiences_scores_covers_all_stored_experiences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experiences_with_embeddings(db, character.id, session.id, 5)
    respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    _, scores = await retrieve_experiences(db, character.id, "test", ollama, top_k=2)
    assert len(scores) == 5


@respx.mock
async def test_retrieve_experiences_scores_are_floats(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experiences_with_embeddings(db, character.id, session.id, 2)
    respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    _, scores = await retrieve_experiences(db, character.id, "test", ollama)
    for v in scores.values():
        assert isinstance(v, float)


@respx.mock
async def test_retrieve_experiences_returns_empty_tuple_when_no_db_entries(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    route = respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response())
    )
    new_exps, scores = await retrieve_experiences(db, character.id, "test", ollama)
    assert new_exps == []
    assert scores == {}
    assert not route.called


@respx.mock
async def test_retrieve_experiences_returns_empty_tuple_when_no_embeddings(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    # Insert an experience with NULL embedding via raw SQL
    await db.execute(
        """INSERT INTO experiences
               (character_id, session_id, statement, source, embedding, approved_at)
           VALUES (?, ?, ?, ?, NULL, CURRENT_TIMESTAMP)""",
        (character.id, session.id, "No embedding", "observed"),
    )
    await db.commit()
    route = respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response())
    )
    new_exps, scores = await retrieve_experiences(db, character.id, "test", ollama)
    assert new_exps == []
    assert scores == {}
    assert not route.called


@respx.mock
async def test_retrieve_experiences_returns_empty_tuple_on_ollama_error(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    await _insert_experiences_with_embeddings(db, character.id, session.id, 1)
    respx.post(_EMBED_URL).mock(return_value=httpx.Response(503, content=b"unavailable"))
    new_exps, scores = await retrieve_experiences(db, character.id, "test", ollama)
    assert new_exps == []
    assert scores == {}


# ---------------------------------------------------------------------------
# cold_start_retrieve (mocked Ollama + real DB)
# ---------------------------------------------------------------------------


@respx.mock
async def test_cold_start_retrieve_returns_empty_when_no_previous_session(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    route = respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response())
    )
    result = await cold_start_retrieve(db, character.id, session.id, ollama)
    assert result == []
    assert not route.called


@respx.mock
async def test_cold_start_retrieve_returns_empty_when_no_closing_journal(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    # Create a second session; first session has no closing journal
    session2 = await __import__("memories.database", fromlist=["create_session"]).create_session(
        db, character_id=character.id
    )
    route = respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response())
    )
    result = await cold_start_retrieve(db, character.id, session2.id, ollama)
    assert result == []
    assert not route.called


@respx.mock
async def test_cold_start_retrieve_embeds_closing_journal(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import create_session as _create_session
    from memories.database import update_session_closing_journal

    await update_session_closing_journal(db, session.id, "A memorable day.")
    session2 = await _create_session(db, character_id=character.id)
    await _insert_experiences_with_embeddings(db, character.id, session.id, 1)

    route = respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    await cold_start_retrieve(db, character.id, session2.id, ollama)
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert "A memorable day." in body.get("input", "")


@respx.mock
async def test_cold_start_retrieve_returns_retrieved_experiences(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import create_session as _create_session
    from memories.database import update_session_closing_journal

    await update_session_closing_journal(db, session.id, "Journal text.")
    session2 = await _create_session(db, character_id=character.id)
    await _insert_experiences_with_embeddings(db, character.id, session.id, 2)

    respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    result = await cold_start_retrieve(db, character.id, session2.id, ollama)
    assert len(result) > 0
    assert all(isinstance(e, Experience) for e in result)


@respx.mock
async def test_cold_start_retrieve_returns_list_not_tuple(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import create_session as _create_session
    from memories.database import update_session_closing_journal

    await update_session_closing_journal(db, session.id, "Journal.")
    session2 = await _create_session(db, character_id=character.id)
    await _insert_experiences_with_embeddings(db, character.id, session.id, 1)

    respx.post(_EMBED_URL).mock(
        return_value=httpx.Response(200, content=make_embed_response([1.0, 0.0, 0.0, 0.0]))
    )
    result = await cold_start_retrieve(db, character.id, session2.id, ollama)
    assert isinstance(result, list)


@respx.mock
async def test_cold_start_retrieve_returns_empty_on_ollama_error(
    db: aiosqlite.Connection, character: Character, session: Session, ollama: OllamaClient
) -> None:
    from memories.database import create_session as _create_session
    from memories.database import update_session_closing_journal

    await update_session_closing_journal(db, session.id, "Journal text.")
    session2 = await _create_session(db, character_id=character.id)
    await _insert_experiences_with_embeddings(db, character.id, session.id, 1)

    respx.post(_EMBED_URL).mock(return_value=httpx.Response(503, content=b"unavailable"))
    result = await cold_start_retrieve(db, character.id, session2.id, ollama)
    assert result == []


# ---------------------------------------------------------------------------
# build_session_end_prompt
# ---------------------------------------------------------------------------


def test_session_end_prompt_includes_character_name() -> None:
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, [])
    assert "Alice" in prompt


def test_session_end_prompt_includes_all_facts() -> None:
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, [])
    assert "occupation" in prompt
    assert "surgeon" in prompt


def test_session_end_prompt_includes_all_inferences() -> None:
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, [])
    assert "Alice works long hours" in prompt


def test_session_end_prompt_includes_all_messages() -> None:
    messages = [
        _make_message(msg_id=1, role="user", content="Hello there", turn_id=1),
        _make_message(msg_id=2, role="assistant", content="Hi, how are you?", turn_id=1),
    ]
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, messages)
    assert "Hello there" in prompt
    assert "Hi, how are you?" in prompt


def test_session_end_prompt_labels_user_messages() -> None:
    messages = [_make_message(role="user", content="User message", turn_id=1)]
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, messages)
    assert "User" in prompt


def test_session_end_prompt_labels_character_messages() -> None:
    messages = [_make_message(role="assistant", content="Character message", turn_id=1)]
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, messages)
    assert _CHARACTER.name in prompt


def test_session_end_prompt_includes_task_instructions() -> None:
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, [])
    prompt_lower = prompt.lower()
    assert "closing journal" in prompt_lower
    assert "proposed experiences" in prompt_lower or "proposed_experiences" in prompt_lower


def test_session_end_prompt_no_facts_shows_fallback() -> None:
    prompt = build_session_end_prompt(_CHARACTER, [], _INFERENCES, [])
    assert "(none)" in prompt


def test_session_end_prompt_no_inferences_shows_fallback() -> None:
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, [], [])
    assert "(none)" in prompt


def test_session_end_prompt_no_messages_shows_empty_section() -> None:
    prompt = build_session_end_prompt(_CHARACTER, _FACTS, _INFERENCES, [])
    assert "Full Conversation" in prompt


# ---------------------------------------------------------------------------
# run_session_end_evaluator (mocked Ollama)
# ---------------------------------------------------------------------------


def _session_end_ndjson(
    closing_journal: str = "It was a good session.",
    proposed: list[dict[str, Any]] | None = None,
) -> bytes:
    data = {
        "closing_journal": closing_journal,
        "proposed_experiences": proposed or [],
    }
    return make_ollama_ndjson(json.dumps(data))


@respx.mock
async def test_session_end_evaluator_returns_closing_journal(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_session_end_ndjson("A fine day indeed."))
    )
    result = await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    assert result.closing_journal == "A fine day indeed."


@respx.mock
async def test_session_end_evaluator_returns_proposed_experiences(ollama: OllamaClient) -> None:
    proposals = [
        {"statement": "Lives in Chicago", "source": "told_by_user", "turn_reference": 1},
        {"statement": "Likes jazz", "source": "observed", "turn_reference": 3},
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(
            200, content=_session_end_ndjson("Good session.", proposed=proposals)
        )
    )
    result = await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    assert len(result.proposed_experiences) == 2


@respx.mock
async def test_session_end_evaluator_experience_has_statement(ollama: OllamaClient) -> None:
    proposals = [{"statement": "User owns a cat", "source": "told_by_user"}]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_session_end_ndjson(proposed=proposals))
    )
    result = await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    assert result.proposed_experiences[0].statement == "User owns a cat"


@respx.mock
async def test_session_end_evaluator_experience_has_source(ollama: OllamaClient) -> None:
    proposals = [
        {"statement": "User owns a cat", "source": "told_by_user"},
        {"statement": "User seems sad", "source": "observed"},
    ]
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_session_end_ndjson(proposed=proposals))
    )
    result = await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    sources = {p.source for p in result.proposed_experiences}
    assert sources <= {"told_by_user", "observed"}


@respx.mock
async def test_session_end_evaluator_empty_proposals_is_valid(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_session_end_ndjson("Journal entry.", proposed=[]))
    )
    result = await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    assert isinstance(result, SessionEndResult)
    assert result.proposed_experiences == []


@respx.mock
async def test_session_end_evaluator_raises_on_non_json(ollama: OllamaClient) -> None:
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson("This is not JSON"))
    )
    with pytest.raises(SessionEndParseError):
        await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)


@respx.mock
async def test_session_end_evaluator_raises_on_missing_closing_journal(
    ollama: OllamaClient,
) -> None:
    # Missing required "closing_journal" field
    data = {"proposed_experiences": []}
    respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=make_ollama_ndjson(json.dumps(data)))
    )
    with pytest.raises(SessionEndParseError):
        await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)


@respx.mock
async def test_session_end_evaluator_sends_think_false(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_session_end_ndjson())
    )
    await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("think") is False


@respx.mock
async def test_session_end_evaluator_sends_format_json(ollama: OllamaClient) -> None:
    route = respx.post(_CHAT_URL).mock(
        return_value=httpx.Response(200, content=_session_end_ndjson())
    )
    await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    body = json.loads(route.calls[0].request.content)
    assert body.get("format") == "json"


@respx.mock
async def test_session_end_evaluator_strips_markdown_code_fences(ollama: OllamaClient) -> None:
    data = {"closing_journal": "Good session.", "proposed_experiences": []}
    fenced = "```json\n" + json.dumps(data) + "\n```"
    respx.post(_CHAT_URL).mock(return_value=httpx.Response(200, content=make_ollama_ndjson(fenced)))
    result = await run_session_end_evaluator(_CHARACTER, _FACTS, _INFERENCES, [], ollama)
    assert result.closing_journal == "Good session."
