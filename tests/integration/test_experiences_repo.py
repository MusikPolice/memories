"""Integration tests for experience repository functions."""

from __future__ import annotations

import aiosqlite
import pytest

from memories.database import (
    _embedding_to_blob,
    _experience_embedding_cache,
    create_experience,
    create_session,
    delete_experience,
    get_experience,
    get_experiences,
    get_experiences_with_embeddings,
    get_previous_session,
    update_session_closing_journal,
)
from memories.exceptions import NotFoundError
from memories.models import Character, Experience, Session


@pytest.fixture
def embed_bytes() -> bytes:
    return _embedding_to_blob([0.1, 0.2, 0.3, 0.4])


@pytest.fixture
async def experience(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> Experience:
    return await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="User lives in Chicago",
        source="told_by_user",
        embedding=embed_bytes,
    )


# ---------------------------------------------------------------------------
# get_experience
# ---------------------------------------------------------------------------


async def test_get_experience_returns_experience_by_id(
    db: aiosqlite.Connection, experience: Experience
) -> None:
    result = await get_experience(db, experience.id)
    assert result is not None
    assert result.id == experience.id


async def test_get_experience_returns_none_for_unknown_id(
    db: aiosqlite.Connection,
) -> None:
    result = await get_experience(db, 99999)
    assert result is None


# ---------------------------------------------------------------------------
# create_experience
# ---------------------------------------------------------------------------


async def test_create_experience_returns_experience(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Test statement",
        source="observed",
        embedding=embed_bytes,
    )
    assert isinstance(exp, Experience)


async def test_create_experience_persists_to_db(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Persisted statement",
        source="told_by_user",
        embedding=embed_bytes,
    )
    row = await (await db.execute("SELECT id FROM experiences WHERE id = ?", (exp.id,))).fetchone()
    assert row is not None


async def test_create_experience_statement_stored(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="My specific statement",
        source="observed",
        embedding=embed_bytes,
    )
    assert exp.statement == "My specific statement"


async def test_create_experience_source_told_by_user(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Told statement",
        source="told_by_user",
        embedding=embed_bytes,
    )
    assert exp.source == "told_by_user"


async def test_create_experience_source_observed(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Observed statement",
        source="observed",
        embedding=embed_bytes,
    )
    assert exp.source == "observed"


async def test_create_experience_approved_at_is_set(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Statement",
        source="observed",
        embedding=embed_bytes,
    )
    assert exp.approved_at is not None


# ---------------------------------------------------------------------------
# get_experiences
# ---------------------------------------------------------------------------


async def test_get_experiences_returns_all_for_character(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    for i in range(3):
        await create_experience(
            db,
            character_id=character.id,
            session_id=session.id,
            statement=f"Experience {i}",
            source="told_by_user",
            embedding=embed_bytes,
        )
    results = await get_experiences(db, character.id)
    assert len(results) == 3


async def test_get_experiences_returns_empty_list_when_none(
    db: aiosqlite.Connection, character: Character
) -> None:
    results = await get_experiences(db, character.id)
    assert results == []


async def test_get_experiences_filters_by_character_id(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    from memories.database import create_character

    char2 = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    sess2 = await create_session(db, character_id=char2.id)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Alice's experience",
        source="observed",
        embedding=embed_bytes,
    )
    await create_experience(
        db,
        character_id=char2.id,
        session_id=sess2.id,
        statement="Bob's experience",
        source="observed",
        embedding=embed_bytes,
    )
    alice_exps = await get_experiences(db, character.id)
    bob_exps = await get_experiences(db, char2.id)
    assert len(alice_exps) == 1
    assert len(bob_exps) == 1
    assert alice_exps[0].statement == "Alice's experience"
    assert bob_exps[0].statement == "Bob's experience"


async def test_get_experiences_returned_in_creation_order(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    for label in ["first", "second", "third"]:
        await create_experience(
            db,
            character_id=character.id,
            session_id=session.id,
            statement=label,
            source="told_by_user",
            embedding=embed_bytes,
        )
    results = await get_experiences(db, character.id)
    assert [r.statement for r in results] == ["first", "second", "third"]


async def test_get_experiences_excludes_embedding_from_model(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Test",
        source="told_by_user",
        embedding=embed_bytes,
    )
    # Should not raise; embedding is not in the Experience model
    assert not hasattr(exp, "embedding")


# ---------------------------------------------------------------------------
# get_experiences_with_embeddings
# ---------------------------------------------------------------------------


async def test_get_experiences_with_embeddings_returns_vectors(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="With embedding",
        source="told_by_user",
        embedding=embed_bytes,
    )
    results = await get_experiences_with_embeddings(db, character.id)
    assert len(results) == 1
    exp, vec = results[0]
    assert isinstance(exp, Experience)
    assert isinstance(vec, list)
    assert all(isinstance(v, float) for v in vec)


async def test_get_experiences_with_embeddings_decodes_blob(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    original_vec = [0.1, 0.2, 0.3, 0.4]
    blob = _embedding_to_blob(original_vec)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Test",
        source="told_by_user",
        embedding=blob,
    )
    results = await get_experiences_with_embeddings(db, character.id)
    _, decoded_vec = results[0]
    assert decoded_vec == pytest.approx(original_vec)


async def test_get_experiences_with_embeddings_skips_null_embedding(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await db.execute(
        """INSERT INTO experiences
               (character_id, session_id, statement, source, embedding, approved_at)
           VALUES (?, ?, ?, ?, NULL, CURRENT_TIMESTAMP)""",
        (character.id, session.id, "No embedding", "observed"),
    )
    await db.commit()
    results = await get_experiences_with_embeddings(db, character.id)
    assert results == []


# ---------------------------------------------------------------------------
# Embedding cache behaviour
# ---------------------------------------------------------------------------


async def test_cache_populated_after_first_load(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Cached exp",
        source="told_by_user",
        embedding=embed_bytes,
    )
    await get_experiences_with_embeddings(db, character.id)
    assert character.id in _experience_embedding_cache
    assert len(_experience_embedding_cache[character.id]) == 1


async def test_cache_serves_result_without_db_on_second_call(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Should be cached",
        source="told_by_user",
        embedding=embed_bytes,
    )
    await get_experiences_with_embeddings(db, character.id)
    # Delete the row directly so a real DB query would return nothing
    await db.execute("DELETE FROM experiences WHERE id = ?", (exp.id,))
    await db.commit()
    # Evict delete's own cache update by re-inserting into cache manually is NOT needed —
    # we want to test that the *get* path uses the cache, so we bypass delete eviction
    # by restoring the cache entry that delete just removed.
    _experience_embedding_cache[character.id][exp.id] = (exp, [0.1, 0.2, 0.3, 0.4])
    results = await get_experiences_with_embeddings(db, character.id)
    assert len(results) == 1
    assert results[0][0].id == exp.id


async def test_cache_empty_dict_stored_when_no_experiences(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await get_experiences_with_embeddings(db, character.id)
    assert character.id in _experience_embedding_cache
    assert _experience_embedding_cache[character.id] == {}


async def test_create_experience_writes_through_to_loaded_cache(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    # Pre-warm the cache (will be empty for this character)
    await get_experiences_with_embeddings(db, character.id)
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Write-through test",
        source="told_by_user",
        embedding=embed_bytes,
    )
    assert exp.id in _experience_embedding_cache[character.id]


async def test_create_experience_does_not_populate_cache_before_first_load(
    db: aiosqlite.Connection, character: Character, session: Session, embed_bytes: bytes
) -> None:
    # Cache has never been loaded for this character
    assert character.id not in _experience_embedding_cache
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="No cache yet",
        source="told_by_user",
        embedding=embed_bytes,
    )
    assert character.id not in _experience_embedding_cache


async def test_delete_experience_evicts_from_cache(
    db: aiosqlite.Connection, character: Character, session: Session, experience: Experience
) -> None:
    await get_experiences_with_embeddings(db, character.id)
    assert experience.id in _experience_embedding_cache[character.id]
    await delete_experience(db, experience.id)
    assert experience.id not in _experience_embedding_cache.get(character.id, {})


async def test_delete_experience_no_op_when_cache_not_loaded(
    db: aiosqlite.Connection, experience: Experience
) -> None:
    # Cache was never loaded; delete should not raise
    await delete_experience(db, experience.id)


# ---------------------------------------------------------------------------
# delete_experience
# ---------------------------------------------------------------------------


async def test_delete_experience_returns_none_on_success(
    db: aiosqlite.Connection, experience: Experience
) -> None:
    result = await delete_experience(db, experience.id)
    assert result is None


async def test_delete_experience_removes_row_from_db(
    db: aiosqlite.Connection, character: Character, experience: Experience
) -> None:
    await delete_experience(db, experience.id)
    remaining = await get_experiences(db, character.id)
    assert not any(e.id == experience.id for e in remaining)


async def test_delete_experience_raises_not_found_for_unknown_id(
    db: aiosqlite.Connection,
) -> None:
    with pytest.raises(NotFoundError):
        await delete_experience(db, 99999)


# ---------------------------------------------------------------------------
# update_session_closing_journal
# ---------------------------------------------------------------------------


async def test_update_session_closing_journal_stores_text(
    db: aiosqlite.Connection, session: Session
) -> None:
    updated = await update_session_closing_journal(db, session.id, "Memorable session.")
    assert updated.closing_journal == "Memorable session."


async def test_update_session_closing_journal_raises_not_found(
    db: aiosqlite.Connection,
) -> None:
    with pytest.raises(NotFoundError):
        await update_session_closing_journal(db, 99999, "Text")


# ---------------------------------------------------------------------------
# get_previous_session
# ---------------------------------------------------------------------------


async def test_get_previous_session_returns_most_recent_with_journal(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    # session is session 1 (no journal initially)
    # session2 with a journal
    session2 = await create_session(db, character_id=character.id)
    await update_session_closing_journal(db, session2.id, "Session 2 journal")
    # session3 is the "current" session
    session3 = await create_session(db, character_id=character.id)

    result = await get_previous_session(db, character.id, before_session_id=session3.id)
    assert result is not None
    assert result.id == session2.id


async def test_get_previous_session_excludes_session_without_journal(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    # session has no closing journal
    session2 = await create_session(db, character_id=character.id)
    result = await get_previous_session(db, character.id, before_session_id=session2.id)
    assert result is None


async def test_get_previous_session_excludes_same_and_later_sessions(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    await update_session_closing_journal(db, session.id, "Past journal")
    # Add a later session with a journal — should NOT be returned for session.id
    session2 = await create_session(db, character_id=character.id)
    await update_session_closing_journal(db, session2.id, "Later journal")

    result = await get_previous_session(db, character.id, before_session_id=session.id)
    assert result is None


async def test_get_previous_session_returns_none_when_no_prior_sessions(
    db: aiosqlite.Connection, character: Character, session: Session
) -> None:
    # session is the first session; no prior sessions
    result = await get_previous_session(db, character.id, before_session_id=session.id)
    assert result is None
