"""Integration tests for the experiences API endpoints."""

from __future__ import annotations

import json

import aiosqlite
import httpx
import respx
from httpx import AsyncClient

from memories.database import _embedding_to_blob, create_character, create_experience
from memories.database import create_session as _create_session
from memories.models import Character, Session
from memories.services.experience_service import add_active_experiences, get_active_experiences

_OLLAMA_BASE = "http://test-ollama-integration:11434"
_EMBED_URL = f"{_OLLAMA_BASE}/api/embed"
_EMBED_VEC = [1.0, 0.0, 0.0, 0.0]


def _mock_embed() -> httpx.Response:
    return httpx.Response(200, json={"embeddings": [_EMBED_VEC]})


# ---------------------------------------------------------------------------
# Experience creation (from accepted proposal)
# ---------------------------------------------------------------------------


async def test_create_experience_returns_201(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        response = await client.post(
            f"/api/characters/{character.id}/experiences",
            json={
                "session_id": session.id,
                "statement": "User lives in Chicago",
                "source": "told_by_user",
            },
        )
    assert response.status_code == 201


async def test_create_experience_response_has_statement(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        response = await client.post(
            f"/api/characters/{character.id}/experiences",
            json={
                "session_id": session.id,
                "statement": "User lives in Chicago",
                "source": "told_by_user",
            },
        )
    assert response.json()["statement"] == "User lives in Chicago"


async def test_create_experience_response_has_source(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        response = await client.post(
            f"/api/characters/{character.id}/experiences",
            json={
                "session_id": session.id,
                "statement": "User lives in Chicago",
                "source": "told_by_user",
            },
        )
    assert response.json()["source"] == "told_by_user"


async def test_create_experience_response_has_id(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        response = await client.post(
            f"/api/characters/{character.id}/experiences",
            json={
                "session_id": session.id,
                "statement": "Test statement",
                "source": "observed",
            },
        )
    assert isinstance(response.json()["id"], int)


async def test_create_experience_response_has_no_embedding_field(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        response = await client.post(
            f"/api/characters/{character.id}/experiences",
            json={"session_id": session.id, "statement": "Test", "source": "observed"},
        )
    assert "embedding" not in response.json()


async def test_create_experience_persisted_to_db(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        await client.post(
            f"/api/characters/{character.id}/experiences",
            json={
                "session_id": session.id,
                "statement": "Persisted experience",
                "source": "told_by_user",
            },
        )
    list_resp = await client.get(f"/api/characters/{character.id}/experiences")
    statements = [e["statement"] for e in list_resp.json()]
    assert "Persisted experience" in statements


async def test_create_experience_calls_embed_endpoint(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        route = respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        await client.post(
            f"/api/characters/{character.id}/experiences",
            json={
                "session_id": session.id,
                "statement": "Embed this statement",
                "source": "told_by_user",
            },
        )
    assert route.called
    body = json.loads(route.calls[0].request.content)
    assert body["input"] == "Embed this statement"


async def test_create_experience_unknown_character_returns_404(
    client: AsyncClient, session: Session
) -> None:
    response = await client.post(
        "/api/characters/99999/experiences",
        json={"session_id": session.id, "statement": "Test", "source": "observed"},
    )
    assert response.status_code == 404


async def test_create_experience_unknown_session_returns_404(
    client: AsyncClient, character: Character
) -> None:
    response = await client.post(
        f"/api/characters/{character.id}/experiences",
        json={"session_id": 99999, "statement": "Test", "source": "observed"},
    )
    assert response.status_code == 404


async def test_create_experience_session_wrong_character_returns_404(
    db: aiosqlite.Connection, client: AsyncClient, character: Character
) -> None:
    char2 = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    sess2 = await _create_session(db, character_id=char2.id)
    response = await client.post(
        f"/api/characters/{character.id}/experiences",
        json={"session_id": sess2.id, "statement": "Test", "source": "observed"},
    )
    assert response.status_code == 404


async def test_create_experience_invalid_source_returns_422(
    client: AsyncClient, character: Character, session: Session
) -> None:
    response = await client.post(
        f"/api/characters/{character.id}/experiences",
        json={"session_id": session.id, "statement": "Test", "source": "unknown"},
    )
    assert response.status_code == 422


async def test_create_experience_told_by_user_source_accepted(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        response = await client.post(
            f"/api/characters/{character.id}/experiences",
            json={"session_id": session.id, "statement": "Test told", "source": "told_by_user"},
        )
    assert response.status_code == 201


async def test_create_experience_observed_source_accepted(
    client: AsyncClient, character: Character, session: Session
) -> None:
    with respx.mock:
        respx.post(_EMBED_URL).mock(return_value=_mock_embed())
        response = await client.post(
            f"/api/characters/{character.id}/experiences",
            json={"session_id": session.id, "statement": "Test observed", "source": "observed"},
        )
    assert response.status_code == 201


# ---------------------------------------------------------------------------
# Experience list
# ---------------------------------------------------------------------------


async def test_list_experiences_returns_200(client: AsyncClient, character: Character) -> None:
    response = await client.get(f"/api/characters/{character.id}/experiences")
    assert response.status_code == 200


async def test_list_experiences_returns_all_experiences(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    blob = _embedding_to_blob(_EMBED_VEC)
    for i in range(3):
        await create_experience(
            db,
            character_id=character.id,
            session_id=session.id,
            statement=f"Experience {i}",
            source="told_by_user",
            embedding=blob,
        )
    response = await client.get(f"/api/characters/{character.id}/experiences")
    assert len(response.json()) == 3


async def test_list_experiences_returns_empty_list_when_none(
    client: AsyncClient, character: Character
) -> None:
    response = await client.get(f"/api/characters/{character.id}/experiences")
    assert response.json() == []


async def test_list_experiences_unknown_character_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/characters/99999/experiences")
    assert response.status_code == 404


async def test_list_experiences_has_no_embedding_field_in_items(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    blob = _embedding_to_blob(_EMBED_VEC)
    await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Test",
        source="observed",
        embedding=blob,
    )
    response = await client.get(f"/api/characters/{character.id}/experiences")
    for item in response.json():
        assert "embedding" not in item


# ---------------------------------------------------------------------------
# Experience delete
# ---------------------------------------------------------------------------


async def test_delete_experience_returns_204(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    blob = _embedding_to_blob(_EMBED_VEC)
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="To delete",
        source="observed",
        embedding=blob,
    )
    response = await client.delete(f"/api/characters/{character.id}/experiences/{exp.id}")
    assert response.status_code == 204


async def test_delete_experience_removes_from_db(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    blob = _embedding_to_blob(_EMBED_VEC)
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Removable",
        source="observed",
        embedding=blob,
    )
    await client.delete(f"/api/characters/{character.id}/experiences/{exp.id}")
    list_resp = await client.get(f"/api/characters/{character.id}/experiences")
    ids = [e["id"] for e in list_resp.json()]
    assert exp.id not in ids


async def test_delete_experience_unknown_id_returns_404(
    client: AsyncClient, character: Character
) -> None:
    response = await client.delete(f"/api/characters/{character.id}/experiences/99999")
    assert response.status_code == 404


async def test_delete_experience_unknown_character_returns_404(client: AsyncClient) -> None:
    response = await client.delete("/api/characters/99999/experiences/1")
    assert response.status_code == 404


async def test_delete_experience_clears_experience_from_active_sessions(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    blob = _embedding_to_blob(_EMBED_VEC)
    exp = await create_experience(
        db,
        character_id=character.id,
        session_id=session.id,
        statement="Active experience",
        source="observed",
        embedding=blob,
    )
    add_active_experiences(session.id, [exp])
    assert any(e.id == exp.id for e in get_active_experiences(session.id))

    await client.delete(f"/api/characters/{character.id}/experiences/{exp.id}")

    assert not any(e.id == exp.id for e in get_active_experiences(session.id))


async def test_delete_experience_wrong_character_returns_404(
    db: aiosqlite.Connection, client: AsyncClient, character: Character, session: Session
) -> None:
    char2 = await create_character(db, name="Bob", modelfile_base="qwen3:7b")
    sess2 = await _create_session(db, character_id=char2.id)
    blob = _embedding_to_blob(_EMBED_VEC)
    exp = await create_experience(
        db,
        character_id=char2.id,
        session_id=sess2.id,
        statement="Bob's experience",
        source="observed",
        embedding=blob,
    )
    response = await client.delete(f"/api/characters/{character.id}/experiences/{exp.id}")
    assert response.status_code == 404
