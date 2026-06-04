"""Integration tests for the sessions API."""

from httpx import AsyncClient


async def test_start_session_201(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post("/api/sessions/", json={"character_id": char_id})
    assert response.status_code == 201
    assert "id" in response.json()


async def test_start_session_unknown_character_404(client: AsyncClient) -> None:
    response = await client.post("/api/sessions/", json={"character_id": 9999})
    assert response.status_code == 404


async def test_end_session_200(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    sess_resp = await client.post("/api/sessions/", json={"character_id": char_id})
    session_id = sess_resp.json()["id"]
    response = await client.post(f"/api/sessions/{session_id}/end")
    assert response.status_code == 200
    assert response.json()["ended_at"] is not None


async def test_end_unknown_session_404(client: AsyncClient) -> None:
    response = await client.post("/api/sessions/9999/end")
    assert response.status_code == 404


async def test_get_session_messages_initially_empty(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    sess_resp = await client.post("/api/sessions/", json={"character_id": char_id})
    session_id = sess_resp.json()["id"]
    response = await client.get(f"/api/sessions/{session_id}/messages")
    assert response.status_code == 200
    assert response.json() == []
