"""Integration tests for the characters API."""

from httpx import AsyncClient


async def test_create_character_201(client: AsyncClient) -> None:
    response = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    assert data["name"] == "Alice"


async def test_list_characters_empty_200(client: AsyncClient) -> None:
    response = await client.get("/api/characters/")
    assert response.status_code == 200
    assert response.json() == []


async def test_list_characters_populated(client: AsyncClient) -> None:
    await client.post("/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"})
    await client.post("/api/characters/", json={"name": "Bob", "modelfile_base": "qwen3:7b"})
    response = await client.get("/api/characters/")
    assert response.status_code == 200
    assert len(response.json()) == 2


async def test_get_character_200(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = create_resp.json()["id"]
    response = await client.get(f"/api/characters/{char_id}")
    assert response.status_code == 200
    assert response.json()["name"] == "Alice"


async def test_get_character_404(client: AsyncClient) -> None:
    response = await client.get("/api/characters/9999")
    assert response.status_code == 404
