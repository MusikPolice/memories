"""Integration tests for the facts API."""

from httpx import AsyncClient


async def test_add_fact_201(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"}
    )
    assert response.status_code == 201
    assert response.json()["key"] == "age"


async def test_add_fact_duplicate_key_409(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.post(
        f"/api/characters/{char_id}/facts", json={"key": "age", "value": "31"}
    )
    assert response.status_code == 409


async def test_list_facts_for_character(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.get(f"/api/characters/{char_id}/facts")
    assert response.status_code == 200
    assert len(response.json()) == 1


async def test_update_fact_200(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.put(f"/api/characters/{char_id}/facts/age", json={"value": "31"})
    assert response.status_code == 200
    assert response.json()["value"] == "31"


async def test_update_nonexistent_fact_404(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.put(f"/api/characters/{char_id}/facts/nonexistent", json={"value": "x"})
    assert response.status_code == 404


async def test_delete_fact_204(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    await client.post(f"/api/characters/{char_id}/facts", json={"key": "age", "value": "30"})
    response = await client.delete(f"/api/characters/{char_id}/facts/age")
    assert response.status_code == 204


async def test_delete_nonexistent_fact_404(client: AsyncClient) -> None:
    char_resp = await client.post(
        "/api/characters/", json={"name": "Alice", "modelfile_base": "qwen3:7b"}
    )
    char_id = char_resp.json()["id"]
    response = await client.delete(f"/api/characters/{char_id}/facts/nonexistent")
    assert response.status_code == 404


async def test_facts_for_unknown_character_404(client: AsyncClient) -> None:
    response = await client.get("/api/characters/9999/facts")
    assert response.status_code == 404
