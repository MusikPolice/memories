"""Integration tests for GET /api/schema."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from memories.main import app


@pytest.mark.anyio
async def test_get_schema_returns_200(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_get_schema_content_type_is_json(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "application/json" in response.headers["content-type"]


@pytest.mark.anyio
async def test_get_schema_has_character_key(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "Character" in response.json()


@pytest.mark.anyio
async def test_get_schema_has_user_key(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "User" in response.json()


@pytest.mark.anyio
async def test_get_schema_has_setting_key(client: AsyncClient) -> None:
    response = await client.get("/api/schema")
    assert "Setting" in response.json()


@pytest.mark.anyio
async def test_get_schema_character_identity_name_is_leaf(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    leaf = data["Character"]["Identity"]["Name"]
    assert "Type" in leaf
    assert "Mutability" in leaf
    assert "Description" in leaf


@pytest.mark.anyio
async def test_get_schema_leaf_type_is_string(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["Identity"]["Name"]["Type"] == "String"


@pytest.mark.anyio
async def test_get_schema_leaf_mutability_is_immutable(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["Identity"]["Name"]["Mutability"] == "Immutable"


@pytest.mark.anyio
async def test_get_schema_enum_leaf_has_constraint_list(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    mood = data["Character"]["State-Of-Mind"]["Mood"]
    assert mood["Type"] == "Enum"
    assert isinstance(mood["Constraint"], list)
    assert len(mood["Constraint"]) > 0


@pytest.mark.anyio
async def test_get_schema_fluid_leaf_mutability(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["State-Of-Mind"]["Mood"]["Mutability"] == "Fluid"


@pytest.mark.anyio
async def test_get_schema_mutable_leaf_mutability(client: AsyncClient) -> None:
    data = (await client.get("/api/schema")).json()
    assert data["Character"]["Appearance"]["Outfit"]["Top"]["Mutability"] == "Mutable"


@pytest.mark.anyio
async def test_get_schema_no_db_access_required() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/schema")
    assert response.status_code == 200
