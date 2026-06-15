"""Schema introspection endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from memories.schema_loader import load_schema

router = APIRouter()


@router.get("/schema")
async def get_schema() -> dict[str, Any]:
    """Return the fact schema as JSON.

    Loaded once at first request and cached in memory for the lifetime of the process.
    """
    return load_schema()
