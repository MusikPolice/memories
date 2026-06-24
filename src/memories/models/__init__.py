from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class Character(BaseModel):
    id: int
    name: str
    modelfile_base: str
    current_model_name: str | None = None
    created_at: datetime


class Fact(BaseModel):
    id: int
    character_id: int
    key: str
    value: str
    category: Literal["user", "character", "setting"] = "character"
    mutability: Literal["immutable", "low", "high"] = "immutable"
    created_at: datetime


class Session(BaseModel):
    id: int
    character_id: int
    started_at: datetime
    ended_at: datetime | None = None
    closing_journal: str | None = None


class Message(BaseModel):
    id: int
    character_id: int
    session_id: int
    role: str
    content: str
    turn_id: int
    created_at: datetime


class Decision(BaseModel):
    id: int
    character_id: int
    session_id: int
    turn_id: int
    pass_name: str
    tool_name: str
    tool_args: dict[str, Any]
    user_input: dict[str, Any] | None = None
    created_at: datetime


class Inference(BaseModel):
    id: int
    character_id: int
    statement: str
    derivation: str
    source_fact_ids: list[int] = []
    source_inference_ids: list[int] = []
    depth: int = 1
    inference_type: str = "logical"
    status: str = "active"
    created_at: datetime


class Experience(BaseModel):
    id: int
    character_id: int
    session_id: int
    statement: str
    source: Literal["told_by_user", "observed"]
    approved_at: datetime
    created_at: datetime
