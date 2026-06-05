from __future__ import annotations

from datetime import datetime
from typing import Any

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
    created_at: datetime


class Session(BaseModel):
    id: int
    character_id: int
    started_at: datetime
    ended_at: datetime | None = None
    closing_journal: str | None = None


class Segment(BaseModel):
    id: int
    session_id: int
    start_turn: int
    end_turn: int | None = None
    boundary_reason: str | None = None
    status: str = "verbatim"
    journal_text: str | None = None
    created_at: datetime


class Message(BaseModel):
    id: int
    character_id: int
    session_id: int
    segment_id: int
    role: str
    content: str
    turn_id: int
    captured_by: list[str] | None = None
    ungrounded_implications: list[dict[str, Any]] | None = None
    created_at: datetime


class Decision(BaseModel):
    id: int
    character_id: int
    session_id: int
    turn_id: int
    reasoning: str
    verdict: str
    violations: list[dict[str, Any]] | None = None
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
