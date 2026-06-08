"""Fact extraction service — Phase 6.

Analyses the user's message for explicit and implicit facts before the
character LLM is invoked.  Returns a structured ExtractionResult that
chat_service uses to write Tier 1/2 facts before the character call and
surface Tier 3/4 proposals post-response.

Phase 6 stub: all functions raise NotImplementedError.
"""

from __future__ import annotations

from pydantic import BaseModel

from memories.models import Character, Fact, Inference
from memories.services.ollama_client import OllamaClient


class ExtractionParseError(Exception):
    """Raised when the extractor returns unparseable or invalid JSON."""


class ExtractedFact(BaseModel):
    """A Tier 1 extraction: explicit new fact stated by the user."""

    key: str
    value: str
    category: str
    mutability: str
    source_quote: str


class FactUpdate(BaseModel):
    """A Tier 2 extraction: explicit update to an existing fact."""

    fact_id: int
    key: str
    old_value: str
    new_value: str
    source_quote: str


class ImplicitProposal(BaseModel):
    """A Tier 3/4 extraction: implied fact or fact update requiring user confirmation.

    Tier 3 (new): existing_fact_id is None.
    Tier 4 (conflicting): existing_fact_id is set; old_value carries the current stored value.
    """

    key: str
    value: str
    category: str
    mutability: str
    source_quote: str
    existing_fact_id: int | None = None
    old_value: str | None = None


class ExtractionResult(BaseModel):
    """Structured output from run_fact_extractor."""

    new_facts: list[ExtractedFact] = []
    fact_updates: list[FactUpdate] = []
    implicit_proposals: list[ImplicitProposal] = []


def build_extractor_prompt(
    user_message: str,
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],
) -> str:
    """Build the system + user prompt for the fact extractor LLM call.

    Phase 6: not yet implemented.
    """
    raise NotImplementedError("Phase 6: build_extractor_prompt not implemented")


def parse_extraction_result(content: str) -> ExtractionResult:
    """Parse the raw JSON string from the extractor LLM into an ExtractionResult.

    Raises ExtractionParseError on invalid or incomplete JSON.

    Phase 6: not yet implemented.
    """
    raise NotImplementedError("Phase 6: parse_extraction_result not implemented")


async def run_fact_extractor(
    user_message: str,
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],
    ollama: OllamaClient,
) -> ExtractionResult:
    """Issue a non-streaming Ollama call to extract facts from the user message.

    Returns an ExtractionResult on success.
    Raises ExtractionParseError if the LLM response cannot be parsed.

    Phase 6: not yet implemented.
    """
    raise NotImplementedError("Phase 6: run_fact_extractor not implemented")
