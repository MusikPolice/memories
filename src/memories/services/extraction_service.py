"""Fact extraction service — Phase 6.

Analyses the user's message for explicit and implicit facts before the
character LLM is invoked.  Returns a structured ExtractionResult that
chat_service uses to write Tier 1/2 facts before the character call and
surface Tier 3/4 proposals post-response.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from memories.models import Character, Fact, Inference
from memories.services.ollama_client import OllamaClient


class ExtractionParseError(Exception):
    """Raised when the extractor returns unparseable or invalid JSON."""


class ExtractedFact(BaseModel):
    """A Tier 1 extraction: explicit new fact stated by the user."""

    key: str
    value: str
    category: str = "setting"
    mutability: str = "low"
    source_quote: str = ""
    fact_id: int | None = None


class FactUpdate(BaseModel):
    """A Tier 2 extraction: explicit update to an existing fact."""

    fact_id: int
    key: str
    old_value: str
    new_value: str
    source_quote: str = ""


class ImplicitProposal(BaseModel):
    """A Tier 3/4 extraction: implied fact or fact update requiring user confirmation.

    Tier 3 (new): existing_fact_id is None.
    Tier 4 (conflicting): existing_fact_id is set; old_value carries the current stored value.
    """

    key: str
    value: str
    category: str
    mutability: str
    source_quote: str = ""
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
    """Build the user-facing prompt for the fact extractor LLM call."""
    parts: list[str] = [f"Character: {character.name}"]

    parts.append("\n## Established Facts (id: category/mutability: key = value)")
    if facts:
        for f in facts:
            parts.append(f"[{f.id}] ({f.category}, {f.mutability}): {f.key} = {f.value}")
    else:
        parts.append("(no established facts yet)")

    if inferences:
        parts.append("\n## Established Inferences (id: statement)")
        for inf in inferences:
            parts.append(f"[{inf.id}] {inf.statement}  (from: {inf.derivation})")

    parts.append(f'\n## User Message\n"{user_message}"')

    parts.append(
        """
## Your Task
Extract factual information stated in the user's message and classify it into one
of four tiers:

**Tier 1 — Explicit new fact**: The user states a clear, unambiguous fact about a
topic not yet covered by any established Fact. Add to new_facts.

**Tier 2 — Explicit fact update**: The user states a clear, unambiguous new value
for a topic already covered by an established Fact (identified by its id in the
Established Facts list above). Add to fact_updates with the correct fact_id.

**Tier 3 — Implicit new fact**: The user implies a fact but does not state it
outright. Add to implicit_proposals with existing_fact_id as null.

**Tier 4 — Implicit fact update**: The user implies a change to an existing Fact.
Add to implicit_proposals with existing_fact_id set and old_value.

Return a JSON object with this exact structure:

{
  "new_facts": [
    {
      "key": "...",
      "value": "...",
      "category": "user | setting",
      "mutability": "immutable | low | high",
      "source_quote": "exact phrase from the message"
    }
  ],
  "fact_updates": [
    {
      "fact_id": <integer id from Established Facts>,
      "key": "...",
      "old_value": "...",
      "new_value": "...",
      "source_quote": "exact phrase from the message"
    }
  ],
  "implicit_proposals": [
    {
      "key": "...",
      "value": "...",
      "category": "user | setting",
      "mutability": "immutable | low | high",
      "source_quote": "exact phrase from the message",
      "existing_fact_id": <integer id or null>,
      "old_value": "<current value if updating, or null>"
    }
  ]
}

Return only the JSON object. All three lists may be empty if no facts are extractable."""
    )

    return "\n".join(parts)


def parse_extraction_result(content: str) -> ExtractionResult:
    """Parse the raw JSON string from the extractor LLM into an ExtractionResult.

    Raises ExtractionParseError on invalid or incomplete JSON.
    """
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        stripped = "\n".join(lines[start:end]).strip()

    try:
        data: dict[str, Any] = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ExtractionParseError(f"Extractor returned non-JSON content: {content!r}") from exc

    try:
        result = ExtractionResult.model_validate(data)
    except ValidationError as exc:
        raise ExtractionParseError(f"Failed to validate extraction result: {exc}") from exc

    # Trim source_quote to 200 chars
    for nf in result.new_facts:
        nf.source_quote = nf.source_quote[:200]
    for fu in result.fact_updates:
        fu.source_quote = fu.source_quote[:200]
    for ip in result.implicit_proposals:
        ip.source_quote = ip.source_quote[:200]

    return result


async def run_fact_extractor(
    user_message: str,
    character: Character,
    facts: list[Fact],
    inferences: list[Inference],
    ollama: OllamaClient,
) -> ExtractionResult:
    """Issue an Ollama call to extract facts from the user message.

    Returns an ExtractionResult on success.
    Raises ExtractionParseError if the LLM response cannot be parsed.
    """
    prompt = build_extractor_prompt(user_message, character, facts, inferences)
    model = character.current_model_name or character.modelfile_base
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a precise fact-extraction system for a character roleplay application. "
                "Extract factual information from the user's message and return only valid JSON "
                "following the schema you are given."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    content, _ = await ollama.chat(model, messages, think=False, format="json")
    return parse_extraction_result(content)
