"""Evaluator LLM service.

Runs a second Ollama call after each character response to check it against
established Facts.  Returns a structured verdict that chat_service uses to
decide whether to deliver, regenerate, or surface a notification.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from memories.models import Character, Fact
from memories.services.ollama_client import OllamaClient

_VALID_VERDICTS = frozenset(
    {
        "pass",
        "contradiction",
        "implication",
        "new_inference_logical",
        "new_inference_probabilistic",
        "experience_update",
    }
)


class EvaluatorParseError(Exception):
    """Raised when the evaluator returns unparseable or invalid JSON."""


class NewInference(BaseModel):
    inference_type: str
    statement: str
    derivation: str
    source_fact_ids: list[int] = []
    source_inference_ids: list[int] = []


class Violation(BaseModel):
    type: str
    description: str
    suggested_fact: dict[str, str] | None = None


class ContradictionNotification(BaseModel):
    iteration: int
    description: str


class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    decision_log: str = ""
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False


def build_evaluator_prompt(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
) -> str:
    """Build the user-facing content for the evaluator Ollama call."""
    parts: list[str] = [f"Character: {character.name}"]

    parts.append("\n## Established Facts")
    if facts:
        for f in facts:
            parts.append(f"{f.key}: {f.value}")
    else:
        parts.append("(no facts established yet)")

    parts.append(f'\n## Conversation Context\nUser said: "{user_message}"')
    parts.append(f"\n## Character Response to Evaluate\n{character_response}")

    if contradiction_hints:
        parts.append("\n## Previously Flagged Contradictions")
        for hint in contradiction_hints:
            parts.append(f"- {hint}")

    parts.append(
        """
## Your Task
Analyze the character's response against the established Facts above.
Return a JSON object with this exact structure:

{
  "verdict": "<pass|contradiction|implication|new_inference_logical|new_inference_probabilistic>",
  "new_inferences": [
    {
      "inference_type": "logical | probabilistic",
      "statement": "...",
      "derivation": "brief explanation of how this follows from the facts",
      "source_fact_ids": [],
      "source_inference_ids": []
    }
  ],
  "violations": [
    {
      "type": "contradiction | implication",
      "description": "what was wrong",
      "suggested_fact": {"key": "...", "value": "..."} or null
    }
  ],
  "decision_log": "One-sentence summary of why you chose this verdict."
}

Verdict definitions:
- pass: response is grounded; no contradictions or unestablished details
- contradiction: response contradicts an established Fact (highest priority — overrides all others)
- implication: response asserts an unestablished, non-derivable detail (a new Fact is implied)
- new_inference_logical: response asserts something derivable from Facts by logic (not yet stored)
- new_inference_probabilistic: something likely but not strictly derivable from Facts

Return only the JSON object, no other text."""
    )

    return "\n".join(parts)


async def run_evaluator(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    ollama: OllamaClient,
    contradiction_hints: list[str] | None = None,
) -> EvaluatorResult:
    """Run the evaluator LLM and return a parsed verdict."""
    prompt = build_evaluator_prompt(
        character, facts, user_message, character_response, contradiction_hints
    )
    model = character.current_model_name or character.modelfile_base
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a strict fact-checker for a character roleplay system. "
                "Evaluate the character's response against their established facts. "
                "Return only valid JSON following the schema you are given."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    content, _ = await ollama.chat(model, messages, think=False, format="json")

    try:
        data: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise EvaluatorParseError(f"Evaluator returned non-JSON content: {content!r}") from exc

    verdict = data.get("verdict")

    # experience_update is Phase 4; coerce to pass now
    if verdict == "experience_update":
        data["verdict"] = "pass"
        verdict = "pass"

    if verdict not in _VALID_VERDICTS - {"experience_update"}:
        raise EvaluatorParseError(f"Unknown evaluator verdict: {verdict!r}")

    if "decision_log" not in data:
        raise EvaluatorParseError("Evaluator response missing required 'decision_log' field")

    # Contradiction priority: if any violation has type "contradiction", force the verdict
    violations_raw: list[dict[str, Any]] = data.get("violations", []) or []
    if any(v.get("type") == "contradiction" for v in violations_raw):
        data["verdict"] = "contradiction"

    try:
        return EvaluatorResult.model_validate(data)
    except ValidationError as exc:
        raise EvaluatorParseError(f"Failed to validate evaluator result: {exc}") from exc
