"""Evaluator LLM service.

Runs a second Ollama call after each character response to check it against
established Facts.  Returns a structured verdict that chat_service uses to
decide whether to deliver, regenerate, or surface a notification.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from memories.models import Character, Experience, Inference
from memories.schema_loader import render_current_fact_values, render_schema_for_prompt
from memories.services.ollama_client import OllamaClient

_VALID_VERDICTS = frozenset(
    {
        "pass",
        "contradiction",
        "new_inference_logical",
        "new_inference_probabilistic",
    }
)


class EvaluatorParseError(Exception):
    """Raised when the evaluator returns unparseable or invalid JSON."""


class NewInference(BaseModel):
    inference_type: str
    statement: str
    derivation: str
    source_fact_paths: list[str] = []
    source_inference_ids: list[int] = []


class Violation(BaseModel):
    type: str
    description: str


class ContradictionNotification(BaseModel):
    iteration: int
    description: str


class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    decision_log: str
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False


def build_evaluator_prompt(
    character: Character,
    facts_blob: dict[str, Any],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
    """Build the user-facing content for the evaluator Ollama call."""
    parts: list[str] = [f"Character: {character.name}"]

    parts.append("")
    parts.append(render_schema_for_prompt())

    parts.append("")
    parts.append(render_current_fact_values(facts_blob))

    parts.append("\n## Established Inferences (id: statement)")
    if inferences:
        for inf in inferences:
            parts.append(f"[{inf.id}] {inf.statement}  (from: {inf.derivation})")
    else:
        parts.append("(no inferences established yet)")

    parts.append("\n## Active Experiences (id: statement  [source])")
    if experiences:
        for exp in experiences:
            parts.append(f"[{exp.id}] {exp.statement}  [{exp.source}]")
    else:
        parts.append("(no active experiences this session)")

    parts.append(f'\n## Conversation Context\nUser said: "{user_message}"')
    parts.append(f"\n## Character Response to Evaluate\n{character_response}")

    if contradiction_hints:
        parts.append("\n## Previously Flagged Contradictions")
        for hint in contradiction_hints:
            parts.append(f"- {hint}")

    parts.append(
        """
## Your Task
Analyze the character's response against the Fact Schema and Current Fact Values above.

IMMUTABLE facts: any response that contradicts an immutable fact that is already set
is a `contradiction` — the value cannot change and the response must be regenerated.

MUTABLE facts: if the character implies a change to a mutable fact, note it in
`decision_log` but do not flag it as a contradiction. It will be surfaced for user
approval separately.

FLUID facts: changes are expected and accepted silently. Do not flag them.

For details the character invented that have no matching schema path, derive an
inference (`new_inference_logical` or `new_inference_probabilistic`).

Return a JSON object with this exact structure:

{
  "verdict": "<pass|contradiction|new_inference_logical|new_inference_probabilistic>",
  "new_inferences": [
    {
      "inference_type": "logical | probabilistic",
      "statement": "...",
      "derivation": "brief explanation of how this follows from the facts",
      "source_fact_paths": ["Character.Identity.Name", ...],
      "source_inference_ids": []
    }
  ],
  "violations": [
    {
      "type": "contradiction",
      "description": "what immutable fact was contradicted"
    }
  ],
  "decision_log": "One-sentence summary of why you chose this verdict."
}

Verdict definitions (evaluate in this priority order):
1. contradiction: ONLY for immutable Fact violations (fact already set) — HIGHEST PRIORITY
2. new_inference_logical: something strictly provable from Facts by pure logic
3. new_inference_probabilistic: a broad behavioural/personality tendency likely given the Facts
4. pass: ONLY when every specific claim in the response is a direct Fact or a strict derivation

Return only the JSON object, no other text."""
    )

    return "\n".join(parts)


async def run_evaluator(
    character: Character,
    facts_blob: dict[str, Any],
    user_message: str,
    character_response: str,
    ollama: OllamaClient,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> EvaluatorResult:
    """Run the evaluator LLM and return a parsed verdict."""
    prompt = build_evaluator_prompt(
        character,
        facts_blob,
        user_message,
        character_response,
        contradiction_hints,
        inferences,
        experiences,
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
        # Strip markdown code fences that some models emit despite being told not to
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.split("\n")
            start = 1
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            stripped = "\n".join(lines[start:end]).strip()
        data: dict[str, Any] = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise EvaluatorParseError(f"Evaluator returned non-JSON content: {content!r}") from exc

    verdict = data.get("verdict")

    if verdict not in _VALID_VERDICTS:
        raise EvaluatorParseError(f"Unknown evaluator verdict: {verdict!r}")

    # Contradiction priority: if any violation has type "contradiction", force the verdict
    violations_raw: list[dict[str, Any]] = data.get("violations", []) or []
    if any(v.get("type") == "contradiction" for v in violations_raw):
        data["verdict"] = "contradiction"

    # Coerce source_inference_ids: drop any value that isn't an integer.
    for inf in data.get("new_inferences", []) or []:
        raw = inf.get("source_inference_ids", []) or []
        inf["source_inference_ids"] = [v for v in raw if isinstance(v, int)]

    try:
        result = EvaluatorResult.model_validate(data)
    except ValidationError as exc:
        raise EvaluatorParseError(f"Failed to validate evaluator result: {exc}") from exc

    return result
