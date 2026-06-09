"""Evaluator LLM service.

Runs a second Ollama call after each character response to check it against
established Facts.  Returns a structured verdict that chat_service uses to
decide whether to deliver, regenerate, or surface a notification.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, ValidationError

from memories.models import Character, Experience, Fact, Inference
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


class ExperienceUpdate(BaseModel):
    contradicted_experience_id: int
    description: str


class EvaluatorResult(BaseModel):
    verdict: str
    new_inferences: list[NewInference] = []
    violations: list[Violation] = []
    experience_updates: list[ExperienceUpdate] = []
    decision_log: str = ""
    contradiction_notifications: list[ContradictionNotification] = []
    max_retries_exceeded: bool = False


def _violation_duplicates_existing_fact(
    suggested_fact: dict[str, str] | None, facts: list[Fact]
) -> bool:
    """Return True when suggested_fact is already captured by an existing Fact.

    Catches two cases:
    - Exact match: the evaluator re-proposes a key+value that already exists verbatim.
    - Subset match: the suggested value is a component of a composite fact value
      (e.g. 'oak desk' ⊆ 'oak desk, ergonomic chair, couch').
    """
    if not suggested_fact:
        return False
    key = suggested_fact.get("key", "").strip().lower()
    value = suggested_fact.get("value", "").strip().lower()
    if not key or not value:
        return False
    for fact in facts:
        if fact.key.strip().lower() == key:
            existing = fact.value.strip().lower()
            if value == existing or value in existing:
                return True
    return False


def build_evaluator_prompt(
    character: Character,
    facts: list[Fact],
    user_message: str,
    character_response: str,
    contradiction_hints: list[str] | None = None,
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
    """Build the user-facing content for the evaluator Ollama call."""
    parts: list[str] = [f"Character: {character.name}"]

    parts.append("\n## Established Facts (id: key: value  (category, mutability))")
    if facts:
        for f in facts:
            parts.append(
                f"[{f.id}] {f.key}: {f.value}  (category: {f.category}, mutability: {f.mutability})"
            )
    else:
        parts.append("(no facts established yet)")

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
## Mutability Rules
These rules govern how you classify violations against established Facts.

MANDATORY FIRST STEP — before looking for new inferences, scan every Fact marked
`mutability: high` or `mutability: low`. For each one, ask: "Does the character's
response imply a value different from what this Fact currently states?" If yes for
any Fact, the verdict must be `implication`. Only proceed to `new_inference_*`
verdicts after completing this scan and finding no high/low-mutability Fact changes.

- IMMUTABLE facts: any response that contradicts an immutable Fact is a `contradiction`
  regardless of context. Do not surface these as implications — the value cannot change.
  Examples: height, birthdate, eye colour, bone structure.

- LOW-mutability facts: these change infrequently and only with clear narrative context
  (e.g., the character changed their clothes, moved to a new city). If the character's
  response implies a different value for a low-mutability Fact, return `implication` (not
  `contradiction`) — the change is plausible but needs user confirmation. Include a
  violation entry with the new implied value as `suggested_fact`.

- HIGH-mutability facts: these can change fluidly within a session (mood, emotional state,
  immediate desires, stress level). If the character's response reflects a different value
  for a high-mutability Fact, return `implication` — the change is expected and natural.
  Include a violation entry with the new implied value as `suggested_fact`. In the
  violation description, note that this is a high-mutability change: e.g.,
  "Stress level appears to have shifted from 'low' to 'high' (high-mutability fact)".

  CRITICAL: `new_inference_*` verdicts are NEVER valid when an existing high-mutability
  Fact already covers the same domain. If `stress_level: low (mutability: high)` exists
  and the character says "my stress is through the roof", that is `implication`, NOT
  `new_inference_probabilistic`. New inferences only apply to domains with no existing
  Fact at all.

When building a `suggested_fact`, always include a `category` field that reflects whose
fact it is:
- `"character"` — something about the character themselves (their own clothing, mood, etc.)
- `"user"` — something about the person they are talking with
- `"setting"` — something about the current environment or situation

If the category is unclear, default to `"character"`.

## Your Task
Analyze the character's response. Every specific claim must be TRACEABLE to an
established Fact or strictly derived from one.

CRITICAL DISTINCTION: "consistent with facts" is NOT the same as "grounded in facts."
A detail is grounded if it can be directly looked up in the **Established Facts** list,
found verbatim in the **Established Inferences** list, or is a necessary logical
consequence of the above. A detail is NOT automatically grounded just because it is
consistent with the facts or sounds plausible.
If the character INVENTED a specific detail — clothing, accessories, a hairstyle,
a location, a relationship, a personal history item — that is an IMPLICATION,
even if it seems plausible for this type of character.

The `new_inference_*` verdicts apply ONLY when the observation concerns a domain with
no existing Fact. If an existing Fact (any mutability) already covers that domain,
use `implication` (high/low mutability) or `contradiction` (immutable) instead.

The new_inference_logical / new_inference_probabilistic verdicts should only fire for
conclusions that are NOT already in the Established Inferences list AND have no
existing Fact covering the same domain.

Examples:
- Fact `mood: happy (mutability: high)` + character expresses anxiety → implication
  (NOT new_inference_probabilistic — the mood domain is already covered by a Fact)
- Fact `stress_level: low (mutability: high)` + character says "stress is through the roof"
  → implication (NOT new_inference_probabilistic)
- Character says "I enjoy organising things" (occupation=PA, no mood/preference Fact)
  → new_inference_probabilistic
- Character describes wearing a specific outfit not in the facts → implication
- Character states a birthplace not in the facts → implication
- Character says their eye colour contradicts the eye colour fact → contradiction
- Character says "I'm 26" and the age fact is 26 → pass

## Experience Contradiction Rules
If the character's response contradicts an Active Experience — implying the world has
changed in a way that invalidates it — return verdict `experience_update`:
- Unlike `contradiction`, the response IS delivered; no regeneration occurs.
- Include the contradicted Experience's id in `experience_updates[].contradicted_experience_id`.
- `experience_update` takes priority over `implication` but lower than `contradiction`.

Note: `contradiction` applies ONLY to immutable Facts. A character response that contradicts
an Experience is never a `contradiction` — it is always `experience_update`.

Return a JSON object with this exact structure:

{
  "verdict": "<contradiction|implication|new_inference_logical
    |new_inference_probabilistic|experience_update|pass>",
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
      "description": "what was wrong or what new fact was implied",
      "suggested_fact": {"key": "...", "value": "...", "category": "user|character|setting"} or null
    }
  ],
  "experience_updates": [
    {
      "contradicted_experience_id": 5,
      "description": "brief description of why this experience was contradicted"
    }
  ],
  "decision_log": "One-sentence summary of why you chose this verdict."
}

Verdict definitions (evaluate in this priority order):
1. contradiction: ONLY for immutable Fact violations — HIGHEST PRIORITY
2. implication: for low- or high-mutability Fact changes, or invented specific details
3. new_inference_logical: something strictly provable from Facts by pure logic
4. new_inference_probabilistic: a broad behavioural/personality tendency likely
   given the Facts but not a specific new assertion
5. pass: ONLY when every specific claim in the response is a direct Fact or a
   strict logical derivation — NOT merely "consistent with" or "plausible for"

BEFORE flagging any implication: check the Established Facts list again. If the
key AND value you are about to suggest already appear there — verbatim, or as a
component of a composite value — do NOT flag it. The character is correctly
referencing an established fact. Only surface implications for details that are
genuinely new.

pass is the LAST resort. When in doubt, prefer implication or new_inference_*.

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
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> EvaluatorResult:
    """Run the evaluator LLM and return a parsed verdict."""
    prompt = build_evaluator_prompt(
        character,
        facts,
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

    if "decision_log" not in data:
        raise EvaluatorParseError("Evaluator response missing required 'decision_log' field")

    # Contradiction priority: if any violation has type "contradiction", force the verdict
    violations_raw: list[dict[str, Any]] = data.get("violations", []) or []
    if any(v.get("type") == "contradiction" for v in violations_raw):
        data["verdict"] = "contradiction"

    # Coerce source_fact_ids / source_inference_ids: drop any value that isn't an integer.
    # Small models sometimes put "key: value" strings here instead of the numeric IDs.
    for inf in data.get("new_inferences", []) or []:
        for field in ("source_fact_ids", "source_inference_ids"):
            raw = inf.get(field, []) or []
            inf[field] = [v for v in raw if isinstance(v, int)]

    try:
        result = EvaluatorResult.model_validate(data)
    except ValidationError as exc:
        raise EvaluatorParseError(f"Failed to validate evaluator result: {exc}") from exc

    # Strip any implication violation whose suggested_fact is already an
    # established Fact (exact match or subset of a composite value).  The
    # evaluator LLM occasionally re-proposes existing facts, especially when
    # the character response merely repeats a detail that is already grounded.
    if result.verdict == "implication" and result.violations:
        filtered = [
            v
            for v in result.violations
            if not _violation_duplicates_existing_fact(v.suggested_fact, facts)
        ]
        if len(filtered) < len(result.violations):
            result.violations = filtered
            if not filtered:
                result.verdict = "pass"

    return result
