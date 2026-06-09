"""Inference generation, revalidation, and cascade service."""

from __future__ import annotations

import json
import os

import aiosqlite

from memories.database import (
    create_inference,
    get_character,
    get_facts,
    get_inferences,
    update_inference_status,
)
from memories.models import Character, Fact, Inference
from memories.services.ollama_client import OllamaClient

MAX_INFERENCE_DEPTH: int = int(os.getenv("MAX_INFERENCE_DEPTH", "5"))
MAX_INFERENCE_BREADTH: int = int(os.getenv("MAX_INFERENCE_BREADTH", "5"))


class InferenceParseError(Exception):
    """Raised when the eager-pass LLM returns unparseable output."""


def compute_depth(
    source_inference_ids: list[int],
    known_inferences: list[Inference],
) -> int:
    """Compute depth for a new inference from its source inference ids.

    Returns 1 when source_inference_ids is empty (derived from Facts only).
    Otherwise returns max(depth of matched sources) + 1.  Unknown ids are
    silently skipped; if no valid source is found, returns 1 as a fallback.
    """
    if not source_inference_ids:
        return 1
    valid_depths = [inf.depth for inf in known_inferences if inf.id in source_inference_ids]
    if not valid_depths:
        return 1
    return max(valid_depths) + 1


def build_eager_pass_prompt(
    character: Character,
    facts: list[Fact],
    existing_inferences: list[Inference],
    max_breadth: int,
    max_depth: int = MAX_INFERENCE_DEPTH,
) -> str:
    lines: list[str] = [f"Character: {character.name}", ""]

    lines.append("## Current Facts (id: key: value)")
    for f in facts:
        lines.append(f"[{f.id}] {f.key}: {f.value}")

    lines.append("")
    lines.append("## Already Established Inferences (do NOT re-derive these)")
    if existing_inferences:
        for inf in existing_inferences:
            lines.append(f"[{inf.id}] {inf.statement}  (from: {inf.derivation})")
    else:
        lines.append("(none established yet)")

    lines.extend(
        [
            "",
            "## Your Task",
            f"You are a logical reasoner. Derive up to {max_breadth} NEW conclusions from the",
            "Facts above that are not already listed in Established Inferences.",
            "",
            "Rules:",
            "- LOGICAL: only derive what is certain — a strict logical consequence of one or",
            "  more Facts (e.g. birth year from age + current year).",
            "- PROBABILISTIC: a well-founded tendency or likelihood given the Facts, not a",
            "  specific invented detail.",
            "- Do NOT re-derive anything already in the Established Inferences list.",
            "- Cite source Facts and Inferences by id.",
            "- Cross-references within this same response are NOT allowed — only cite Facts",
            "  and Inferences already established before this pass.",
            f"- Aim for depth {max_depth} or fewer hops from root Facts.",
            "",
            "Return a JSON array (empty array if nothing new to derive):",
            "[",
            "  {",
            '    "inference_type": "logical | probabilistic",',
            '    "statement": "...",',
            '    "derivation": "brief explanation of how this follows",',
            '    "source_fact_ids": [int, ...],',
            '    "source_inference_ids": [int, ...]',
            "  }",
            "]",
            "",
            "Return only the JSON array, no other text.",
        ]
    )

    return "\n".join(lines)


def _strip_fences(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.split("\n")
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        stripped = "\n".join(lines[start:end]).strip()
    return stripped


def _coerce_id_list(raw: object) -> list[int]:
    if not isinstance(raw, list):
        return []
    return [v for v in raw if isinstance(v, int)]


async def run_eager_pass(
    db: aiosqlite.Connection,
    character: Character,
    facts: list[Fact],
    existing_inferences: list[Inference],
    ollama: OllamaClient,
    max_depth: int = MAX_INFERENCE_DEPTH,
    max_breadth: int = MAX_INFERENCE_BREADTH,
) -> list[Inference]:
    model = character.current_model_name or character.modelfile_base
    system_msg = (
        "You are a logical reasoning assistant. "
        "Return only a JSON array as instructed. No other text."
    )
    user_msg = build_eager_pass_prompt(
        character, facts, existing_inferences, max_breadth, max_depth
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    content, _ = await ollama.chat(model, messages, think=False, format="json")

    try:
        parsed = json.loads(_strip_fences(content))
    except (json.JSONDecodeError, ValueError) as exc:
        raise InferenceParseError(f"Eager pass returned non-JSON: {content!r}") from exc

    if not isinstance(parsed, list):
        raise InferenceParseError(f"Eager pass returned non-array JSON: {parsed!r}")

    existing_ids = {inf.id for inf in existing_inferences}
    results: list[Inference] = []

    for item in parsed:
        if not isinstance(item, dict):
            continue

        src_fact_ids = _coerce_id_list(item.get("source_fact_ids", []))
        src_inf_ids = _coerce_id_list(item.get("source_inference_ids", []))

        # Reject same-pass cross-references
        if any(sid not in existing_ids for sid in src_inf_ids):
            continue

        depth = compute_depth(src_inf_ids, existing_inferences)
        if depth > max_depth:
            continue

        if len(results) >= max_breadth:
            break

        stored = await create_inference(
            db,
            character_id=character.id,
            statement=str(item.get("statement", "")),
            derivation=str(item.get("derivation", "")),
            source_fact_ids=src_fact_ids,
            source_inference_ids=src_inf_ids,
            inference_type=str(item.get("inference_type", "logical")),
            depth=depth,
        )
        results.append(stored)

    return results


def build_revalidation_prompt(
    inference: Inference,
    facts: list[Fact],
    active_inferences: list[Inference],
) -> str:
    lines: list[str] = ["## Current Facts (id: key: value)"]
    for f in facts:
        lines.append(f"[{f.id}] {f.key}: {f.value}")

    lines.append("")
    lines.append("## Other Active Inferences (context only)")
    if active_inferences:
        for inf in active_inferences:
            lines.append(f"[{inf.id}] {inf.statement}")
    else:
        lines.append("(none)")

    lines.extend(
        [
            "",
            "## Inference to Revalidate",
            f'Statement: "{inference.statement}"',
            f'Original derivation: "{inference.derivation}"',
            f"Original sources: Facts {inference.source_fact_ids}, "
            f"Inferences {inference.source_inference_ids}",
            "",
            "## Your Task",
            "Given the CURRENT facts above, does this inference still hold?",
            "Return JSON exactly:",
            '{"holds": true | false, "reason": "one sentence"}',
        ]
    )

    return "\n".join(lines)


async def revalidate_single_inference(
    inference: Inference,
    facts: list[Fact],
    active_inferences: list[Inference],
    ollama: OllamaClient,
    model: str = "default",
) -> bool:
    prompt = build_revalidation_prompt(inference, facts, active_inferences)
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a logical fact-checker. Return only the JSON object as instructed."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        content, _ = await ollama.chat(model, messages, think=False, format="json")
        data = json.loads(_strip_fences(content))
        # Accept both {"holds": false} and {"verdict": "stale"} as "does not hold"
        if data.get("verdict") == "stale":
            return False
        return bool(data.get("holds", True))
    except (json.JSONDecodeError, ValueError, KeyError):
        # Conservative: don't mark stale on parse ambiguity.
        # OllamaConnectionError / OllamaResponseError propagate upward.
        return True


async def cascade_on_fact_edit(
    db: aiosqlite.Connection,
    character_id: int,
    changed_fact_id: int,
    ollama: OllamaClient,
) -> list[Inference]:
    """Mark inferences stale when a fact changes. Returns all newly-stale inferences."""
    character = await get_character(db, character_id)
    model = (character.current_model_name or character.modelfile_base) if character else "default"

    # Include stale inferences in the cascade so transitive chains through a
    # pre-existing stale intermediary are propagated correctly (Bug 6 fix).
    # Already-stale inferences are used as propagation seeds without re-calling
    # the LLM; only active inferences are revalidated.
    _all = await get_inferences(db, character_id, status="all")
    non_invalidated = [inf for inf in _all if inf.status != "invalidated"]
    active = [inf for inf in non_invalidated if inf.status == "active"]
    facts = await get_facts(db, character_id)

    newly_stale: set[int] = set()
    processed: set[int] = set()

    # Seed with all non-invalidated inferences that directly depend on the changed fact
    worklist = [inf for inf in non_invalidated if changed_fact_id in inf.source_fact_ids]

    while worklist:
        inference = worklist.pop(0)
        if inference.id in processed:
            continue
        processed.add(inference.id)

        should_propagate = False

        if inference.status == "stale":
            # Already stale: propagate to downstream inferences without an LLM call
            should_propagate = True
        else:
            # Active: revalidate and mark stale if it no longer holds
            remaining_active = [
                i for i in active if i.id not in newly_stale and i.id != inference.id
            ]
            holds = await revalidate_single_inference(
                inference, facts, remaining_active, ollama, model=model
            )
            if not holds:
                await update_inference_status(db, inference.id, "stale")
                newly_stale.add(inference.id)
                should_propagate = True

        if should_propagate:
            for other in non_invalidated:
                if other.id not in processed and inference.id in other.source_inference_ids:
                    worklist.append(other)

    # Re-fetch stale inferences to return current DB state
    all_stale = await get_inferences(db, character_id, status="stale")
    return [s for s in all_stale if s.id in newly_stale]


async def cascade_on_fact_delete(
    db: aiosqlite.Connection,
    character_id: int,
    deleted_fact_id: int,
) -> list[Inference]:
    """Invalidate all inferences that depend on a deleted fact (pure DB, no LLM)."""
    active = await get_inferences(db, character_id, status="active")

    to_invalidate: set[int] = {inf.id for inf in active if deleted_fact_id in inf.source_fact_ids}

    # Iteratively expand through transitive inference dependencies
    changed = True
    while changed:
        changed = False
        for inf in active:
            if inf.id not in to_invalidate and any(
                sid in to_invalidate for sid in inf.source_inference_ids
            ):
                to_invalidate.add(inf.id)
                changed = True

    for inf_id in to_invalidate:
        await update_inference_status(db, inf_id, "invalidated")

    all_invalidated = await get_inferences(db, character_id, status="invalidated")
    return [i for i in all_invalidated if i.id in to_invalidate]


__all__ = [
    "MAX_INFERENCE_BREADTH",
    "MAX_INFERENCE_DEPTH",
    "InferenceParseError",
    "build_eager_pass_prompt",
    "build_revalidation_prompt",
    "cascade_on_fact_delete",
    "cascade_on_fact_edit",
    "compute_depth",
    "revalidate_single_inference",
    "run_eager_pass",
]
