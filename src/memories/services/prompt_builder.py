"""System prompt construction for the character LLM call."""

from __future__ import annotations

from typing import Any

from memories.models import Character, Experience, Inference
from memories.schema_loader import load_schema

_WORLD_STATE_PREAMBLE = """\
## World State

The following describes everything known about the world right now.
Facts are organised by category and grouped by topic.

Mutability levels:
- IMMUTABLE: fixed permanently once set. You may not invent or change these.
  If a value is missing and you need it, call require_fact() rather than making one up.
- MUTABLE: stable but can change with narrative context. If your response implies a
  change, the evaluator will surface it to the user for approval.
- FLUID: expected to change freely within a session. Update via the evaluator naturally.

For ENUM facts, only the listed values are valid."""


def _lookup_blob_value(
    blob: dict[str, Any],
    path_parts: list[str],
) -> str | int | None:
    """Walk the blob following path_parts; return the leaf Value or None if absent."""
    node: Any = blob
    for part in path_parts:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if not isinstance(node, dict):
        return None
    return node.get("Value")


def _render_schema_node(
    schema_node: dict[str, Any],
    blob_node: dict[str, Any],
    prefix: str,
    lines: list[str],
) -> None:
    """Recursively walk schema_node, rendering leaves into lines."""
    for key, child in schema_node.items():
        path = f"{prefix}.{key}" if prefix else key
        if "Type" in child:
            mutability = child["Mutability"].upper()
            description = child["Description"]
            blob_leaf = blob_node.get(key, {}) if isinstance(blob_node, dict) else {}
            value = blob_leaf.get("Value") if isinstance(blob_leaf, dict) else None

            lines.append(f"[{mutability}] {path} — {description}")
            if value is None:
                lines.append("  Value: (not set)")
            else:
                lines.append(f"  Value: {value}")
            if child["Type"] == "Enum":
                constraint = " | ".join(child["Constraint"])
                lines.append(f"  Valid values: {constraint}")
            lines.append("")
        else:
            blob_child = blob_node.get(key, {}) if isinstance(blob_node, dict) else {}
            _render_schema_node(child, blob_child, path, lines)


def build_system_prompt(
    character: Character,
    facts_blob: dict[str, Any],
    inferences: list[Inference] | None = None,
    experiences: list[Experience] | None = None,
) -> str:
    lines: list[str] = [
        f"You are {character.name}. Stay in character at all times.",
        "",
        _WORLD_STATE_PREAMBLE,
        "",
    ]

    schema = load_schema()
    for section_key in schema:
        lines.append(f"### {section_key}")
        lines.append("")
        section_blob = facts_blob.get(section_key, {})
        _render_schema_node(schema[section_key], section_blob, section_key, lines)

    if inferences:
        lines.append("## Your Inferences")
        lines.append(
            "These conclusions have been derived from your Facts. They are as reliable as the"
        )
        lines.append("Facts they came from. Do not contradict them.")
        lines.append("")
        for inf in inferences:
            lines.append(f"{inf.statement} (from: {inf.derivation})")

    if experiences:
        lines.append("")
        lines.append("## Your Experiences")
        lines.append("These are things you have learned or observed through past conversations.")
        lines.append("They are working memory — you may reference them freely.")
        lines.append("")
        for exp in experiences:
            source_label = "told by user" if exp.source == "told_by_user" else "observed"
            lines.append(f"[{source_label}] {exp.statement}")

    return "\n".join(lines)
