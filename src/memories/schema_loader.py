from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_schema: dict[str, Any] | None = None
_SCHEMA_PATH = Path(__file__).parent / "fact_schema.json"


def _reset_schema_cache() -> None:
    global _schema
    _schema = None


def load_schema() -> dict[str, Any]:
    global _schema
    if _schema is None:
        _schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return _schema


def apply_mask(
    blob: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if schema is None:
        schema = load_schema()
    return _mask_node(blob, schema)


def _mask_node(
    blob_node: dict[str, Any],
    schema_node: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, blob_value in blob_node.items():
        if key not in schema_node:
            continue
        schema_child = schema_node[key]
        if "Type" in schema_child:
            result[key] = blob_value
        else:
            if isinstance(blob_value, dict):
                masked = _mask_node(blob_value, schema_child)
                if masked:
                    result[key] = masked
    return result


def check_write_permitted(
    path: str,
    schema: dict[str, Any] | None = None,
) -> str:
    if schema is None:
        schema = load_schema()
    parts = path.split(".")
    node: Any = schema
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(
                f"Unknown schema path: {path!r}. "
                "You may only write to paths listed in the Fact Schema."
            )
        node = node[part]
    if "Type" not in node:
        raise ValueError(
            f"Path {path!r} is a grouping, not a writable leaf. "
            "Specify a full path to a leaf (e.g. 'Character.Identity.Name')."
        )
    return str(node["Mutability"])


def render_schema_for_prompt(schema: dict[str, Any] | None = None) -> str:
    if schema is None:
        schema = load_schema()

    leaves = _collect_leaves(schema)

    by_mutability: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "Immutable": [],
        "Mutable": [],
        "Fluid": [],
    }
    for path, leaf in leaves:
        bucket = leaf["Mutability"]
        by_mutability[bucket].append((path, leaf))

    lines = [
        "## Fact Schema",
        "You may UPDATE values for paths listed below.",
        "You may NOT create new paths.",
    ]

    labels = {
        "Immutable": "IMMUTABLE (cannot be changed once set)",
        "Mutable": "MUTABLE (contextually appropriate; surfaced to user for approval)",
        "Fluid": "FLUID (applied silently; no approval needed)",
    }

    for mutability in ("Immutable", "Mutable", "Fluid"):
        group = by_mutability[mutability]
        if not group:
            continue
        lines.append("")
        lines.append(f"{labels[mutability]}:")
        for path, leaf in group:
            if leaf["Type"] == "Enum":
                suffix = " | ".join(leaf["Constraint"])
            else:
                suffix = leaf["Description"]
            lines.append(f"  {path} — {suffix}")

    return "\n".join(lines)


def _collect_leaves(
    node: dict[str, Any],
    prefix: str = "",
) -> list[tuple[str, dict[str, Any]]]:
    leaves: list[tuple[str, dict[str, Any]]] = []
    for key, child in node.items():
        path = f"{prefix}.{key}" if prefix else key
        if "Type" in child:
            leaves.append((path, child))
        else:
            leaves.extend(_collect_leaves(child, path))
    return leaves
