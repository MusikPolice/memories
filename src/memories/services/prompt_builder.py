"""System prompt construction for the character LLM call."""

from memories.models import Character, Fact, Inference

_CATEGORY_ORDER = ["user", "character", "setting"]

_HEADERS = {
    "user": "## Facts About The User",
    "character": "## Your Facts (Character)",
    "setting": "## Setting Facts",
}

_DESCRIPTIONS = {
    "user": "These are established truths about the person you are talking with.",
    "character": (
        "These are established truths about you. Never contradict them and never invent\n"
        "details that are not listed here."
    ),
    "setting": "These are established truths about the current environment.",
}


def _fact_line(fact: Fact) -> str:
    if fact.mutability == "low":
        annotation = " [low-mutability — changes infrequently and with context]"
    elif fact.mutability == "high":
        annotation = " [fluid — may change within a session]"
    else:
        annotation = ""
    return f"{fact.key}: {fact.value}{annotation}"


def build_system_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference] | None = None,
) -> str:
    lines: list[str] = [
        f"You are {character.name}. Stay in character at all times.",
        "",
    ]

    if not facts:
        lines.append("## Your Facts")
        lines.append("No facts have been established yet. Do not invent biographical details.")
    else:
        by_category: dict[str, list[Fact]] = {cat: [] for cat in _CATEGORY_ORDER}
        for fact in sorted(facts, key=lambda f: f.id):
            bucket = fact.category if fact.category in by_category else "character"
            by_category[bucket].append(fact)

        for cat in _CATEGORY_ORDER:
            cat_facts = by_category[cat]
            if not cat_facts:
                continue
            lines.append(_HEADERS[cat])
            lines.append(_DESCRIPTIONS[cat])
            lines.append("")
            for fact in cat_facts:
                lines.append(_fact_line(fact))
            lines.append("")

    if inferences:
        lines.append("## Your Inferences")
        lines.append(
            "These conclusions have been derived from your Facts. They are as reliable as the"
        )
        lines.append("Facts they came from. Do not contradict them.")
        lines.append("")
        for inf in inferences:
            lines.append(f"{inf.statement} (from: {inf.derivation})")

    return "\n".join(lines)
