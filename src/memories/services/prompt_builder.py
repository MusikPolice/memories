"""System prompt construction for the character LLM call."""

from memories.models import Character, Fact, Inference


def build_system_prompt(
    character: Character,
    facts: list[Fact],
    inferences: list[Inference] | None = None,
) -> str:
    lines: list[str] = [
        f"You are {character.name}. Stay in character at all times.",
        "",
        "## Your Facts",
        "These are established truths about you. Never contradict them and never invent",
        "details that are not listed here.",
        "",
    ]
    if facts:
        for fact in facts:
            lines.append(f"{fact.key}: {fact.value}")
    else:
        lines.append("No facts have been established yet. Do not invent biographical details.")

    if inferences:
        lines.append("")
        lines.append("## Your Inferences")
        lines.append(
            "These conclusions have been derived from your Facts. They are as reliable as the"
        )
        lines.append("Facts they came from. Do not contradict them.")
        lines.append("")
        for inf in inferences:
            lines.append(f"{inf.statement} (from: {inf.derivation})")

    return "\n".join(lines)
