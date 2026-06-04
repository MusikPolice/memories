"""System prompt construction for the character LLM call."""

from memories.models import Character, Fact


def build_system_prompt(character: Character, facts: list[Fact]) -> str:
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
    return "\n".join(lines)
