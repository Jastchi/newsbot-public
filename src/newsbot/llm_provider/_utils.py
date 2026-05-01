"""Shared utilities for LLM providers."""

from datetime import UTC, datetime


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def with_date_context(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Prepend today's date as a system message."""
    date_str = f"Today's date is {_today()}."
    if messages and messages[0].get("role") == "system":
        combined = f"{date_str} {messages[0]['content']}"
        return [{"role": "system", "content": combined}, *messages[1:]]
    return [{"role": "system", "content": date_str}, *messages]


def with_date_prompt(prompt: str) -> str:
    """Prepend today's date to a raw prompt string."""
    return f"Today's date is {_today()}.\n\n{prompt}"
