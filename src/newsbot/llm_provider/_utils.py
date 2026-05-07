"""Shared utilities for LLM providers."""

from datetime import UTC, datetime
from typing import Any


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def get_options(
    self: object,
    options: dict[str, Any],
) -> tuple[float, int]:
    """Extract temperature and max_tokens from options dict."""
    temperature = options.get(
        "temperature", getattr(self, "temperature", 0.7),
    )
    max_tokens = options.get(
        "num_predict", getattr(self, "max_tokens", 1024),
    )
    return temperature, max_tokens


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
