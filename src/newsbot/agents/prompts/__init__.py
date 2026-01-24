"""
Prompt loader for LLM provider-specific prompts.

Loads prompt templates from provider-specific subdirectories (llama/,
gemini/).
Falls back to llama/ prompts if a provider-specific prompt is not found.

get_prompt(provider: str, filename: str) -> str: Load a prompt template.

- Provider names: "ollama" maps to llama/, "gemini" maps to gemini/.
- Prompts are cached for performance.
"""

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Mapping from provider name to prompt directory
PROVIDER_TO_DIR = {
    "ollama": "llama",
    "gemini": "gemini",
}

# Default directory when provider-specific prompt is not found
DEFAULT_DIR = "llama"


@lru_cache(maxsize=128)
def get_prompt(provider: str, filename: str) -> str:
    """
    Load a prompt template for the specified provider.

    Args:
        provider: LLM provider name (ollama, gemini).
        filename: Prompt template filename.

    Returns:
        Prompt template content as a string.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist in either
            provider-specific or default directory.

    """
    prompts_dir = Path(__file__).parent
    provider_dir = PROVIDER_TO_DIR.get(provider, DEFAULT_DIR)

    # Try provider-specific prompt first
    provider_path = prompts_dir / provider_dir / filename
    if provider_path.exists():
        logger.debug(f"Loading prompt from {provider_dir}/{filename}")
        return provider_path.read_text(encoding="utf-8")

    # Fall back to default directory
    if provider_dir != DEFAULT_DIR:
        default_path = prompts_dir / DEFAULT_DIR / filename
        if default_path.exists():
            logger.debug(
                f"Prompt {filename} not found for {provider}, "
                f"falling back to {DEFAULT_DIR}",
            )
            return default_path.read_text(encoding="utf-8")

    msg = f"Prompt file {filename} not found for provider {provider}"
    raise FileNotFoundError(msg)


def clear_prompt_cache() -> None:
    """Clear the prompt cache. Useful for testing."""
    get_prompt.cache_clear()

