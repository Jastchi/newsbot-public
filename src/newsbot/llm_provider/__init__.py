"""
LLM Provider abstraction layer.

Provides a unified interface for LLM providers (Ollama, Gemini).
The provider is selected via the `llm.provider` config field.

LLMProvider: Protocol defining the LLM interface.
OllamaProvider: Implementation using local Ollama service.
GeminiProvider: Google Gemini API (optional dependency).
get_llm_provider(config): Factory function.
get_required_env_vars(provider_name): Returns required env vars.

- Ollama requires a running local Ollama service.
- Gemini requires GEMINI_API_KEY and: uv sync --group gemini
- The provider field in config determines which is used.
- Gemini API calls retry on rate limit errors with exponential backoff.

To add a new provider:
1. Create llm_provider/openai.py implementing the protocol.
2. Define REQUIRED_ENV_VARS and register via register_provider().
3. Add optional dependency group in pyproject.toml if needed.
"""

import logging
from collections.abc import Callable
from typing import Any, Protocol

from newsbot.llm_provider.gemini import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_MAX_RETRIES,
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_TOO_MANY_REQUESTS,
    GeminiProvider,
    _is_gemini_retryable_error,
)
from newsbot.llm_provider.gemini import (
    REQUIRED_ENV_VARS as GEMINI_ENV_VARS,
)
from newsbot.llm_provider.ollama import REQUIRED_ENV_VARS as OLLAMA_ENV_VARS
from newsbot.llm_provider.ollama import OllamaProvider
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

# Registry: provider_name -> (factory, required_env_vars)
# Factory receives ConfigModel and returns an LLMProvider instance.
_PROVIDER_REGISTRY: dict[
    str,
    tuple[
        Callable[[ConfigModel], Any],
        list[str],
    ],
] = {}


def register_provider(
    name: str,
    factory: Callable[[ConfigModel], Any],
    required_env_vars: list[str],
) -> None:
    """Register an LLM provider."""
    _PROVIDER_REGISTRY[name] = (factory, required_env_vars)


def get_required_env_vars(provider_name: str) -> list[str]:
    """
    Get required environment variable names for a provider.

    Returns empty list if provider is unknown or has no required vars.
    """
    if provider_name in _PROVIDER_REGISTRY:
        return _PROVIDER_REGISTRY[provider_name][1]
    return []


class LLMProvider(Protocol):
    """Protocol for LLM provider interface with shared utilities."""

    def _get_options(self, options: dict[str, Any]) -> tuple[float, int]:
        """Extract temperature and max_tokens from options."""
        ...

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        """Generate text from a prompt."""
        ...

    def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> str:
        """Generate text from a chat conversation."""
        ...

    def chat_json(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate structured JSON from a chat conversation."""
        ...


def _gemini_factory(config: ConfigModel) -> GeminiProvider:
    """Create Gemini provider instance."""
    logger.info(
        "Using Gemini LLM provider (model: "
        f"{config.llm.model or DEFAULT_GEMINI_MODEL})",
    )
    return GeminiProvider(config)


# Register built-in providers
register_provider("ollama", OllamaProvider, OLLAMA_ENV_VARS)
register_provider("gemini", _gemini_factory, GEMINI_ENV_VARS)


def get_llm_provider(config: ConfigModel) -> LLMProvider:
    """
    Get the appropriate LLM provider based on configuration.

    Uses the provider registry. Unknown providers fall back to Ollama.

    Args:
        config: Configuration dictionary containing LLM settings.

    Returns:
        An LLM provider instance based on config.

    """
    provider_name = config.llm.provider

    if provider_name in _PROVIDER_REGISTRY:
        factory, _ = _PROVIDER_REGISTRY[provider_name]
        try:
            return factory(config)
        except ImportError:
            logger.exception(
                "Failed to load provider '%s'. "
                "You may need to install optional dependencies.",
                provider_name,
            )
            raise

    logger.warning(
        f"Unknown provider '{provider_name}', falling back to Ollama",
    )
    return OllamaProvider(config)


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "GEMINI_MAX_RETRIES",
    "HTTP_SERVICE_UNAVAILABLE",
    "HTTP_TOO_MANY_REQUESTS",
    "GeminiProvider",
    "LLMProvider",
    "OllamaProvider",
    "_is_gemini_retryable_error",
    "get_llm_provider",
    "get_required_env_vars",
    "register_provider",
]
