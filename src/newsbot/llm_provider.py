"""
LLM Provider abstraction layer.

Provides a unified interface for different LLM providers (Ollama,
Gemini). The provider is selected via the `llm.provider` config field.

LLMProvider: Protocol defining the LLM interface.
OllamaProvider: Implementation using local Ollama service.
GeminiProvider: Implementation using Google Gemini API.
get_llm_provider(config: Config) -> LLMProvider: Factory function.

- Ollama requires a running local Ollama service.
- Gemini requires GEMINI_API_KEY environment variable.
- The provider field in config determines which is used.
- Gemini API calls retry on rate limit errors with exponential backoff.
"""

import json
import logging
import os
import subprocess
import sys
import time
from typing import Any, Protocol

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

# Retry configuration for Gemini transient errors
GEMINI_MAX_RETRIES = 4
GEMINI_BACKOFF_MIN = 30  # seconds
GEMINI_BACKOFF_MAX = 120  # seconds
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503


def _is_gemini_retryable_error(exception: BaseException) -> bool:
    """Check if exception is a retryable Gemini error (429 or 503)."""
    try:
        from google.genai.errors import ClientError, ServerError
    except ImportError:
        return False

    # Retry on 429 (rate limit) ClientError
    if isinstance(exception, ClientError):
        return exception.code == HTTP_TOO_MANY_REQUESTS

    # Retry on 503 (service unavailable/overloaded) ServerError
    if isinstance(exception, ServerError):
        return exception.code == HTTP_SERVICE_UNAVAILABLE

    return False


# Retry decorator for Gemini API calls that handles rate limiting
# and service errors
_gemini_retry = retry(
    retry=retry_if_exception(_is_gemini_retryable_error),
    stop=stop_after_attempt(GEMINI_MAX_RETRIES),
    wait=wait_exponential(
        multiplier=1,
        min=GEMINI_BACKOFF_MIN,
        max=GEMINI_BACKOFF_MAX,
    ),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class LLMProvider(Protocol):
    """Protocol for LLM provider interface with shared utilities."""

    def _get_options(self, options: dict[str, Any]) -> tuple[float, int]:
        """
        Extract temperature and max_tokens from options.

        Args:
            options: Options dictionary.

        Returns:
            Tuple of (temperature, max_tokens).

        """
        temperature = options.get(
            "temperature", getattr(self, "temperature", 0.7),
        )
        max_tokens = options.get(
            "num_predict", getattr(self, "max_tokens", 1024),
        )
        return temperature, max_tokens

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: The input prompt.
            options: Provider-specific options.

        Returns:
            Generated text response.

        """
        ...

    def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> str:
        """
        Generate text from a chat conversation.

        Args:
            messages: List of message dicts with role/content.
            options: Provider-specific options.

        Returns:
            Generated text response.

        """
        ...

    def chat_json(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Generate structured JSON from a chat conversation.

        Args:
            messages: List of message dicts with role/content.
            options: Provider-specific options.
            schema: JSON schema for structured output.

        Returns:
            Parsed JSON response as a dictionary.

        """
        ...


class OllamaProvider(LLMProvider):
    """LLM provider implementation using local Ollama service."""

    def __init__(self, config: ConfigModel) -> None:
        """
        Initialize the Ollama provider.

        Args:
            config: Configuration dictionary containing LLM settings.

        """
        llm_config = config.llm
        self.model = llm_config.model or "llama2"
        self.base_url = llm_config.base_url or "http://localhost:11434"
        self.temperature = llm_config.temperature
        self.max_tokens = llm_config.max_tokens

        # Initialize Ollama client
        self._initialize_client()
        self._ensure_model_available()

        logger.info(
            f"OllamaProvider initialized: model={self.model}, "
            f"base_url={self.base_url}",
        )

    def _initialize_client(self) -> None:
        """Initialize the Ollama client, starting service if needed."""
        try:
            import ollama

            self.client = ollama.Client(host=self.base_url)
            # Test connection by listing models
            self.client.list()
            logger.info("Ollama client connected")
        except Exception:
            logger.warning("Ollama not running, attempting to start service")
            if not self._start_service():
                sys.exit("Ollama service could not be started. Exiting.")
            self.client = ollama.Client(host=self.base_url)

    def _start_service(self) -> bool:
        """Attempt to start the Ollama service."""
        try:
            import ollama

            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(5)  # Wait for Ollama to start
            ollama.Client(host=self.base_url).list()
            logger.info("Ollama started successfully")
        except Exception:
            logger.exception("Failed to start Ollama. Is it installed?")
            return False
        else:
            return True

    def _ensure_model_available(self) -> None:
        """Check if the model exists, and pull it if it doesn't."""
        try:
            import ollama

            models = self.client.list()
            model_names = [
                m.get("model", m.get("name", ""))
                for m in models.get("models", [])
            ]

            model_exists = any(
                self.model == name
                or self.model in name
                or name.startswith(self.model)
                for name in model_names
            )

            if not model_exists:
                logger.warning(
                    f"Model '{self.model}' not found. Pulling from Ollama...",
                )
                logger.info(
                    f"Downloading model '{self.model}' "
                    "(this may take a few minutes)...",
                )
                ollama.pull(self.model)
                logger.info(f"Successfully pulled model '{self.model}'")
                logger.info(f"Model '{self.model}' downloaded successfully!")
            else:
                logger.debug(f"Model '{self.model}' is available")

        except Exception:
            logger.exception(f"Error checking/pulling model '{self.model}'")
            msg = f"Failed to ensure model '{self.model}' is available."
            sys.exit(msg)

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        """Generate text from a prompt using Ollama."""
        import ollama

        temperature, max_tokens = self._get_options(options)

        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["response"].strip()

    def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> str:
        """Generate text from a chat conversation using Ollama."""
        temperature, max_tokens = self._get_options(options)

        response = self.client.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"].strip()

    def chat_json(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate structured JSON via chat using Ollama."""
        temperature, max_tokens = self._get_options(options)

        response = self.client.chat(
            model=self.model,
            messages=messages,
            format=schema,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        content = response["message"]["content"]
        return json.loads(content)


class GeminiProvider(LLMProvider):
    """LLM provider implementation using Google Gemini API."""

    def __init__(self, config: ConfigModel) -> None:
        """
        Initialize the Gemini provider.

        Args:
            config: Configuration dictionary containing LLM settings.

        """
        # Import here to avoid import errors when not using Gemini
        try:
            from google import genai
            from google.genai import types
        except ImportError as err:
            msg = "google-genai package not installed."
            raise ImportError(msg) from err

        llm_config = config.llm
        self.model_name = llm_config.model or DEFAULT_GEMINI_MODEL
        self.temperature = llm_config.temperature
        self.max_tokens = llm_config.max_tokens

        # Get API key from environment
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            msg = (
                "GEMINI_API_KEY environment variable not set. "
                "Please set it in your .env file or environment."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Create the client
        logger.info("Initializing Gemini API client...")
        self.client = genai.Client(api_key=api_key)
        self._types = types

        logger.info(
            f"GeminiProvider initialized: model={self.model_name}, "
            f"temperature={self.temperature}, max_tokens={self.max_tokens}",
        )

    @_gemini_retry
    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        """Generate text from a prompt using Gemini."""
        temperature, max_tokens = self._get_options(options)

        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        # Handle blocked or empty responses
        if not response.text:
            finish_reason = "unknown"
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            logger.warning(
                "Gemini returned empty response. Finish reason: %s",
                finish_reason,
            )
            return ""

        return response.text.strip()

    @_gemini_retry
    def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> str:
        """Generate text from a chat conversation using Gemini."""
        temperature, max_tokens = self._get_options(options)

        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        # Convert messages to Gemini format
        gemini_contents = self._convert_messages(messages)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config,
        )

        # Handle blocked or empty responses
        if not response.text:
            finish_reason = "unknown"
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            logger.warning(
                "Gemini returned empty response. Finish reason: %s",
                finish_reason,
            )
            return ""

        return response.text.strip()

    @_gemini_retry
    def chat_json(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate structured JSON via chat using Gemini."""
        temperature, max_tokens = self._get_options(options)

        config = self._types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json",
            response_schema=schema,
        )

        # Convert messages to Gemini format
        gemini_contents = self._convert_messages(messages)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=gemini_contents,
            config=config,
        )

        # Handle blocked or empty responses
        if not response.text:
            finish_reason = "unknown"
            if response.candidates:
                finish_reason = str(response.candidates[0].finish_reason)
            logger.warning(
                "Gemini JSON returned empty. Finish reason: %s",
                finish_reason,
            )
            return {}

        return json.loads(response.text)

    def _convert_messages(
        self,
        messages: list[dict[str, str]],
    ) -> list[Any]:
        """
        Convert OpenAI-style messages to Gemini Content format.

        Args:
            messages: List of messages with 'role' and 'content' keys.

        Returns:
            List of Gemini Content objects.

        """
        contents: list[Any] = []
        for msg in messages:
            role = msg["role"]
            # Gemini uses "user" and "model" roles
            if role == "assistant":
                role = "model"
            elif role == "system":
                # Gemini lacks system role; treat as user message
                role = "user"

            contents.append(
                self._types.Content(
                    role=role,
                    parts=[self._types.Part(text=msg["content"])],
                ),
            )
        return contents


def get_llm_provider(config: ConfigModel) -> LLMProvider:
    """
    Get the appropriate LLM provider based on configuration.

    Args:
        config: Configuration dictionary containing LLM settings.

    Returns:
        An LLM provider instance based on config.

    """
    provider_name = config.llm.provider

    if provider_name == "gemini":
        logger.info(
            "Using Gemini LLM provider (model: "
            f"{config.llm.model or DEFAULT_GEMINI_MODEL})",
        )
        return GeminiProvider(config)

    if provider_name == "ollama":
        logger.info("Using Ollama LLM provider")
        return OllamaProvider(config)

    logger.warning(
        f"Unknown provider '{provider_name}', falling back to Ollama",
    )
    return OllamaProvider(config)
