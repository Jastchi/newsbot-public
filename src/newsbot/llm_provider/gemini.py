"""
Gemini LLM provider implementation.

Uses Google Gemini API.
Requires optional dependency: uv sync --group gemini
"""

import json
import logging
import os
from typing import TYPE_CHECKING, Any, cast

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from utilities.models import ConfigModel

if TYPE_CHECKING:
    from tenacity._utils import LoggerProtocol

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3-flash"

# Retry configuration for Gemini transient errors (429/503)
GEMINI_MAX_RETRIES = 12
GEMINI_BACKOFF_MIN = 60  # seconds
GEMINI_BACKOFF_MAX = 600  # seconds (10 min) to wait out longer outages
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVICE_UNAVAILABLE = 503

REQUIRED_ENV_VARS = ["GEMINI_API_KEY"]


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
    before_sleep=before_sleep_log(
        cast("LoggerProtocol", logger), logging.WARNING,
    ),
    reraise=True,
)


def _get_options(
    self: object,
    options: dict[str, Any],
) -> tuple[float, int]:
    """Extract temperature and max_tokens from options."""
    temperature = options.get(
        "temperature", getattr(self, "temperature", 0.7),
    )
    max_tokens = options.get(
        "num_predict", getattr(self, "max_tokens", 1024),
    )
    return temperature, max_tokens


class GeminiProvider:
    """LLM provider implementation using Google Gemini API."""

    def _get_options(self, options: dict[str, Any]) -> tuple[float, int]:
        return _get_options(self, options)

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
            msg = (
                "google-genai package not installed. Install with: "
                "uv sync --group gemini"
            )
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
