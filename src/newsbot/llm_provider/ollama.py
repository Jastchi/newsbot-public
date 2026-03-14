"""
Ollama LLM provider implementation.

Uses local Ollama service. No optional dependencies; always available.
"""

import json
import logging
import subprocess
import sys
import time
from typing import Any

from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

# Ollama has no required env vars
REQUIRED_ENV_VARS: list[str] = []


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


class OllamaProvider:
    """LLM provider implementation using local Ollama service."""

    def _get_options(self, options: dict[str, Any]) -> tuple[float, int]:
        return _get_options(self, options)

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
