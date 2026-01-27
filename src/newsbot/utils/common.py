"""Utility functions."""

import logging
import logging.handlers
import os
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup

from utilities.models import ConfigModel

logger = logging.getLogger(__name__)


def clean_text(text: str, config_name: str | None = None) -> str:
    """
    Clean and normalize text.

    Args:
        text: Raw text
        config_name: Optional config name for config-specific cleaning
            rules

    Returns:
        Cleaned text

    """
    if not text:
        return ""

    # Remove HTML tags if present
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', "", text)

    # Config-specific cleaning: Remove "premium" for Oesterreich
    # Die Presse uses the word for articles behind a paywall
    if config_name and config_name.lower() == "oesterreich":
        text = re.sub(r"\bpremium\b", "", text, flags=re.IGNORECASE)
        # Clean up any double spaces left after removing premium
        text = re.sub(r"\s+", " ", text)

    return text.strip()


def setup_logging(
    config: ConfigModel,
    error_handlers: list[logging.Handler],
    config_key: str = "",
) -> None:
    """
    Set up logging configuration.

    Args:
        config: Configuration dictionary
        error_handlers: List of logging handlers for error reporting
        config_key: Config key (e.g., "technology") used
            to derive the log filename (e.g., "logs/technology.log")

    """
    log_config = config.logging
    level = log_config.level
    log_format = log_config.format

    # Derive log filename from config key
    # e.g., "technology" → "logs/technology.log"
    log_file = f"logs/{config_key}.log" if config_key else "logs/newsbot.log"

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(parents=True, exist_ok=True)

    # Create TimedRotatingFileHandler with midnight rotation and 30-day
    # retention
    rotating_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        utc=False,
        encoding="utf-8",
    )
    rotating_handler.setFormatter(logging.Formatter(log_format))

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level),
        format=log_format,
        handlers=[
            rotating_handler,
            logging.StreamHandler(),
            *error_handlers,
        ],
    )

    logger.info("Logging initialized")


def validate_environment(
    config: ConfigModel,
    email_error_handler: logging.Handler | None = None,
) -> None:
    """
    Validate required environment variables based on configuration.

    Checks the config to determine which provider will be used,
    then validates only the environment variables needed for that
    provider.

    Raises SystemExit with clear error message if validation fails.
    Flushes email error handler before exiting if provided.

    Args:
        config: Configuration dictionary
        email_error_handler: Optional email error handler to flush
            before exiting

    """
    llm_config = config.llm
    provider = llm_config.provider

    missing_vars: list[str] = []

    # Validate provider-specific environment variables
    if provider == "gemini" and not os.environ.get("GEMINI_API_KEY"):
        missing_vars.append("GEMINI_API_KEY")

    if missing_vars:
        error_msg = (
            f"\n{'=' * 70}\n"
            "ERROR: Missing required environment variables for provider "
            f"'{provider}':\n"
            f"  {', '.join(missing_vars)}\n\n"
            f"Configuration specifies provider: {provider}\n"
            f"Please set these variables in your .env file or environment.\n"
            f"Example .env entry:\n"
            f"  {missing_vars[0]}=your-api-key-here\n"
            f"{'=' * 70}\n"
        )
        print(error_msg)
        logger.error(error_msg)

        # Flush email error handler before exiting
        if email_error_handler and hasattr(email_error_handler, "flush"):
            try:
                email_error_handler.flush()
            except (OSError, AttributeError) as e:
                # Don't let email flush failure prevent clean exit
                logger.warning(f"Failed to flush email error handler: {e}")

        # Use exit code 0 for configuration errors to prevent process
        # from restarting.
        sys.exit(0)
