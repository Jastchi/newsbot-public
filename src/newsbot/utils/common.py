"""Utility functions."""

import logging
import logging.handlers
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

from newsbot.constants import LOG_FORMAT, LOG_LEVEL, TZ
from newsbot.llm_provider import get_required_env_vars
from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

_thread_local = threading.local()
# Tracks the active file handler per config key so it can be swapped
# without touching handlers belonging to other concurrent configs.
_file_handlers: dict[str, logging.handlers.TimedRotatingFileHandler] = {}
# Tracks error handlers per config key for the same reason.
_active_error_handlers: dict[str, list[logging.Handler]] = {}
# Our shared stream handler (added once; shared across all concurrent
# configs).
_stream_handler: logging.StreamHandler | None = None
_stream_handler_lock = threading.Lock()


def set_log_config_name(name: str) -> None:
    """
    Set the config label for the current thread's log records.

    Call this at the start of any thread that doesn't go through
    setup_logging (e.g. APScheduler worker threads).
    """
    _thread_local.config_name = name


class _ConfigNameFilter(logging.Filter):
    """Add config_name to log records from thread-local storage."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.__dict__["config_name"] = getattr(
            _thread_local,
            "config_name",
            "",
        )
        return True


class _ConfigFileFilter(logging.Filter):
    """
    Pass log records only when config_name matches this handler.

    Add this filter to a handler after _ConfigNameFilter so config_name
    is set on the record before the check runs.
    """

    def __init__(self, config_name: str) -> None:
        super().__init__()
        self._config_name = config_name

    def filter(self, record: logging.LogRecord) -> bool:
        return record.__dict__.get("config_name", "") == self._config_name


class TimezoneFormatter(logging.Formatter):
    """Log formatter using configured TZ (same as scheduler)."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
    ) -> None:
        """Initialize formatter; timestamps use constants.TZ."""
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.converter = self._timetuple_in_tz

    @staticmethod
    def _timetuple_in_tz(sec: float | None) -> time.struct_time:
        """Struct_time in configured TZ for log timestamps."""
        if sec is None:
            sec = time.time()
        return datetime.fromtimestamp(sec, tz=TZ).timetuple()


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

    # Only parse HTML when the text actually contains tags
    if "<" in text:
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
        error_handlers: List of logging handlers for error handling
        config_key: Config key (e.g., "technology") used
            to derive the log filename (e.g., "logs/technology.log")

    """
    config_name = config.name or config_key

    # Store in thread-local so concurrent jobs don't overwrite each
    # other's label — _ConfigNameFilter reads this at emit time.
    _thread_local.config_name = config_name

    log_file = f"logs/{config_key}.log" if config_key else "logs/newsbot.log"
    Path("logs").mkdir(parents=True, exist_ok=True)

    tz_formatter = TimezoneFormatter(LOG_FORMAT)
    config_name_filter = _ConfigNameFilter()
    config_file_filter = _ConfigFileFilter(config_name)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL))

    # Stream handler is shared across all configs — add it only once.
    # Replace any foreign plain StreamHandler (e.g. Django's "console"
    # handler added to the root logger via settings.LOGGING) so that
    # _ConfigNameFilter is always present and %(config_name)s is
    # populated.
    global _stream_handler
    with _stream_handler_lock:
        if _stream_handler is None or _stream_handler not in root.handlers:
            for h in list(root.handlers):
                if type(h) is logging.StreamHandler:
                    root.removeHandler(h)
            _stream_handler = logging.StreamHandler()
            _stream_handler.setFormatter(tz_formatter)
            _stream_handler.addFilter(config_name_filter)
            root.addHandler(_stream_handler)

    # File handler is per-config. Replace this config's handler without
    # touching other configs' handlers (no force=True / full reset).
    old_fh = _file_handlers.get(config_key)
    if old_fh is not None:
        root.removeHandler(old_fh)
        old_fh.close()

    rotating_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        utc=False,
        encoding="utf-8",
    )
    rotating_handler.setFormatter(tz_formatter)
    # _ConfigNameFilter must be first so config_name is set before
    # _ConfigFileFilter checks it.
    rotating_handler.addFilter(config_name_filter)
    rotating_handler.addFilter(config_file_filter)
    root.addHandler(rotating_handler)
    _file_handlers[config_key] = rotating_handler

    # Error handlers are per-job. Remove the previous ones for this
    # config key and attach the new ones.
    for old_eh in _active_error_handlers.get(config_key, []):
        root.removeHandler(old_eh)

    active: list[logging.Handler] = []
    for handler in error_handlers:
        handler.setFormatter(tz_formatter)
        handler.addFilter(config_name_filter)
        handler.addFilter(config_file_filter)
        root.addHandler(handler)
        active.append(handler)
    _active_error_handlers[config_key] = active

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

    missing_vars = [
        var for var in get_required_env_vars(provider)
        if not os.environ.get(var)
    ]

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
