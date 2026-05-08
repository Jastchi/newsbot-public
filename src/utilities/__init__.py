"""Shared utilities for newsbot."""

from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from utilities.django_models import NewsConfig
    from utilities.models import ConfigModel

logger = logging.getLogger(__name__)

_TRUTHY_VALUES = frozenset({"true", "1", "yes"})


def is_truthy_env(key: str, default: str = "false") -> bool:
    """Return True when the given env var holds a truthy value."""
    return os.getenv(key, default).strip().lower() in _TRUTHY_VALUES


def is_dev_environment() -> bool:
    """Return True when ENVIRONMENT is set to 'dev'."""
    return os.getenv("ENVIRONMENT", "prod").strip().lower() == "dev"


class ConfigNotFoundError(Exception):
    """Raised when a config is not found in the database."""


def setup_django() -> None:
    """Set up Django for its ORM."""
    importlib.import_module("utilities.django_setup")


def load_config(config_key: str) -> tuple[ConfigModel, NewsConfig]:
    """
    Load configuration from database by key.

    Args:
        config_key: The unique config key (e.g., "test_technology")

    Returns:
        Tuple of (ConfigModel, NewsConfig)

    Raises:
        ConfigNotFoundError: If config with given key doesn't exist

    """
    from django.db import OperationalError, close_old_connections

    from utilities.django_models import NewsConfig as NewsConfigModel

    @retry(
        retry=retry_if_exception_type(OperationalError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _fetch() -> NewsConfig:
        close_old_connections()
        return NewsConfigModel.objects.get(key=config_key)

    try:
        news_config = _fetch()
    except NewsConfigModel.DoesNotExist as e:
        msg = f"Config not found: {config_key}"
        raise ConfigNotFoundError(msg) from e

    config = news_config.to_config_dict()
    return config, news_config


__all__ = [
    "ConfigNotFoundError",
    "is_dev_environment",
    "is_truthy_env",
    "load_config",
    "setup_django",
]
