"""Shared utilities for newsbot."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utilities.django_models import NewsConfig
    from utilities.models import ConfigModel


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
    # Close any stale database connections before querying
    from django.db import close_old_connections

    close_old_connections()

    # Import here to avoid circular imports during Django setup
    from utilities.django_models import NewsConfig as NewsConfigModel

    try:
        news_config = NewsConfigModel.objects.get(key=config_key)
    except NewsConfigModel.DoesNotExist as e:
        msg = f"Config not found: {config_key}"
        raise ConfigNotFoundError(msg) from e

    config = news_config.to_config_dict()
    return config, news_config


__all__ = ["ConfigNotFoundError", "load_config", "setup_django"]
