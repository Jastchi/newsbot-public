"""
Lazy imports for Django models that ensure setup_django is called first.

This module provides a clean way to import Django models without
triggering ruff's banned-module-level-imports rule. The setup_django()
call happens automatically when this module is imported.
"""

# Import directly to avoid circular import with utilities/__init__.py
import utilities.django_setup

# Now import and re-export the models
from web.newsserver.models import (
    AnalysisSummary,
    Article,
    NewsConfig,
    NewsSource,
    ScrapeSummary,
    Subscriber,
    Topic,
)

__all__ = [
    "AnalysisSummary",
    "Article",
    "NewsConfig",
    "NewsSource",
    "ScrapeSummary",
    "Subscriber",
    "Topic",
]
