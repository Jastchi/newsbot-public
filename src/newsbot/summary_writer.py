"""Module for writing run summaries to the Django database."""

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime

from newsbot.agents.story_clustering_agent import Story
from utilities.django_models import (
    AnalysisSummary,
    NewsConfig,
    ScrapeSummary,
)

logger = logging.getLogger(__name__)


def _serialize_for_json(obj: object) -> object:
    """
    Recursively convert datetime objects to ISO format strings.

    Args:
        obj: Object to serialize (dict, list, datetime, or primitive).

    Returns:
        JSON-serializable version of the object.

    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {key: _serialize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    return obj


class SummaryWriter:
    """Writes scrape and analysis summaries to the Django database."""

    def __init__(self) -> None:
        """Initialize the summary writer and ensure Django is set up."""
        # Django is automatically set up when django_models is imported

    def _get_config(self, config_key: str) -> object:
        """
        Get the NewsConfig instance with the given key.

        Args:
            config_key: The key of the news configuration.

        Returns:
            NewsConfig instance or None if not found.

        """
        try:
            return NewsConfig.objects.filter(key=config_key).first()
        except Exception:
            logger.exception("Error getting config")
            return None

    def save_scrape_summary(
        self,
        config_key: str,
        *,
        success: bool,
        duration: float,
        articles_scraped: int,
        articles_saved: int,
        errors: list[str],
    ) -> None:
        """
        Save a summary of a daily scrape run.

        Args:
            config_key: The key of the news configuration.
            success: Whether the scrape was successful.
            duration: Duration of the scrape in seconds.
            articles_scraped: Number of articles scraped.
            articles_saved: Number of articles saved to DB.
            errors: List of error messages.

        """
        try:
            config = self._get_config(config_key)
            errors_text = json.dumps(errors) if errors else ""

            ScrapeSummary.objects.create(
                config=config,
                success=success,
                duration=duration,
                articles_scraped=articles_scraped,
                articles_saved=articles_saved,
                error_count=len(errors),
                errors=errors_text,
            )
            logger.info(f"Saved scrape summary for {config_key}")
        except Exception:
            logger.exception("Error saving scrape summary")

    def save_analysis_summary(
        self,
        config_key: str,
        *,
        success: bool,
        duration: float,
        articles_analyzed: int,
        stories_identified: int,
        top_stories: list[Story],
        errors: list[str],
    ) -> None:
        """
        Save a summary of a weekly analysis run.

        Args:
            config_key: The key of the news configuration.
            success: Whether the analysis was successful.
            duration: Duration of the analysis in seconds.
            articles_analyzed: Number of articles analyzed.
            stories_identified: Number of stories identified.
            top_stories: List of top stories.
            errors: List of error messages.

        """
        try:
            config = self._get_config(config_key)
            errors_text = json.dumps(errors) if errors else ""

            # Serialize top stories
            serialized_stories = []
            for story in top_stories:
                if isinstance(story, dict):
                    serialized_stories.append(story)
                elif is_dataclass(story) and not isinstance(story, type):
                    serialized_stories.append(asdict(story))
                elif hasattr(story, "__dict__"):
                    serialized_stories.append(
                        {
                            "title": getattr(story, "title", ""),
                            "article_count": getattr(
                                story,
                                "article_count",
                                0,
                            ),
                            "sources": getattr(story, "sources", []),
                        },
                    )
                else:
                    serialized_stories.append(str(story))

            top_stories_text = json.dumps(
                _serialize_for_json(serialized_stories),
            )

            AnalysisSummary.objects.create(
                config=config,
                success=success,
                duration=duration,
                articles_analyzed=articles_analyzed,
                stories_identified=stories_identified,
                top_stories=top_stories_text,
                error_count=len(errors),
                errors=errors_text,
            )
            logger.info(f"Saved analysis summary for {config_key}")
        except Exception:
            logger.exception("Error saving analysis summary")
