"""Database Manager for article database operations."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from django.db import (
    DatabaseError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    close_old_connections,
    connection,
)

from newsbot.constants import TZ
from newsbot.models import Article
from utilities.django_models import Article as DjangoArticle

if TYPE_CHECKING:
    from utilities.django_models import NewsConfig as NewsConfigType

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all database operations for articles."""

    def __init__(self, news_config: NewsConfigType) -> None:
        """
        Initialize the Database Manager.

        Args:
            news_config: NewsConfig Django model instance for ForeignKey
                relationships and filtering articles by config.

        """
        self._news_config = news_config

    @property
    def news_config(self) -> NewsConfigType:
        """Get the NewsConfig instance."""
        return self._news_config

    def save_articles(self, articles: list[Article]) -> int:
        """
        Save articles to database using bulk operations.

        Args:
            articles: list of Article objects

        Returns:
            Number of articles saved

        """
        if not articles:
            return 0

        close_old_connections()
        saved_count = 0

        try:
            articles_by_url = {a.url: a for a in articles}
            urls = list(articles_by_url.keys())

            existing_rows = list(
                DjangoArticle.objects.filter(
                    config=self._news_config,
                    url__in=urls,
                ).only("id", "url", "content", "scraped_date"),
            )
            existing_urls = {row.url for row in existing_rows}

            new_objects = [
                DjangoArticle(
                    config=self._news_config,
                    title=article.title,
                    content=article.content or "",
                    summary=article.summary or "",
                    source=article.source,
                    url=article.url,
                    published_date=article.published_date,
                    scraped_date=article.scraped_date,
                    sentiment_score=article.sentiment.compound
                    if article.sentiment
                    else None,
                    sentiment_label=article.sentiment.label
                    if article.sentiment
                    else "",
                )
                for url, article in articles_by_url.items()
                if url not in existing_urls
            ]

            if new_objects:
                # batch_size=100 is a safe default for SQLite/Postgres
                DjangoArticle.objects.bulk_create(
                    new_objects,
                    batch_size=100,
                    ignore_conflicts=True,
                )
                saved_count = len(new_objects)
                logger.info(f"Saved {saved_count} new articles to database")
            else:
                logger.info("No new articles to save")

            self._backfill_empty_content(articles_by_url, existing_rows)

        except (
            DatabaseError,
            InterfaceError,
            IntegrityError,
            OperationalError,
        ):
            logger.exception("Database error saving articles")
            connection.close()
            saved_count = 0

        return saved_count

    def _backfill_empty_content(
        self,
        articles_by_url: dict[str, Article],
        existing_rows: list[DjangoArticle],
    ) -> None:
        """
        Backfill content for existing rows that have no content.

        Only when DB content is blank and the incoming article for that
        URL has non-empty content.
        """
        to_backfill = []
        for db_art in existing_rows:
            incoming = articles_by_url.get(db_art.url)
            if not incoming or not (incoming.content or "").strip():
                continue
            if not (db_art.content or "").strip():
                db_art.content = incoming.content or ""
                db_art.scraped_date = incoming.scraped_date
                to_backfill.append(db_art)

        if to_backfill:
            count = DjangoArticle.objects.bulk_update(
                to_backfill,
                ["content", "scraped_date"],
                batch_size=100,
            )
            logger.info(
                f"Backfilled content for {count} existing "
                "articles that had no content",
            )

    def load_articles(self, days_back: int) -> list[Article]:
        """
        Load articles from database for analysis.

        Args:
            days_back: Number of days to look back

        Returns:
            List of articles from database

        """
        close_old_connections()
        cutoff_date = datetime.now(TZ) - timedelta(days=days_back)

        db_articles = DjangoArticle.objects.filter(
            config=self._news_config,
            scraped_date__gte=cutoff_date,
        ).only(
            "title",
            "content",
            "source",
            "url",
            "published_date",
            "scraped_date",
            "summary",
        )

        articles = [
            Article(
                title=db_art.title,
                content=db_art.content or "",
                source=db_art.source,
                url=db_art.url,
                published_date=db_art.published_date,
                scraped_date=db_art.scraped_date,
                summary=db_art.summary,
            )
            for db_art in db_articles
        ]

        logger.info(f"Loaded {len(articles)} articles from database")
        return articles

    def update_articles_with_analysis(self, articles: list[Article]) -> int:
        """
        Update existing articles in database with analysis results.

        Update with summaries and sentiment using bulk_update. Retries
        once after closing the connection if a stale connection error
        is hit (long-running workers can outlive the server-side TCP
        connection).

        Args:
            articles: list of Article objects with analysis results

        Returns:
            Number of articles updated

        """
        if not articles:
            return 0

        for attempt in range(2):
            try:
                return self._do_update_articles_with_analysis(articles)
            except (DatabaseError, IntegrityError, OperationalError):
                connection.close()
                if attempt == 0:
                    logger.warning(
                        "Database error updating articles, retrying...",
                    )
                else:
                    logger.exception("Database error updating articles")
        return 0

    def _do_update_articles_with_analysis(
        self, articles: list[Article],
    ) -> int:
        """Apply summary/sentiment updates in a single bulk_update."""
        articles_by_url = {a.url: a for a in articles}
        urls = list(articles_by_url.keys())

        existing_objects = DjangoArticle.objects.filter(
            config=self._news_config,
            url__in=urls,
        ).only(
            "id", "url", "summary", "sentiment_score", "sentiment_label",
        )

        objects_to_update = []
        fields_to_update: set[str] = set()

        for db_obj in existing_objects:
            article = articles_by_url.get(db_obj.url)
            if not article:
                continue

            changed = False

            if article.summary and article.summary != db_obj.summary:
                db_obj.summary = article.summary
                changed = True
                fields_to_update.add("summary")

            if article.sentiment and (
                article.sentiment.compound != db_obj.sentiment_score
                or article.sentiment.label != db_obj.sentiment_label
            ):
                db_obj.sentiment_score = article.sentiment.compound
                db_obj.sentiment_label = article.sentiment.label
                changed = True
                fields_to_update.add("sentiment_score")
                fields_to_update.add("sentiment_label")

            if changed:
                objects_to_update.append(db_obj)

        if not objects_to_update or not fields_to_update:
            logger.debug("No articles needed updating")
            return 0

        updated_count = DjangoArticle.objects.bulk_update(
            objects_to_update,
            fields=list(fields_to_update),
            batch_size=100,
        )
        logger.debug(
            f"Updated {updated_count} articles with analysis results",
        )
        return updated_count

    def url_exists(self, url: str) -> bool:
        """
        Check if an article with the given URL already exists.

        For this config.

        Args:
            url: Article URL to check

        Returns:
            True if article exists for this config, False otherwise

        """
        close_old_connections()
        try:
            return DjangoArticle.objects.filter(
                url=url,
                config=self._news_config,
            ).exists()
        except (
            DatabaseError,
            InterfaceError,
            IntegrityError,
            OperationalError,
        ):
            logger.exception("Database error checking if URL exists")
            connection.close()
            return False

    def url_exists_with_content(self, url: str) -> bool:
        """
        Check if an article with the given URL exists and has content.

        Used by the scraper to decide whether to skip fetching full
        content: only skip when the URL exists and already has content
        (so we re-fetch
        when the stored article has no content).

        Args:
            url: Article URL to check

        Returns:
            True if article exists for this config and has non-blank
            content, False otherwise

        """
        close_old_connections()
        try:
            content = DjangoArticle.objects.filter(
                url=url,
                config=self._news_config,
            ).values_list("content", flat=True).first()
            return content is not None and bool((content or "").strip())
        except (
            DatabaseError,
            InterfaceError,
            IntegrityError,
            OperationalError,
        ):
            logger.exception(
                "Database error checking if URL exists with content",
            )
            connection.close()
            return False

    def url_exists_in_any_config(
        self, url: str, config_keys: list[str],
    ) -> bool:
        """
        Check if URL exists in any of the given configs.

        Args:
            url: Article URL to check
            config_keys: List of config keys to check

        Returns:
            True if URL exists in any of the configs, False otherwise.
            Returns False if config_keys is empty.

        """
        if not config_keys:
            return False
        close_old_connections()
        try:
            return DjangoArticle.objects.filter(
                config__key__in=config_keys,
                url=url,
            ).exists()
        except (
            DatabaseError,
            InterfaceError,
            IntegrityError,
            OperationalError,
        ):
            logger.exception(
                "Database error checking if URL exists in configs %s",
                config_keys,
            )
            connection.close()
            return False

    def has_scraped_today(self) -> bool:
        """
        Check if articles have already been scraped today.

        Returns:
            True if articles from today exist in database

        """
        close_old_connections()
        try:
            today_start = datetime.now(TZ).replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0,
            )
            today_end = today_start + timedelta(days=1)

            return DjangoArticle.objects.filter(
                config=self._news_config,
                scraped_date__gte=today_start,
                scraped_date__lt=today_end,
            ).exists()
        except (
            DatabaseError, InterfaceError, IntegrityError, OperationalError,
        ):
            logger.exception("Database error checking if scraped today")
            connection.close()
            return False
