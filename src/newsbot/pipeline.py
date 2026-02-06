"""
Pipeline Orchestrator.

Coordinates all agents to execute the news analysis pipeline
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, cast

from django.db import DatabaseError, IntegrityError, OperationalError

from after_analysis import run_hooks
from newsbot.constants import SENTIMENT_THRESHOLD, TZ
from newsbot.managers import AgentManager, DatabaseManager
from newsbot.models import (
    AnalysisData,
    Article,
    PipelineStatus,
    Results,
    SentimentResult,
    SentimentSummary,
    StoryAnalysis,
    SummaryItem,
)
from utilities.django_models import NewsConfig

if TYPE_CHECKING:
    from newsbot.agents.story_clustering_agent import Story
    from utilities.django_models import NewsConfig as NewsConfigType
    from utilities.models import ConfigModel

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the entire news analysis pipeline."""

    def __init__(
        self,
        config: ConfigModel,
        config_key: str = "",
        news_config: NewsConfigType | None = None,
    ) -> None:
        """
        Initialize the Pipeline Orchestrator.

        Args:
            config: Configuration dictionary
            config_key: Config key (e.g., "technology") used to
                isolate articles per config in the database
            news_config: Optional NewsConfig Django model instance.
                If not provided, will be looked up from config_key.

        """
        self.config = config
        self.email_receivers_override = None

        # Store or look up NewsConfig instance for ForeignKey
        # relationships
        resolved_config: NewsConfigType | None
        if news_config is not None:
            resolved_config = news_config
        else:
            resolved_config = self._lookup_news_config(config_key)

        if resolved_config is None:
            msg = (
                f"NewsConfig not found for config_key: {config_key}. "
                "Ensure a NewsConfig exists in the database."
            )
            raise ValueError(msg)

        self._news_config: NewsConfigType = resolved_config
        # Set config_key from news_config if empty
        self.config_key = config_key or resolved_config.key

        # Initialize managers for database ops and lazy agent
        # initialization
        self.database_manager = DatabaseManager(self._news_config)
        self.agent_manager = AgentManager(
            config,
            database_manager=self.database_manager,
        )

        logger.info("Pipeline orchestrator initialized (agents lazy-loaded)")

    def _lookup_news_config(self, config_key: str) -> NewsConfigType | None:
        """
        Look up NewsConfig by key.

        Args:
            config_key: The config key (e.g., "technology")

        Returns:
            NewsConfig instance or None if not found

        """
        if not config_key:
            return None
        try:
            return NewsConfig.objects.filter(key=config_key).first()
        except (DatabaseError, IntegrityError, OperationalError) as e:
            logger.warning(
                f"Could not look up NewsConfig for key: {config_key}: {e}",
            )
            return None

    @property
    def news_config(self) -> NewsConfigType:
        """Get the NewsConfig instance for this pipeline."""
        return self._news_config

    def set_email_receivers_override(
        self,
        email_receivers: list[str],
    ) -> None:
        """
        Set email receivers to override database values.

        Args:
            email_receivers: List of email addresses or empty list to
                disable

        """
        self.email_receivers_override = email_receivers
        logger.info(
            f"Email receivers override set: "
            f"{email_receivers or 'disabled'}",
        )

    def set_news_config(self, news_config: NewsConfigType) -> None:
        """
        Update the NewsConfig instance for this pipeline.

        Args:
            news_config: NewsConfig Django model instance to use

        """
        self._news_config = news_config
        # Update database manager with new news_config
        self.database_manager = DatabaseManager(news_config)
        logger.info("NewsConfig updated in pipeline orchestrator")

    def _scrape_articles(self) -> list[Article]:
        """
        Step 1: Scrape news articles from all sources.

        Returns:
            List of scraped articles

        """
        logger.info(
            "\n[STEP 1/5] Scraping news articles from all sources...",
        )
        articles = self.agent_manager.scraper.scrape_all_sources()

        if articles:
            logger.info(
                f"Scraped {len(articles)} articles from "
                f"{len({a.source for a in articles})} sources",
            )

        return articles

    def _identify_top_stories(
        self,
        articles: list[Article],
        top_n: int,
    ) -> list[Story]:
        """
        Step 2: Identify top N stories across sources.

        Args:
            articles: List of articles to analyze
            top_n: Number of top stories to identify

        Returns:
            List of top stories

        """
        logger.info(
            f"\n[STEP 2/5] Identifying top {top_n} stories across sources...",
        )
        top_stories = self.agent_manager.story_clustering.identify_top_stories(
            articles,
            top_n=top_n,
        )

        if top_stories:
            logger.info(f"✓ Identified {len(top_stories)} top stories")

        return top_stories

    def _summarize_stories(
        self,
        top_stories: list[Story],
    ) -> list[StoryAnalysis]:
        """
        Step 3: Summarize articles for each story.

        Summarize by source using two-pass summarization.

        Args:
            top_stories: List of top stories to summarize

        Returns:
            List of story analyses with summaries

        """
        logger.info(
            "\n[STEP 3/5] Summarizing articles for each story by source...",
        )
        story_analyses = []

        for i, story in enumerate(top_stories, 1):
            logger.info(
                f"Processing story {i}/{len(top_stories)}: "
                f"'{story.title[:50]}...'",
            )

            # Summarize story using two-pass approach
            self.agent_manager.summarizer.summarize_story(story)

            # Group summaries by source for analysis
            source_summaries: dict[str, list[SummaryItem]] = {}
            for article in story.articles:
                if article.source not in source_summaries:
                    source_summaries[article.source] = []
                source_summaries[article.source].append(
                    {
                        "article": article,
                        "summary": article.summary,
                    },
                )

            story_analyses.append(
                {
                    "story": story,
                    "source_summaries": source_summaries,
                },
            )

        logger.info(
            "Summarized "
            f"{sum(len(story.articles) for story in top_stories)} "
            f"articles across {len(top_stories)} stories",
        )

        return story_analyses

    def _analyze_sentiment(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> None:
        """
        Step 4: Analyze sentiment for each story by source.

        Args:
            story_analyses: List of story analyses to add sentiment to

        """
        logger.info(
            "\n[STEP 4/5] Analyzing sentiment for each story by source...",
        )

        for analysis in story_analyses:
            story = analysis["story"]
            source_sentiments: dict[str, list[SentimentResult]] = {}

            # Analyze sentiment for all articles in this story
            for article in story.articles:
                sentiment = (
                    self.agent_manager.sentiment_analyzer.analyze_article(
                        article,
                    )
                )
                article.sentiment = sentiment

                if article.source not in source_sentiments:
                    source_sentiments[article.source] = []
                source_sentiments[article.source].append(sentiment)

            # Calculate average sentiment per source for this story
            source_sentiment_summary: dict[str, SentimentSummary] = {}
            for source, sentiments in source_sentiments.items():
                avg_compound = sum(s.compound for s in sentiments) / len(
                    sentiments,
                )
                source_sentiment_summary[source] = cast(
                    "SentimentSummary",
                    {
                        "avg_sentiment": avg_compound,
                        "label": "positive"
                        if avg_compound > SENTIMENT_THRESHOLD
                        else "negative"
                        if avg_compound < -SENTIMENT_THRESHOLD
                        else "neutral",
                        "article_count": len(sentiments),
                        "sentiments": sentiments,
                    },
                )

            analysis["source_sentiments"] = source_sentiment_summary

        logger.info(
            f"Completed sentiment analysis for {len(story_analyses)} stories",
        )

    def _generate_report(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> str:
        """
        Step 5: Generate top stories report.

        Args:
            story_analyses: List of story analyses to include in report

        Returns:
            Path to generated report

        """
        logger.info("\n[STEP 5/5] Generating top stories report...")
        _, report_path = (
            self.agent_manager.report_generator.generate_top_stories_report(
                story_analyses,
            )
        )
        logger.info("Report generated")
        return report_path

    def run_daily_scrape(self, *, force: bool = False) -> Results:
        """
        Daily job: scrape and store articles only.

        Args:
            force: Force scraping even if already scraped today

        Returns:
            Results object with scraping results

        """
        logger.info("Starting daily scrape job")
        results = Results()

        try:
            # Check if already scraped today
            if not force and self.database_manager.has_scraped_today():
                logger.info("Articles already scraped today. Skipping scrape.")
                logger.info("Use --force to override this check.")
                results.success = True
                results.end_time = datetime.now(TZ)
                results.duration = (
                    results.end_time - results.start_time
                ).total_seconds()
                return results

            # Step 1: Scrape articles
            logger.info("Scraping news articles from all sources...")
            articles = self.agent_manager.scraper.scrape_all_sources()

            if not articles:
                logger.error("No articles were scraped.")
                results.errors.append("No articles scraped")
                results.end_time = datetime.now(TZ)
                results.duration = (
                    results.end_time - results.start_time
                ).total_seconds()
                return results

            results.articles_count = len(articles)
            logger.info(
                f"Scraped {len(articles)} articles from "
                f"{len({a.source for a in articles})} sources",
            )

            # Step 2: Save to database
            logger.info("Saving articles to database...")
            saved_count = self.database_manager.save_articles(articles)
            results.saved_to_db = saved_count

            # Finalize results
            results.success = True
            results.end_time = datetime.now(TZ)
            results.duration = (
                results.end_time - results.start_time
            ).total_seconds()

            logger.info("Daily scrape completed successfully!")
            logger.info(f"Duration: {results.duration:.2f} seconds")
            logger.info(f"Articles saved: {saved_count}/{len(articles)}")

        except Exception as e:
            logger.exception(
                "Daily scrape failed with error",
            )
            results.errors.append(str(e))
            results.end_time = datetime.now(TZ)
            results.duration = (
                results.end_time - results.start_time
            ).total_seconds()
            return results
        else:
            return results

    def _weekly_step1_load_articles(
        self,
        days_back: int,
        results: Results,
    ) -> list[Article] | None:
        """
        Step 1: Load articles from database.

        Args:
            days_back: Number of days to look back
            results: Results object to update

        Returns:
            List of articles or None if error/no articles

        """
        try:
            articles = self.database_manager.load_articles(days_back)
        except RuntimeError as e:
            logger.exception("Failed to load articles from database")
            results.errors.append(str(e))
            results.end_time = datetime.now(TZ)
            results.duration = (
                results.end_time - results.start_time
            ).total_seconds()
            return None

        if not articles:
            logger.error("No articles found for analysis period")
            results.errors.append(
                "No articles in database for this period",
            )
            results.end_time = datetime.now(TZ)
            results.duration = (
                results.end_time - results.start_time
            ).total_seconds()
            return None

        results.articles_count = len(articles)
        return articles

    def _weekly_step2_identify_stories(
        self,
        articles: list[Article],
        results: Results,
    ) -> list[Story] | None:
        """
        Step 2: Identify top N stories.

        Args:
            articles: List of articles to analyze
            results: Results object to update

        Returns:
            List of top stories or None if no stories found

        """
        top_n = self.config.story_clustering.top_stories_count
        top_stories = self._identify_top_stories(articles, top_n)

        if not top_stories:
            logger.error("No stories identified. Analysis stopped.")
            results.errors.append("No stories identified")
            results.end_time = datetime.now(TZ)
            results.duration = (
                results.end_time - results.start_time
            ).total_seconds()
            return None

        results.stories_count = len(top_stories)
        return top_stories

    def _weekly_step3_summarize(
        self,
        top_stories: list[Story],
    ) -> list[StoryAnalysis]:
        """
        Step 3: Summarize articles for each story.

        Args:
            top_stories: List of top stories to summarize

        Returns:
            List of story analyses with summaries

        """
        return self._summarize_stories(top_stories)

    def _weekly_step4_sentiment(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> None:
        """
        Step 4: Analyze sentiment for each story.

        Args:
            story_analyses: List of story analyses to add sentiment to

        """
        self._analyze_sentiment(story_analyses)

    def _weekly_step5_generate_report(
        self,
        story_analyses: list[StoryAnalysis],
    ) -> str:
        """
        Step 5: Generate report.

        Args:
            story_analyses: List of story analyses to include in report

        Returns:
            Path to generated report

        """
        return self._generate_report(story_analyses)

    def _weekly_finalize_results(
        self,
        results: Results,
        articles: list[Article],
        top_stories: list[Story],
        story_analyses: list[StoryAnalysis],
        report_path: str,
        days_back: int,
    ) -> None:
        """
        Finalize weekly analysis results and run hooks.

        Args:
            results: Results object to finalize
            articles: List of analyzed articles
            top_stories: List of top stories
            story_analyses: List of story analyses
            report_path: Path to generated report
            days_back: Number of days analyzed

        """
        # Update database with summaries and sentiments
        logger.info("Updating database with analysis results...")
        self.database_manager.update_articles_with_analysis(articles)
        logger.info("Database updated with analysis results")

        # Finalize results
        results.success = True
        results.end_time = datetime.now(TZ)
        results.duration = (
            results.end_time - results.start_time
        ).total_seconds()
        results.top_stories = top_stories
        results.story_analyses = story_analyses
        results.report_path = report_path

        logger.info("Weekly analysis completed successfully!")
        logger.info(f"Duration: {results.duration:.2f} seconds")
        logger.info(f"Top Stories: {len(top_stories)}")
        logger.info(
            f"Total Sources: {len({a.source for a in articles})}",
        )

        # Run post-analysis hooks
        if report_path:
            cutoff_date = datetime.now(TZ) - timedelta(days=days_back)
            analysis_data: AnalysisData = {
                "success": results.success,
                "articles_count": results.articles_count,
                "stories_count": results.stories_count,
                "duration": results.duration,
                "timestamp": results.end_time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                ),
                "format": self.config.report.format,
                "config_name": self.config.name,
                "config_key": self.config_key,
                "from_date": cutoff_date,
                "to_date": datetime.now(TZ),
                "email_receivers_override": self.email_receivers_override,
            }

            run_hooks(Path(report_path), analysis_data)

    def run_weekly_analysis(self, days_back: int = 7) -> Results:
        """
        Weekly job: analyze articles from database.

        Args:
            days_back: Number of days to analyze

        Returns:
            Results object with analysis results

        """
        logger.info(f"Starting weekly analysis for last {days_back} days")
        results = Results()

        try:
            # Step 1: Load articles from database
            articles = self._weekly_step1_load_articles(days_back, results)
            if articles is None:
                return results

            # Step 2: Identify top N stories
            top_stories = self._weekly_step2_identify_stories(
                articles,
                results,
            )
            if top_stories is None:
                return results

            # Step 3: Summarize articles for each story
            story_analyses = self._weekly_step3_summarize(top_stories)

            # Step 4: Analyze sentiment for each story
            self._weekly_step4_sentiment(story_analyses)

            # Step 5: Generate report
            report_path = self._weekly_step5_generate_report(story_analyses)

            # Finalize results and run hooks
            self._weekly_finalize_results(
                results,
                articles,
                top_stories,
                story_analyses,
                report_path,
                days_back,
            )

        except Exception as e:
            logger.exception(
                "Weekly analysis failed with error",
            )
            results.errors.append(str(e))
            results.end_time = datetime.now(TZ)
            results.duration = (
                results.end_time - results.start_time
            ).total_seconds()
            return results
        else:
            return results

    def get_pipeline_status(self) -> PipelineStatus:
        """
        Get current status of the pipeline.

        Returns:
            Status dictionary

        """
        return cast(
            "PipelineStatus",
            {
                "agents": self.agent_manager.get_agent_status(),
                "config": {
                    "country": self.config.country,
                    "sources": len(self.config.news_sources),
                    "llm_model": self.config.llm.model,
                },
            },
        )
