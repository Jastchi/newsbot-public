"""
Business logic handlers for API endpoints.

These handlers wrap the newsbot pipeline operations and return
structured results suitable for API responses.
"""

from __future__ import annotations

import logging
import time
from graphlib import CycleError, TopologicalSorter
from typing import Any

from django.db import close_old_connections

from api.job_manager import JobStatus, job_manager
from newsbot.constants import DAILY_SCRAPE_HOUR, DAILY_SCRAPE_MINUTE
from newsbot.error_handling.email_handler import get_email_error_handler
from newsbot.pipeline import PipelineOrchestrator
from newsbot.summary_writer import SummaryWriter
from newsbot.test_data import insert_test_articles
from newsbot.utils import setup_logging, validate_environment
from utilities import load_config
from utilities.django_models import NewsConfig

logger = logging.getLogger(__name__)


def handle_run(
    config_key: str,
    *,
    force: bool = False,
    job_id: str | None = None,
) -> dict[str, Any]:
    """
    Handle a run (daily scrape) request.

    Args:
        config_key: The config key from database (e.g., "technology")
        force: Force scrape even if already done today
        job_id: Optional job ID to update

    Returns:
        Dictionary with operation results

    Raises:
        ConfigNotFoundError: If config is not found

    """
    # Close any stale database connections before starting
    close_old_connections()

    start_time = time.time()

    # Load configuration from database
    config, news_config = load_config(config_key)

    # Setup logging
    email_error_handler = get_email_error_handler()
    email_error_handler.config_name = config.name
    setup_logging(config, [email_error_handler], config_key=config_key)

    # Validate environment based on configured provider
    validate_environment(config, email_error_handler)

    if job_id:
        job_manager.update_job(job_id, JobStatus.RUNNING)

    logger.info(f"API: Starting run for config '{config_key}'")

    errors: list[str] = []

    try:
        # Create pipeline orchestrator with NewsConfig instance
        orchestrator = PipelineOrchestrator(
            config,
            config_key=config_key,
            news_config=news_config,
        )

        # Get status
        status = orchestrator.get_pipeline_status()
        logger.info(f"Pipeline Status: {status}")

        # Run daily scrape
        results = orchestrator.run_daily_scrape(force=force)

        duration = time.time() - start_time

        # Save summary to DB
        summary_writer = SummaryWriter()
        summary_writer.save_scrape_summary(
            config_key=config_key,
            success=results.success,
            duration=results.duration,
            articles_scraped=results.articles_count,
            articles_saved=results.saved_to_db,
            errors=results.errors,
        )

        result = {
            "success": results.success,
            "config_key": config_key,
            "config_name": config.name,
            "articles_scraped": results.articles_count,
            "articles_saved": results.saved_to_db,
            "duration": round(duration, 2),
            "errors": results.errors,
        }

        if job_id:
            job_manager.update_job(
                job_id,
                JobStatus.COMPLETED,
                result=result,
            )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        errors.append(error_msg)
        logger.exception(f"Error during run for config '{config_key}'")

        result = {
            "success": False,
            "config_key": config_key,
            "config_name": config.name if config else config_key,
            "articles_scraped": 0,
            "articles_saved": 0,
            "duration": round(duration, 2),
            "error": error_msg,
            "errors": errors,
        }

        if job_id:
            job_manager.update_job(
                job_id,
                JobStatus.FAILED,
                error=error_msg,
                result=result,
            )

        return result
    else:
        return result

    finally:
        # Send all collected errors in one email at the end
        email_error_handler.flush()


def handle_analyze(
    config_key: str,
    days: int | None = None,
    *,
    test: bool = False,
    email_receivers: list[str] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """
    Handle an analyze (weekly analysis) request.

    Args:
        config_key: The config key from database (e.g., "technology")
        days: Number of days to look back (defaults to config value)
        test: If True, insert test articles (only for test configs)
        email_receivers: Optional list of email receivers to override
        job_id: Optional job ID to update

    Returns:
        Dictionary with operation results

    Raises:
        ConfigNotFoundError: If config is not found

    """
    # Close any stale database connections before starting
    close_old_connections()

    start_time = time.time()

    # Load configuration from database
    config, news_config = load_config(config_key)

    # Setup logging
    email_error_handler = get_email_error_handler()
    email_error_handler.config_name = config.name
    setup_logging(config, [email_error_handler], config_key=config_key)

    # Validate environment based on configured provider
    validate_environment(config, email_error_handler)

    if job_id:
        job_manager.update_job(job_id, JobStatus.RUNNING)

    logger.info(f"API: Starting analysis for config '{config_key}'")

    errors: list[str] = []

    # Insert test articles if requested (only for test configs)
    if test:
        if not config_key.startswith("test"):
            return {
                "success": False,
                "config_key": config_key,
                "config_name": config.name,
                "articles_analyzed": 0,
                "stories_identified": 0,
                "duration": 0.0,
                "error": "--test requires a test config (i.e. test_*)",
                "errors": ["--test requires a test config (i.e. test_*)"],
            }

        logger.info("Inserting test articles (test flag provided)")
        count = insert_test_articles(config_key)
        logger.info(f"Inserted {count} test articles (2 clusters)")

    try:
        # Create pipeline orchestrator with NewsConfig instance
        orchestrator = PipelineOrchestrator(
            config,
            config_key=config_key,
            news_config=news_config,
        )

        # Set email receivers override if provided
        if email_receivers is not None:
            orchestrator.set_email_receivers_override(email_receivers)

        # Get lookback days from args or config
        lookback_days = (
            days if days is not None else config.report.lookback_days
        )

        # Run weekly analysis
        results = orchestrator.run_weekly_analysis(days_back=lookback_days)

        duration = time.time() - start_time

        # Save summary to DB
        summary_writer = SummaryWriter()
        summary_writer.save_analysis_summary(
            config_key=config_key,
            success=results.success,
            duration=results.duration,
            articles_analyzed=results.articles_count,
            stories_identified=results.stories_count,
            top_stories=results.top_stories,
            errors=results.errors,
        )

        # Build top stories summary for response
        top_stories_summary = []
        if results.top_stories:
            top_stories_summary.extend(
                [
                    {
                        "title": story.title,
                        "sources": story.sources,
                        "article_count": story.article_count,
                    }
                    for story in results.top_stories[:5]
                ],
            )

        result = {
            "success": results.success,
            "config_key": config_key,
            "config_name": config.name,
            "articles_analyzed": results.articles_count,
            "stories_identified": results.stories_count,
            "lookback_days": lookback_days,
            "top_stories": top_stories_summary,
            "duration": round(duration, 2),
            "errors": results.errors,
        }

        if job_id:
            job_manager.update_job(
                job_id,
                JobStatus.COMPLETED,
                result=result,
            )

    except Exception as e:
        duration = time.time() - start_time
        error_msg = str(e)
        errors.append(error_msg)
        logger.exception(f"Error during analysis for config '{config_key}'")

        result = {
            "success": False,
            "config_key": config_key,
            "config_name": config.name if config else config_key,
            "articles_analyzed": 0,
            "stories_identified": 0,
            "duration": round(duration, 2),
            "error": error_msg,
            "errors": errors,
        }

        if job_id:
            job_manager.update_job(
                job_id,
                JobStatus.FAILED,
                error=error_msg,
                result=result,
            )

        return result

    else:
        return result

    finally:
        # Send all collected errors in one email at the end
        email_error_handler.flush()


def daily_scrape_to_cron(hour: int, minute: int) -> str:
    """
    Convert daily scrape time to cron expression.

    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)

    Returns:
        Cron expression string (e.g., "0 2 * * *")

    """
    return f"{minute} {hour} * * *"


def weekly_analysis_to_cron(day_of_week: str, hour: int, minute: int) -> str:
    """
    Convert weekly analysis schedule to cron expression.

    Args:
        day_of_week: Day of week (mon, tue, wed, thu, fri, sat, sun)
        hour: Hour (0-23)
        minute: Minute (0-59)

    Returns:
        Cron expression string (e.g., "0 9 * * 1")

    """
    # Map day names to cron day numbers (0=Sunday, 1=Monday, etc.)
    day_map = {
        "sun": 0,
        "mon": 1,
        "tue": 2,
        "wed": 3,
        "thu": 4,
        "fri": 5,
        "sat": 6,
    }
    day_num = day_map.get(day_of_week.lower(), 1)  # Default to Monday
    return f"{minute} {hour} * * {day_num}"


def _schedule_dependency_order(
    configs: list[NewsConfig],
) -> list[NewsConfig]:
    """Order so exclude-from configs run before excluders."""
    if not configs:
        return []

    key_to_config = {c.key: c for c in configs}
    keys_set = set(key_to_config)

    # Build graph: {node: {dependencies}}
    # A config depends on the configs it excludes from.
    graph = {
        config.key: {
            dep.key
            for dep in config.exclude_articles_from_configs.all()
            if dep.key in keys_set
        }
        for config in configs
    }

    ts = TopologicalSorter(graph)
    try:
        order = list(ts.static_order())
    except CycleError:
        logger.warning("Cycle detected in exclude_articles_from_configs.")
        return configs

    return [key_to_config[k] for k in order]


def get_all_schedules() -> list[dict[str, Any]]:
    """
    Get all active config schedules in cron format.

    Returns configs in dependency order (topological sort by
    exclude_articles_from_configs) so the scheduler can run daily
    scrapes sequentially with exclude-from configs first.

    daily_scrape contains only enabled and optional cron (no per-config
    hour/minute); scrape time is fixed after refresh (e.g. 00:05).

    Returns:
        List of schedule dictionaries with cron expressions

    """
    # Prefetch exclude_articles_from_configs for active configs.
    active_configs = list(
        NewsConfig.objects.filter(is_active=True).prefetch_related(
            "exclude_articles_from_configs",
        ),
    )
    ordered_configs = _schedule_dependency_order(active_configs)

    schedules = []
    for news_config in ordered_configs:
        schedule = {
            "key": news_config.key,
            "name": news_config.display_name,
            "is_active": news_config.is_active,
            "daily_scrape": {
                "enabled": news_config.scheduler_daily_scrape_enabled,
                "cron": (
                    daily_scrape_to_cron(
                        DAILY_SCRAPE_HOUR,
                        DAILY_SCRAPE_MINUTE,
                    )
                    if news_config.scheduler_daily_scrape_enabled
                    else None
                ),
            },
            "weekly_analysis": {
                "enabled": news_config.scheduler_weekly_analysis_enabled,
                "day_of_week": (
                    news_config.scheduler_weekly_analysis_day_of_week
                ),
                "hour": news_config.scheduler_weekly_analysis_hour,
                "minute": news_config.scheduler_weekly_analysis_minute,
                "lookback_days": (
                    news_config.scheduler_weekly_analysis_lookback_days
                ),
                "cron": weekly_analysis_to_cron(
                    news_config.scheduler_weekly_analysis_day_of_week,
                    news_config.scheduler_weekly_analysis_hour,
                    news_config.scheduler_weekly_analysis_minute,
                )
                if news_config.scheduler_weekly_analysis_enabled
                else None,
            },
        }
        schedules.append(schedule)

    return schedules
