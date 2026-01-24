"""
Main entry point for NewsBot.

Runs the news analysis pipeline with optional scheduling
"""

import argparse
import logging
from argparse import Namespace
from datetime import datetime
from textwrap import dedent

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from newsbot.constants import TIMEZONE_STR, TZ
from newsbot.error_handling.email_handler import (
    EmailErrorHandler,
    get_email_error_handler,
)
from newsbot.pipeline import PipelineOrchestrator
from newsbot.summary_writer import SummaryWriter
from newsbot.test_data import insert_test_articles
from newsbot.utils import setup_logging, validate_environment
from utilities import load_config, setup_django
from utilities import models as config_models

logger = logging.getLogger(__name__)


def run_once(args: Namespace) -> None:
    """
    Run pipeline once immediately.

    Args:
        args: Command line arguments

    """
    # Setup Django first (needed for load_config)
    setup_django()

    # Load configuration from database
    config, news_config = load_config(args.config)

    # Setup logging
    email_error_handler = get_email_error_handler()
    email_error_handler.config_name = config.name
    setup_logging(config, [email_error_handler], config_key=args.config)

    # Validate environment based on configured provider
    validate_environment(config, email_error_handler)

    logger.info("Starting NewsBot - Single Run Mode")

    try:
        # Create and run pipeline
        orchestrator = PipelineOrchestrator(
            config,
            config_key=args.config,
            news_config=news_config,
        )

        # Set email receivers override if provided
        if (
            hasattr(args, "email_receivers")
            and args.email_receivers is not None
        ):
            orchestrator.set_email_receivers_override(args.email_receivers)

        # Get status
        status = orchestrator.get_pipeline_status()
        logger.info(f"Pipeline Status: {status}")

        # Run daily scrape (only scrapes if not already done today)
        results = orchestrator.run_daily_scrape(force=args.force)

        # Print summary
        print("\n" + "=" * 70)
        print("DAILY SCRAPE SUMMARY")
        print("=" * 70)
        print(f"Success: {results.success}")
        print(f"Articles Scraped: {results.articles_count}")
        print(f"Duration: {results.duration:.2f} seconds")

        if hasattr(results, "saved_to_db"):
            print(f"Articles Saved to DB: {results.saved_to_db}")

        if results.errors:
            print(f"\nErrors: {len(results.errors)}")
            for error in results.errors:
                print(f"  - {error}")

        print("=" * 70 + "\n")

        # Save summary to DB
        summary_writer = SummaryWriter()
        summary_writer.save_scrape_summary(
            config_key=args.config,
            success=results.success,
            duration=results.duration,
            articles_scraped=results.articles_count,
            articles_saved=getattr(results, "saved_to_db", 0),
            errors=results.errors,
        )

    finally:
        # Send all collected errors in one email at the end of the run
        email_error_handler.flush()


def run_analysis(args: Namespace) -> None:
    """
    Run weekly analysis on existing database articles.

    Args:
        args: Command line arguments

    """
    # Setup Django first (needed for load_config)
    setup_django()

    # Load configuration from database
    config, news_config = load_config(args.config)

    # Setup logging
    email_error_handler = get_email_error_handler()
    email_error_handler.config_name = config.name
    setup_logging(config, [email_error_handler], config_key=args.config)

    # Validate environment based on configured provider
    validate_environment(config, email_error_handler)

    logger.info("Starting NewsBot - Analysis Mode")

    # Insert test articles if --test flag is provided
    if hasattr(args, "test") and args.test:
        # Only allow --test with test configs (because database is
        # cleared)
        if not args.config.startswith("test"):
            logger.error("--test requires a test config (i.e. test_*)")
            raise SystemExit(1)

        logger.info("Inserting test articles (--test flag provided)")
        count = insert_test_articles(args.config)
        logger.info("Inserted %d test articles (2 clusters)", count)

    try:
        # Create orchestrator
        orchestrator = PipelineOrchestrator(
            config,
            config_key=args.config,
            news_config=news_config,
        )

        # Set email receivers override if provided
        if (
            hasattr(args, "email_receivers")
            and args.email_receivers is not None
        ):
            orchestrator.set_email_receivers_override(args.email_receivers)

        # Get lookback days from config or args
        lookback_days = (
            args.days
            if hasattr(args, "days") and args.days
            else config.report.lookback_days
        )

        # Run weekly analysis
        results = orchestrator.run_weekly_analysis(days_back=lookback_days)

        # Print summary
        print("\n" + "=" * 70)
        print("WEEKLY ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Success: {results.success}")
        print(f"Analysis Period: Last {lookback_days} days")
        print(f"Articles Analyzed: {results.articles_count}")

        if hasattr(results, "stories_count"):
            print(f"Top Stories Identified: {results.stories_count}")

        print(f"Duration: {results.duration:.2f} seconds")

        # Print top stories if available
        if hasattr(results, "top_stories"):
            print(f"\nTop {len(results.top_stories)} Stories:")
            for i, story in enumerate(results.top_stories[:5], 1):
                print(f"  {i}. {story.title[:70]}...")
                print(
                    f"Sources: {', '.join(story.sources)} "
                    f"({story.article_count} articles)",
                )

        if results.errors:
            print(f"\nErrors: {len(results.errors)}")
            for error in results.errors:
                print(f"  - {error}")

        print("=" * 70 + "\n")

        # Save summary to DB
        summary_writer = SummaryWriter()
        summary_writer.save_analysis_summary(
            config_key=args.config,
            success=results.success,
            duration=results.duration,
            articles_analyzed=results.articles_count,
            stories_identified=getattr(results, "stories_count", 0),
            top_stories=getattr(results, "top_stories", []),
            errors=results.errors,
        )

    finally:
        # Send all collected errors in one email at the end of the run
        email_error_handler.flush()


def _setup_scheduler_config(
    config: config_models.ConfigModel,
) -> tuple[
    config_models.DailyScrapeConfigModel,
    config_models.WeeklyAnalysisConfigModel,
]:
    """
    Extract and parse scheduler configuration.

    Args:
        config: Application configuration dictionary

    Returns:
        Tuple of (daily_config, weekly_config)

    """
    scheduler_config = config.scheduler
    daily_config = scheduler_config.daily_scrape
    weekly_config = scheduler_config.weekly_analysis

    return daily_config, weekly_config


def _refresh_config_and_jobs(
    scheduler: BlockingScheduler,
    email_error_handler: EmailErrorHandler,
    config_key: str,
) -> None:
    """
    Reload configuration from database and update scheduled jobs.

    This function is called daily at midnight to pick up any schedule
    changes made to the database without requiring a restart.

    Args:
        scheduler: APScheduler scheduler instance
        email_error_handler: Email error handler
        config_key: Configuration key to reload

    """
    logger.info(
        "Refreshing configuration from database (midnight refresh)...",
    )

    try:
        # Reload configuration from database
        new_config, _ = load_config(config_key)

        # Extract new scheduler configuration
        new_daily_config, new_weekly_config = (
            _setup_scheduler_config(new_config)
        )

        # Handle daily scrape job
        _update_daily_scrape_job(
            scheduler,
            new_daily_config,
            email_error_handler,
            config_key,
        )

        # Handle weekly analysis job
        _update_weekly_analysis_job(
            scheduler,
            new_weekly_config,
            email_error_handler,
            config_key,
        )

        logger.info("Configuration refresh completed successfully")

    except Exception:
        logger.exception("Failed to refresh configuration")
        # Continue with current schedule on error


def _update_daily_scrape_job(
    scheduler: BlockingScheduler,
    daily_config: config_models.DailyScrapeConfigModel,
    email_error_handler: EmailErrorHandler,
    config_key: str,
) -> None:
    """
    Update or add/remove the daily scrape job based on current config.

    Args:
        scheduler: APScheduler scheduler instance
        daily_config: Daily scrape configuration
        email_error_handler: Email error handler
        config_key: Configuration key

    """
    job_id = "daily_scrape_job"
    existing_job = scheduler.get_job(job_id)

    if daily_config.enabled:
        if existing_job:
            # Reschedule existing job with new time
            new_trigger = CronTrigger(
                hour=daily_config.hour,
                minute=daily_config.minute,
                timezone=TZ,
            )
            scheduler.reschedule_job(
                job_id,
                trigger=new_trigger,
            )
            logger.info(
                f"Daily scrape job rescheduled to "
                f"{daily_config.hour:02d}:{daily_config.minute:02d} "
                f"{TIMEZONE_STR}",
            )
        else:
            # Add new job
            _schedule_daily_scrape(
                scheduler,
                daily_config,
                email_error_handler,
                config_key,
            )
            logger.info("Daily scrape job added")
    elif existing_job:
        # Remove disabled job
        scheduler.remove_job(job_id)
        logger.info("Daily scrape job removed (disabled in config)")


def _update_weekly_analysis_job(
    scheduler: BlockingScheduler,
    weekly_config: config_models.WeeklyAnalysisConfigModel,
    email_error_handler: EmailErrorHandler,
    config_key: str,
) -> None:
    """
    Update or add/remove weekly analysis job based on current config.

    Args:
        scheduler: APScheduler scheduler instance
        weekly_config: Weekly analysis configuration
        email_error_handler: Email error handler
        config_key: Configuration key

    """
    job_id = "weekly_analysis_job"
    existing_job = scheduler.get_job(job_id)

    if weekly_config.enabled:
        if existing_job:
            # Reschedule existing job with new time
            new_trigger = CronTrigger(
                day_of_week=weekly_config.day_of_week,
                hour=weekly_config.hour,
                minute=weekly_config.minute,
                timezone=TZ,
            )
            scheduler.reschedule_job(
                job_id,
                trigger=new_trigger,
            )
            logger.info(
                f"Weekly analysis job rescheduled to "
                f"{weekly_config.day_of_week.upper()} "
                f"{weekly_config.hour:02d}:{weekly_config.minute:02d} "
                f"{TIMEZONE_STR}",
            )
        else:
            # Add new job
            _schedule_weekly_analysis(
                scheduler,
                weekly_config,
                email_error_handler,
                config_key,
            )
            logger.info("Weekly analysis job added")
    elif existing_job:
        # Remove disabled job
        scheduler.remove_job(job_id)
        logger.info("Weekly analysis job removed (disabled in config)")


def _register_scheduled_jobs(
    scheduler: BlockingScheduler,
    email_error_handler: EmailErrorHandler,
    daily_config: config_models.DailyScrapeConfigModel,
    weekly_config: config_models.WeeklyAnalysisConfigModel,
    config_key: str,
) -> None:
    """
    Register all enabled scheduled jobs.

    Args:
        scheduler: APScheduler scheduler instance
        email_error_handler: Email error handler
        daily_config: Daily scrape configuration
        weekly_config: Weekly analysis configuration
        config_key: Configuration key

    """
    daily_enabled = daily_config.enabled
    weekly_enabled = weekly_config.enabled

    if daily_enabled:
        _schedule_daily_scrape(
            scheduler,
            daily_config,
            email_error_handler,
            config_key,
        )

    if weekly_enabled:
        _schedule_weekly_analysis(
            scheduler,
            weekly_config,
            email_error_handler,
            config_key,
        )


def _print_scheduler_info(
    scheduler: BlockingScheduler,
) -> bool:
    """
    Print scheduler information and check if jobs exist.

    Args:
        scheduler: APScheduler scheduler instance

    Returns:
        True if jobs exist, False otherwise

    """
    jobs = scheduler.get_jobs()
    if not jobs:
        print("\nNo scheduled jobs enabled. Check your config.yaml file.")
        logger.warning("No scheduled jobs enabled")
        return False

    print("\n" + "=" * 70)
    print("NEWSBOT SCHEDULER STARTED")
    print("=" * 70)

    for job in jobs:
        print(f"{job.name}:")
        next_run = job.trigger.get_next_fire_time(
            None,
            datetime.now(TZ),
        )
        if next_run:
            print(f"  Next Run: {next_run}")
        else:
            print(f"  Trigger: {job.trigger}")

    print("=" * 70)
    print("\nPress Ctrl+C to stop the scheduler\n")
    return True


def _run_immediate_scrape(
    args: Namespace,
    orchestrator: PipelineOrchestrator,
    email_error_handler: EmailErrorHandler,
) -> None:
    """
    Run immediate scrape if requested.

    Args:
        args: Command line arguments
        orchestrator: Pipeline orchestrator
        email_error_handler: Email error handler

    """
    if not args.run_now:
        return

    logger.info("Running daily scrape immediately as requested...")
    try:
        orchestrator.run_daily_scrape(force=False)
    finally:
        email_error_handler.flush()


def run_scheduled(args: Namespace) -> None:
    """
    Run pipeline on a schedule.

    Args:
        args: Command line arguments

    """
    # Setup Django first (needed for load_config)
    setup_django()

    # Load configuration from database
    config, news_config = load_config(args.config)

    # Setup logging
    email_error_handler = get_email_error_handler()
    email_error_handler.config_name = config.name
    setup_logging(config, [email_error_handler], config_key=args.config)

    # Validate environment based on configured provider
    validate_environment(config, email_error_handler)

    logger.info("Starting NewsBot - Scheduled Mode")

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config,
        config_key=args.config,
        news_config=news_config,
    )

    # Set email receivers override if provided
    if hasattr(args, "email_receivers") and args.email_receivers is not None:
        orchestrator.set_email_receivers_override(args.email_receivers)

    # Extract scheduler configuration
    daily_config, weekly_config = (
        _setup_scheduler_config(config)
    )

    # Create scheduler
    scheduler = BlockingScheduler(timezone=TZ)

    # Register all enabled jobs
    _register_scheduled_jobs(
        scheduler,
        email_error_handler,
        daily_config,
        weekly_config,
        args.config,
    )

    # Add midnight config refresh job
    midnight_trigger = CronTrigger(
        hour=0,
        minute=0,
        timezone=TZ,
    )
    scheduler.add_job(
        _refresh_config_and_jobs,
        trigger=midnight_trigger,
        id="config_refresh_job",
        name="Midnight Config Refresh",
        args=[
            scheduler,
            email_error_handler,
            args.config,
        ],
        replace_existing=True,
    )
    logger.info(
        f"Config refresh job scheduled: Daily at 00:00 {TIMEZONE_STR}",
    )

    # Print scheduler info and check if jobs exist
    if not _print_scheduler_info(scheduler):
        return

    # Run immediate scrape if requested
    _run_immediate_scrape(args, orchestrator, email_error_handler)

    # Start scheduler
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down scheduler...")
        logger.info("Scheduler stopped by user")


def main() -> None:
    """
    Enter here.

    This is the main entry point for NewsBot. It parses command line
    arguments and executes the appropriate mode: run once, analyze, or
    schedule.
    """
    parser = argparse.ArgumentParser(
        description="NewsBot - Automated News Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""
            Examples:
              # Run daily scrape (skips if already scraped today)
              newsbot run --config technology

              # Force daily scrape even if already done
              newsbot run --config technology --force

              # Run weekly analysis on last 7 days
              newsbot analyze --config technology

              # Run weekly analysis on last 14 days
              newsbot analyze --config technology --days 14

              # Run analysis with test articles (for testing)
              newsbot analyze --config test_technology --test

              # Run on schedule (daily scrape + weekly analysis)
              newsbot schedule --config technology

              # Run scrape now and continue on schedule
              newsbot schedule --config technology --run-now
        """),
    )

    parser.add_argument(
        "mode",
        choices=["run", "analyze", "schedule"],
        help=(
            "Execution mode: run (daily scrape), analyze (weekly analysis), "
            "or schedule (both)"
        ),
    )

    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Config key from database (e.g., 'technology', 'test_technology')"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force scrape even if already done today (run mode only)",
    )

    parser.add_argument(
        "--days",
        type=int,
        help=(
            "Number of days to analyze "
            "(analyze mode only, default from config)"
        ),
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Use test articles instead of database articles "
            "(analyze mode only, inserts 5 deterministic test articles). "
            "Must be used with a test config (i.e. test_*)"
        ),
    )

    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run daily scrape immediately when starting scheduler",
    )

    parser.add_argument(
        "--email-receivers",
        nargs="*",
        help=(
            "Override email receivers from database. "
            "Provide one or more email addresses, or no arguments to "
            "disable emails"
        ),
    )

    args = parser.parse_args()

    # Execute based on mode
    if args.mode == "run":
        run_once(args)
    elif args.mode == "analyze":
        run_analysis(args)
    elif args.mode == "schedule":
        run_scheduled(args)


def _schedule_daily_scrape(
    scheduler: BlockingScheduler,
    daily_config: config_models.DailyScrapeConfigModel,
    email_error_handler: EmailErrorHandler,
    config_key: str,
) -> None:
    """
    Schedule the daily scrape job.

    Args:
        scheduler: APScheduler instance
        daily_config: Daily scrape configuration
        email_error_handler: Email error handler to flush after job
            completion
        config_key: Configuration key

    """
    daily_hour = daily_config.hour
    daily_minute = daily_config.minute

    daily_trigger = CronTrigger(
        hour=daily_hour,
        minute=daily_minute,
        timezone=TZ,
    )

    def run_daily_scrape_with_flush() -> None:
        """Run daily scrape and flush email errors at the end."""
        try:
            # Instantiate a fresh orchestrator
            job_config, job_news_config = load_config(config_key)
            orchestrator = PipelineOrchestrator(
                job_config,
                config_key=config_key,
                news_config=job_news_config,
            )

            results = orchestrator.run_daily_scrape(force=False)

            # Save summary to DB
            summary_writer = SummaryWriter()
            summary_writer.save_scrape_summary(
                config_key=config_key,
                success=results.success,
                duration=results.duration,
                articles_scraped=results.articles_count,
                articles_saved=getattr(results, "saved_to_db", 0),
                errors=results.errors,
            )
        finally:
            email_error_handler.flush()

    scheduler.add_job(
        run_daily_scrape_with_flush,
        trigger=daily_trigger,
        id="daily_scrape_job",
        name="Daily News Scrape",
        replace_existing=True,
    )
    logger.info(
        f"Daily scrape job scheduled: Every day at "
        f"{daily_hour:02d}:{daily_minute:02d} {TIMEZONE_STR}",
    )


def _schedule_weekly_analysis(
    scheduler: BlockingScheduler,
    weekly_config: config_models.WeeklyAnalysisConfigModel,
    email_error_handler: EmailErrorHandler,
    config_key: str,
) -> None:
    """
    Schedule the weekly analysis job.

    Args:
        scheduler: APScheduler instance
        weekly_config: Weekly analysis configuration
        email_error_handler: Email error handler to flush after job
            completion
        config_key: Configuration key

    """
    weekly_day = weekly_config.day_of_week
    weekly_hour = weekly_config.hour
    weekly_minute = weekly_config.minute
    lookback_days = weekly_config.lookback_days

    weekly_trigger = CronTrigger(
        day_of_week=weekly_day,
        hour=weekly_hour,
        minute=weekly_minute,
        timezone=TZ,
    )

    def run_weekly_analysis_with_flush() -> None:
        """Run weekly analysis and flush email errors at the end."""
        try:
            # Instantiate a fresh orchestrator
            job_config, job_news_config = load_config(config_key)
            orchestrator = PipelineOrchestrator(
                job_config,
                config_key=config_key,
                news_config=job_news_config,
            )

            results = orchestrator.run_weekly_analysis(days_back=lookback_days)

            # Save summary to DB
            summary_writer = SummaryWriter()
            summary_writer.save_analysis_summary(
                config_key=config_key,
                success=results.success,
                duration=results.duration,
                articles_analyzed=results.articles_count,
                stories_identified=getattr(results, "stories_count", 0),
                top_stories=getattr(results, "top_stories", []),
                errors=results.errors,
            )
        finally:
            email_error_handler.flush()

    scheduler.add_job(
        run_weekly_analysis_with_flush,
        trigger=weekly_trigger,
        id="weekly_analysis_job",
        name="Weekly News Analysis",
        replace_existing=True,
    )
    logger.info(
        f"Weekly analysis job scheduled: Every {weekly_day.upper()} at "
        f"{weekly_hour:02d}:{weekly_minute:02d} {TIMEZONE_STR}",
    )


if __name__ == "__main__":
    main()
