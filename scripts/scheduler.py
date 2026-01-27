"""
Scheduler service script for NewsBot.

This script runs as a service that:
1. Fetches schedules from the API at 00:05 daily (configurable)
2. Schedules tasks (daily scrape, weekly analysis) using APScheduler
3. Triggers API endpoints when tasks are due

Arguments:
    --host HOST         API server host (default: localhost)
    --port PORT         API server port (default: 8000)
    --refresh-hour H    Hour to refresh schedules (0-23, default: 0)
    --refresh-minute M  Minute to refresh schedules (0-59, default: 5)

Examples:
    # Run with default settings (localhost:8000)
    uv run python scripts/scheduler.py

    # Run with custom port
    uv run python scripts/scheduler.py --port 8080

    # Run with custom host and port
    uv run python scripts/scheduler.py --host api.example.com --port 443

    # Run with custom refresh time (1:30 AM in configured timezone)
    uv run python scripts/scheduler.py \
        --refresh-hour 1 --refresh-minute 30

"""

import argparse
import logging
from datetime import UTC, datetime

import requests
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from newsbot.constants import TIMEZONE_STR, TZ

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scheduler")

# Job ID prefix to identify scheduler-managed jobs
JOB_PREFIX = "scheduler_"
REFRESH_JOB_ID = f"{JOB_PREFIX}config_refresh"
# Minimum number of parts in job ID after splitting by "_"
MIN_JOB_ID_PARTS = 4


def make_api_request(
    method: str,
    url: str,
    *,
    timeout: int = 30,
    json_data: dict | None = None,
) -> dict | None:
    """
    Make an HTTP request to the API.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Full URL to request
        timeout: Request timeout in seconds
        json_data: Optional JSON body for POST requests

    Returns:
        Response JSON as dict, or None on error

    """
    try:
        response = requests.request(
            method,
            url,
            timeout=timeout,
            json=json_data,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        logger.warning(f"Connection error: Could not connect to {url}")
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout: Request to {url} timed out")
    except requests.exceptions.HTTPError as e:
        logger.warning(f"HTTP error {e.response.status_code}: {e}")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Request error: {e}")
    except ValueError as e:
        logger.warning(f"Invalid JSON response: {e}")
    return None


def trigger_daily_scrape(api_base_url: str, config_key: str) -> None:
    """
    Trigger daily scrape for a config via API.

    Args:
        api_base_url: Base URL of the API (e.g., http://localhost:8000)
        config_key: Configuration key to scrape

    """
    url = f"{api_base_url}/run/{config_key}"
    logger.info(f"Triggering daily scrape for '{config_key}'")

    result = make_api_request("POST", url, json_data={"force": False})
    if result:
        logger.info(f"Daily scrape triggered for '{config_key}': {result}")
    else:
        logger.error(f"Failed to trigger daily scrape for '{config_key}'")


def trigger_weekly_analysis(api_base_url: str, config_key: str) -> None:
    """
    Trigger weekly analysis for a config via API.

    Args:
        api_base_url: Base URL of the API (e.g., http://localhost:8000)
        config_key: Configuration key to analyze

    """
    url = f"{api_base_url}/analyze/{config_key}"
    logger.info(f"Triggering weekly analysis for '{config_key}'")

    result = make_api_request("POST", url, json_data={})
    if result:
        logger.info(f"Weekly analysis triggered for '{config_key}': {result}")
    else:
        logger.error(f"Failed to trigger weekly analysis for '{config_key}'")


def schedule_daily_scrape(
    scheduler: BlockingScheduler,
    api_base_url: str,
    config_key: str,
    hour: int,
    minute: int,
) -> None:
    """
    Schedule daily scrape job for a config.

    Args:
        scheduler: APScheduler instance
        api_base_url: Base URL of the API
        config_key: Configuration key
        hour: Hour to run (0-23)
        minute: Minute to run (0-59)

    """
    job_id = f"{JOB_PREFIX}daily_scrape_{config_key}"

    trigger = CronTrigger(
        hour=hour,
        minute=minute,
        timezone=TZ,
    )

    scheduler.add_job(
        trigger_daily_scrape,
        trigger=trigger,
        id=job_id,
        name=f"Daily Scrape: {config_key}",
        args=[api_base_url, config_key],
        replace_existing=True,
    )

    logger.info(
        f"Scheduled daily scrape for '{config_key}' at "
        f"{hour:02d}:{minute:02d} {TIMEZONE_STR}",
    )


def schedule_weekly_analysis(
    scheduler: BlockingScheduler,
    api_base_url: str,
    config_key: str,
    day_of_week: str,
    hour: int,
    minute: int,
) -> None:
    """
    Schedule weekly analysis job for a config.

    Args:
        scheduler: APScheduler instance
        api_base_url: Base URL of the API
        config_key: Configuration key
        day_of_week: Day of week (mon, tue, wed, thu, fri, sat, sun)
        hour: Hour to run (0-23)
        minute: Minute to run (0-59)

    """
    job_id = f"{JOB_PREFIX}weekly_analysis_{config_key}"

    trigger = CronTrigger(
        day_of_week=day_of_week.lower(),
        hour=hour,
        minute=minute,
        timezone=TZ,
    )

    scheduler.add_job(
        trigger_weekly_analysis,
        trigger=trigger,
        id=job_id,
        name=f"Weekly Analysis: {config_key}",
        args=[api_base_url, config_key],
        replace_existing=True,
    )

    logger.info(
        f"Scheduled weekly analysis for '{config_key}' on "
        f"{day_of_week.upper()} at {hour:02d}:{minute:02d} {TIMEZONE_STR}",
    )


def remove_stale_jobs(
    scheduler: BlockingScheduler,
    active_config_keys: set[str],
) -> None:
    """
    Remove jobs for configs that are no longer active.

    Args:
        scheduler: APScheduler instance
        active_config_keys: Set of currently active config keys

    """
    for job in scheduler.get_jobs():
        if not job.id.startswith(JOB_PREFIX):
            continue
        if job.id == REFRESH_JOB_ID:
            continue

        # Extract config key from job ID
        # Format: scheduler_daily_scrape_{config_key} or
        #         scheduler_weekly_analysis_{config_key}
        parts = job.id.split("_", 3)
        if len(parts) >= MIN_JOB_ID_PARTS:
            config_key = parts[3]
            if config_key not in active_config_keys:
                logger.info(f"Removing stale job: {job.id}")
                scheduler.remove_job(job.id)


def refresh_and_schedule_tasks(
    scheduler: BlockingScheduler,
    api_base_url: str,
) -> None:
    """
    Fetch schedules from API and update scheduled jobs.

    Args:
        scheduler: APScheduler instance
        api_base_url: Base URL of the API

    """
    logger.info("Refreshing schedules from API...")

    url = f"{api_base_url}/schedules"
    result = make_api_request("GET", url)

    if not result:
        logger.error("Failed to fetch schedules from API")
        return

    configs = result.get("configs", [])
    if not configs:
        logger.warning("No active configs found in schedules")
        return

    active_config_keys = set()

    for config in configs:
        config_key = config.get("key")
        if not config_key:
            logger.warning("Config missing 'key' field, skipping")
            continue

        # Skip inactive configs
        is_active = config.get("is_active", True)
        if not is_active:
            logger.info(
                f"Skipping inactive config '{config_key}' "
                "(is_active=False)",
            )
            continue

        active_config_keys.add(config_key)

        # Schedule daily scrape if enabled
        daily_scrape = config.get("daily_scrape", {})
        if daily_scrape.get("enabled"):
            schedule_daily_scrape(
                scheduler,
                api_base_url,
                config_key,
                daily_scrape.get("hour", 2),
                daily_scrape.get("minute", 0),
            )

        # Schedule weekly analysis if enabled
        weekly_analysis = config.get("weekly_analysis", {})
        if weekly_analysis.get("enabled"):
            schedule_weekly_analysis(
                scheduler,
                api_base_url,
                config_key,
                weekly_analysis.get("day_of_week", "mon"),
                weekly_analysis.get("hour", 9),
                weekly_analysis.get("minute", 0),
            )

    # Remove jobs for configs that are no longer active
    remove_stale_jobs(scheduler, active_config_keys)

    logger.info(
        f"Schedule refresh complete. "
        f"{len(active_config_keys)} active configs.",
    )


def print_scheduler_info(scheduler: BlockingScheduler) -> None:
    """Print information about scheduled jobs."""
    jobs = scheduler.get_jobs()

    print("\n" + "=" * 60)
    print("NEWSBOT SCHEDULER SERVICE")
    print("=" * 60)
    utc_now = datetime.now(UTC)
    print(f"Started at: {utc_now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Jobs scheduled: {len(jobs)}")
    print("-" * 60)

    for job in jobs:
        next_run = getattr(job, "next_run_time", None)
        next_run_str = (
            next_run.strftime("%Y-%m-%d %H:%M:%S %Z") if next_run else "N/A"
        )
        print(f"  {job.name}")
        print(f"    Next run: {next_run_str}")

    print("-" * 60)
    print("Press Ctrl+C to stop the scheduler")
    print("=" * 60 + "\n")


def main() -> None:
    """Entry point for the scheduler service."""
    parser = argparse.ArgumentParser(
        description="NewsBot Scheduler Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (localhost:8000)
  uv run python scripts/scheduler.py

  # Run with custom port
  uv run python scripts/scheduler.py --port 8080

  # Run with custom host and port
  uv run python scripts/scheduler.py --host api.example.com --port 443
        """,
    )

    parser.add_argument(
        "--host",
        default="localhost",
        help="API server host (default: localhost)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API server port (default: 8000)",
    )

    parser.add_argument(
        "--refresh-hour",
        type=int,
        default=0,
        help="Hour to refresh schedules (0-23, default: 0)",
    )

    parser.add_argument(
        "--refresh-minute",
        type=int,
        default=5,
        help="Minute to refresh schedules (0-59, default: 5)",
    )

    args = parser.parse_args()

    # Build API base URL
    # Use http for localhost, https otherwise (can be overridden)
    protocol = "http" if args.host in ("localhost", "127.0.0.1") else "https"
    api_base_url = f"{protocol}://{args.host}:{args.port}"

    logger.info(f"Scheduler starting with API at {api_base_url}")

    # Create scheduler
    scheduler = BlockingScheduler(timezone=TZ)

    # Schedule daily config refresh at specified time
    # (default: 00:05 {TIMEZONE_STR})
    refresh_trigger = CronTrigger(
        hour=args.refresh_hour,
        minute=args.refresh_minute,
        timezone=TZ,
    )

    scheduler.add_job(
        refresh_and_schedule_tasks,
        trigger=refresh_trigger,
        id=REFRESH_JOB_ID,
        name="Config Refresh",
        args=[scheduler, api_base_url],
        replace_existing=True,
    )

    logger.info(
        f"Config refresh scheduled at "
        f"{args.refresh_hour:02d}:{args.refresh_minute:02d} {TIMEZONE_STR}",
    )

    # Do an initial refresh to load schedules immediately
    logger.info("Performing initial schedule refresh...")
    refresh_and_schedule_tasks(scheduler, api_base_url)

    # Print scheduler info
    print_scheduler_info(scheduler)

    # Start scheduler
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\nShutting down scheduler...")
        logger.info("Scheduler stopped by user")


if __name__ == "__main__":
    main()
