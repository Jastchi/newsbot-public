"""Utility functions for NewsBot views and services."""

import re
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from django.utils import timezone as django_timezone

from newsbot.constants import TZ

if TYPE_CHECKING:
    from pytz import BaseTzInfo


def validate_path_within_directory(
    file_path: Path | str,
    base_directory: Path,
) -> Path | None:
    """
    Validate that a file path is within a base directory.

    Provides path traversal protection by ensuring the resolved path
    is relative to the base directory.

    Args:
        file_path: Path to validate (can be string or Path)
        base_directory: Base directory that the path must be within

    Returns:
        Resolved Path object if valid and safe, None otherwise

    Example:
        >>> base = Path("/var/logs")
        >>> validate_path_within_directory("app.log", base)
        Path("/var/logs/app.log")
        >>> validate_path_within_directory("../../etc/passwd", base)
        None

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    # Resolve the path to handle any relative components
    resolved_path = (base_directory / file_path).resolve()
    base_resolved = base_directory.resolve()

    # Ensure the resolved path is within the base directory
    if not resolved_path.is_relative_to(base_resolved):
        return None

    return resolved_path


def parse_date_or_default(
    date_str: str | None,
    default: date | None = None,
) -> date:
    """
    Parse a date string or return a default value.

    Args:
        date_str: ISO format date string (YYYY-MM-DD) or None
        default: Default date to return if parsing fails or date_str is
                 None. If None, uses today's date.

    Returns:
        Parsed date object

    Example:
        >>> parse_date_or_default("2025-01-15")
        date(2025, 1, 15)
        >>> parse_date_or_default("invalid", default=date(2025, 1, 1))
        date(2025, 1, 1)

    """
    if default is None:
        default = django_timezone.now().date()

    if not date_str:
        return default

    try:
        return date.fromisoformat(date_str)
    except ValueError:
        return default


def make_timezone_aware(
    dt: datetime,
    tz: "BaseTzInfo | None" = None,
) -> datetime:
    """
    Make a datetime timezone-aware if the system is timezone-aware.

    If the system is configured to use timezone-aware datetimes,
    this function will add timezone info. Otherwise, returns the
    datetime unchanged.

    Args:
        dt: Datetime to make timezone-aware
        tz: Timezone to use (defaults to TZ from constants)

    Returns:
        Timezone-aware datetime if system is timezone-aware,
        otherwise returns dt unchanged

    Example:
        >>> dt = datetime(2025, 1, 15, 12, 0, 0)
        >>> make_timezone_aware(dt)
        datetime(2025, 1, 15, 12, 0, 0, tzinfo=<DstTzInfo ...>)

    """
    if tz is None:
        tz = TZ

    # Only make timezone-aware if the system is configured for it
    if django_timezone.is_aware(django_timezone.now()):
        return dt.replace(tzinfo=tz)

    return dt


def get_date_range(
    selected_date: date,
) -> tuple[datetime, datetime]:
    """
    Get start and end of day as timezone-aware datetimes.

    Args:
        selected_date: Date to get range for

    Returns:
        Tuple of (start_of_day, end_of_day) as timezone-aware
        datetimes

    Example:
        >>> start, end = get_date_range(date(2025, 1, 15))
        >>> start
        datetime(2025, 1, 15, 0, 0, 0, tzinfo=<DstTzInfo ...>)
        >>> end
        datetime(2025, 1, 15, 23, 59, 59, 999999,
                 tzinfo=<DstTzInfo ...>)

    """
    start_of_day = datetime.combine(selected_date, datetime.min.time())
    end_of_day = datetime.combine(selected_date, datetime.max.time())

    # Make them timezone-aware if system is configured for it
    start_of_day = make_timezone_aware(start_of_day)
    end_of_day = make_timezone_aware(end_of_day)

    return start_of_day, end_of_day


def is_active_log_file(log_filename: str) -> bool:
    """
    Check if a log file is currently active (being written to).

    Active log files don't have a date suffix pattern like .YYYY-MM-DD.
    For example:
    - newsbot.log -> active
    - newsbot.log.2025-12-13 -> inactive (rotated)

    Args:
        log_filename: Name of the log file

    Returns:
        True if the log file is active, False otherwise

    Example:
        >>> is_active_log_file("newsbot.log")
        True
        >>> is_active_log_file("newsbot.log.2025-12-13")
        False

    """
    # Pattern matches rotated log files: filename.log.YYYY-MM-DD
    rotated_pattern = re.compile(r"\.\d{4}-\d{2}-\d{2}$")
    return not bool(rotated_pattern.search(log_filename))
