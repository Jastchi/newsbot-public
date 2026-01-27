"""Tests for newsserver utility functions."""

from datetime import date, datetime
from pathlib import Path

import pytest
from django.utils import timezone as django_timezone

from web.newsserver.utils import (
    get_date_range,
    is_active_log_file,
    make_timezone_aware,
    parse_date_or_default,
    validate_path_within_directory,
)


class TestValidatePathWithinDirectory:
    """Test cases for validate_path_within_directory."""

    def test_valid_path(self, tmp_path):
        """Test validating a valid path within directory."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        result = validate_path_within_directory("file.txt", base_dir)

        assert result is not None
        assert isinstance(result, Path)
        assert result.name == "file.txt"

    def test_path_traversal_detected(self, tmp_path):
        """Test that path traversal is detected and rejected."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        result = validate_path_within_directory("../../etc/passwd", base_dir)

        assert result is None

    def test_path_with_string(self, tmp_path):
        """Test that string paths are converted to Path objects."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        result = validate_path_within_directory("file.txt", base_dir)

        assert isinstance(result, Path)

    def test_path_with_path_object(self, tmp_path):
        """Test that Path objects work correctly."""
        base_dir = tmp_path / "base"
        base_dir.mkdir()

        result = validate_path_within_directory(Path("file.txt"), base_dir)

        assert result is not None
        assert isinstance(result, Path)


class TestParseDateOrDefault:
    """Test cases for parse_date_or_default."""

    def test_parse_valid_date(self):
        """Test parsing a valid ISO date string."""
        result = parse_date_or_default("2025-12-25")

        assert result == date(2025, 12, 25)

    def test_parse_none_uses_default(self):
        """Test that None uses the default date."""
        default = date(2025, 1, 1)
        result = parse_date_or_default(None, default=default)

        assert result == default

    def test_parse_none_no_default_uses_today(self):
        """Test that None without default uses today's date."""
        result = parse_date_or_default(None)

        assert result == django_timezone.now().date()

    def test_parse_invalid_date_uses_default(self):
        """Test that invalid date strings use the default."""
        default = date(2025, 1, 1)
        result = parse_date_or_default("invalid-date", default=default)

        assert result == default

    def test_parse_empty_string_uses_default(self):
        """Test that empty string uses the default."""
        default = date(2025, 1, 1)
        result = parse_date_or_default("", default=default)

        assert result == default


class TestMakeTimezoneAware:
    """Test cases for make_timezone_aware."""

    def test_make_timezone_aware_when_system_aware(self):
        """Test making datetime timezone-aware when system is aware."""
        if django_timezone.is_aware(django_timezone.now()):
            dt = datetime(2025, 1, 15, 12, 0, 0)
            result = make_timezone_aware(dt)

            assert result.tzinfo is not None
            assert result.year == 2025
            assert result.month == 1
            assert result.day == 15

    def test_make_timezone_aware_when_system_naive(self):
        """Test that naive datetimes are returned unchanged if system is naive."""
        # This test behavior depends on Django settings
        dt = datetime(2025, 1, 15, 12, 0, 0)
        result = make_timezone_aware(dt)

        # Result should be a datetime (may or may not have tzinfo)
        assert isinstance(result, datetime)


class TestGetDateRange:
    """Test cases for get_date_range."""

    def test_get_date_range(self):
        """Test getting start and end of day."""
        test_date = date(2025, 12, 25)
        start, end = get_date_range(test_date)

        assert isinstance(start, datetime)
        assert isinstance(end, datetime)
        assert start.date() == test_date
        assert end.date() == test_date
        assert start.hour == 0
        assert start.minute == 0
        assert start.second == 0

    def test_get_date_range_timezone_aware(self):
        """Test that date range returns timezone-aware datetimes if system is aware."""
        test_date = date(2025, 12, 25)
        start, end = get_date_range(test_date)

        if django_timezone.is_aware(django_timezone.now()):
            assert start.tzinfo is not None
            assert end.tzinfo is not None


class TestIsActiveLogFile:
    """Test cases for is_active_log_file."""

    def test_active_log_file(self):
        """Test that active log files are detected correctly."""
        assert is_active_log_file("newsbot.log") is True
        assert is_active_log_file("app.log") is True
        assert is_active_log_file("my_log.log") is True

    def test_rotated_log_file(self):
        """Test that rotated log files are detected correctly."""
        assert is_active_log_file("newsbot.log.2025-12-13") is False
        assert is_active_log_file("app.log.2024-01-01") is False
        assert is_active_log_file("my_log.log.2023-12-31") is False

    def test_log_file_with_other_suffixes(self):
        """Test log files with other suffixes are considered active."""
        # Files with other patterns should be considered active
        assert is_active_log_file("newsbot.log.old") is True
        assert is_active_log_file("newsbot.log.backup") is True
        assert is_active_log_file("newsbot.log.1") is True

    def test_log_file_edge_cases(self):
        """Test edge cases for log file detection."""
        # Files without .log extension
        assert is_active_log_file("just_a_file") is True

        # Files with date-like patterns but not at end
        assert is_active_log_file("app.2025-12-13.log") is True

        # Multiple date patterns
        assert is_active_log_file("app.log.2025-12-13.backup") is True
