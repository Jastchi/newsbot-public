"""Tests for newsserver services."""

from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from web.newsserver.datatypes import ConfigWithReports, LogFileInfo, ReportInfo
from web.newsserver.models import NewsConfig
from web.newsserver.services.config_service import ConfigService
from web.newsserver.services.log_service import LogService
from web.newsserver.services.report_service import ReportService


class TestConfigService:
    """Test cases for ConfigService."""

    @pytest.mark.django_db
    def test_get_config_by_key_exists(self):
        """Test getting an existing config by key."""
        config = NewsConfig.objects.create(
            key="test_config",
            display_name="Test Config",
            country="US",
            language="en",
            is_active=True,
        )

        result = ConfigService.get_config_by_key("test_config")

        assert result is not None
        assert result.key == "test_config"
        assert result.display_name == "Test Config"

    @pytest.mark.django_db
    def test_get_config_by_key_not_found(self):
        """Test getting a non-existent config returns None."""
        result = ConfigService.get_config_by_key("nonexistent")
        assert result is None

    @pytest.mark.django_db
    def test_get_config_by_key_inactive(self):
        """Test that inactive configs are not returned."""
        NewsConfig.objects.create(
            key="inactive_config",
            display_name="Inactive Config",
            country="US",
            language="en",
            is_active=False,
        )

        result = ConfigService.get_config_by_key("inactive_config")
        assert result is None

    @pytest.mark.django_db
    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.get_supabase_client")
    @patch("web.newsserver.services.config_service.list_supabase_reports")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_active_configs_with_reports_local(
        self,
        mock_settings,
        mock_list_reports,
        mock_get_client,
        mock_should_use,
        tmp_path,
    ):
        """Test getting configs with local reports."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        # Create test config directory with reports
        config_dir = reports_dir / "test_config"
        config_dir.mkdir()
        (config_dir / "report1.html").write_text("<html>Report 1</html>")
        (config_dir / "report2.html").write_text("<html>Report 2</html>")

        mock_should_use.return_value = False
        mock_get_client.return_value = None

        # Create config in database
        NewsConfig.objects.create(
            key="test_config",
            display_name="Test Config",
            country="US",
            language="en",
            is_active=True,
        )

        result = ConfigService.get_active_configs_with_reports()

        assert len(result) == 1
        assert isinstance(result[0], ConfigWithReports)
        assert result[0].key == "test_config"
        assert result[0].name == "Test Config"
        assert result[0].report_count == 2
        assert result[0].storage == "local"

    @pytest.mark.django_db
    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.get_supabase_client")
    @patch("web.newsserver.services.config_service.list_supabase_reports")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_active_configs_with_reports_supabase(
        self,
        mock_settings,
        mock_list_reports,
        mock_get_client,
        mock_should_use,
        tmp_path,
    ):
        """Test getting configs with Supabase reports."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        mock_list_reports.return_value = [
            {
                "name": "report1.html",
                "updated_at": "2025-12-10T12:00:00Z",
                "metadata": {"size": 1024},
            },
            {
                "name": "report2.html",
                "updated_at": "2025-12-11T12:00:00Z",
                "metadata": {"size": 2048},
            },
        ]

        # Create config in database
        NewsConfig.objects.create(
            key="test_config",
            display_name="Test Config",
            country="US",
            language="en",
            is_active=True,
        )

        result = ConfigService.get_active_configs_with_reports()

        assert len(result) == 1
        assert result[0].storage == "supabase"
        assert result[0].report_count == 2
        assert result[0].latest_report == "report2.html"  # Most recent

    @pytest.mark.django_db
    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_active_configs_no_reports(
        self,
        mock_settings,
        mock_should_use,
        tmp_path,
    ):
        """Test that configs without reports are not included."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use.return_value = False

        # Create config in database but no reports directory
        NewsConfig.objects.create(
            key="test_config",
            display_name="Test Config",
            country="US",
            language="en",
            is_active=True,
        )

        result = ConfigService.get_active_configs_with_reports()

        assert len(result) == 0


class TestReportService:
    """Test cases for ReportService."""

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_reports_for_config_local(
        self,
        mock_settings,
        mock_should_use,
        tmp_path,
    ):
        """Test getting reports from local storage."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use.return_value = False

        config_dir = reports_dir / "test_config"
        config_dir.mkdir()
        (config_dir / "report1.html").write_text("<html>Report 1</html>")
        (config_dir / "report2.html").write_text("<html>Report 2</html>")

        result = ReportService.get_reports_for_config("test_config")

        assert len(result) == 2
        assert all(isinstance(r, ReportInfo) for r in result)
        assert all(r.storage == "local" for r in result)
        assert result[0].filename in ["report1.html", "report2.html"]

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.list_supabase_reports")
    def test_get_reports_for_config_supabase(
        self,
        mock_list_reports,
        mock_get_client,
        mock_should_use,
    ):
        """Test getting reports from Supabase."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        mock_list_reports.return_value = [
            {
                "name": "report1.html",
                "updated_at": "2025-12-10T12:00:00Z",
                "metadata": {"size": 1024},
            },
            {
                "name": "report2.html",
                "updated_at": "2025-12-11T12:00:00Z",
                "metadata": {"size": 2048},
            },
        ]

        result = ReportService.get_reports_for_config("test_config")

        assert len(result) == 2
        assert all(isinstance(r, ReportInfo) for r in result)
        assert all(r.storage == "supabase" for r in result)
        # Should be sorted by date (newest first)
        assert result[0].filename == "report2.html"

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_reports_for_config_no_reports(
        self,
        mock_settings,
        mock_should_use,
        tmp_path,
    ):
        """Test getting reports when none exist."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use.return_value = False

        result = ReportService.get_reports_for_config("nonexistent")

        assert result == []

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_report_content_local(
        self,
        mock_settings,
        mock_should_use,
        tmp_path,
    ):
        """Test getting report content from local storage."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use.return_value = False

        config_dir = reports_dir / "test_config"
        config_dir.mkdir()
        (config_dir / "report.html").write_text("<html>Test Content</html>")

        result = ReportService.get_report_content("test_config", "report.html")

        assert result == "<html>Test Content</html>"

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.download_from_supabase")
    def test_get_report_content_supabase(
        self,
        mock_download,
        mock_get_client,
        mock_should_use,
    ):
        """Test getting report content from Supabase."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        mock_download.return_value = b"<html>Supabase Content</html>"

        result = ReportService.get_report_content("test_config", "report.html")

        assert result == "<html>Supabase Content</html>"

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_report_content_not_found(
        self,
        mock_settings,
        mock_should_use,
        tmp_path,
    ):
        """Test getting non-existent report returns None."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use.return_value = False

        result = ReportService.get_report_content("test_config", "nonexistent.html")

        assert result is None

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_download_report_local(
        self,
        mock_settings,
        mock_should_use,
        tmp_path,
    ):
        """Test downloading report from local storage."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use.return_value = False

        config_dir = reports_dir / "test_config"
        config_dir.mkdir()
        (config_dir / "report.html").write_text("<html>Test</html>")

        result = ReportService.download_report("test_config", "report.html")

        assert result is not None
        assert isinstance(result, bytes)
        assert b"<html>Test</html>" in result

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.download_from_supabase")
    def test_download_report_supabase(
        self,
        mock_download,
        mock_get_client,
        mock_should_use,
    ):
        """Test downloading report from Supabase."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        mock_download.return_value = b"<html>Supabase</html>"

        result = ReportService.download_report("test_config", "report.html")

        assert result == b"<html>Supabase</html>"


class TestLogService:
    """Test cases for LogService."""

    @patch("web.newsserver.services.log_service.settings")
    def test_get_log_files(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test getting log files."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        # Create test log files
        (logs_dir / "app.log").write_text("Active log\n")
        (logs_dir / "app.log.2025-12-20").write_text("Rotated log\n")

        result = LogService.get_log_files()

        assert len(result) == 2
        assert all(isinstance(log, LogFileInfo) for log in result)
        # Active log should be first
        assert result[0].filename == "app.log"
        assert result[0].is_active is True
        assert result[1].is_active is False

    @patch("web.newsserver.services.log_service.settings")
    def test_get_log_files_empty_dir(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test getting log files from empty directory."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        result = LogService.get_log_files()

        assert result == []

    @patch("web.newsserver.services.log_service.settings")
    def test_get_log_files_nonexistent_dir(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test getting log files when directory doesn't exist."""
        mock_settings.BASE_DIR = tmp_path

        result = LogService.get_log_files()

        assert result == []

    @patch("web.newsserver.services.log_service.settings")
    def test_get_log_content(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test getting log content."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        log_file = logs_dir / "app.log"
        log_file.write_text("Line 1\nLine 2\nLine 3\n")

        result = LogService.get_log_content("app.log")

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    @patch("web.newsserver.services.log_service.settings")
    def test_get_log_content_max_lines(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test that log content is truncated to max_lines."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        log_file = logs_dir / "app.log"
        with log_file.open("w") as f:
            for i in range(1500):
                f.write(f"Line {i}\n")

        result = LogService.get_log_content("app.log", max_lines=1000)

        lines = result.split("\n")
        # Should have approximately 1000 lines (plus newlines)
        assert len([l for l in lines if l.strip()]) <= 1000
        assert "Line 1499" in result  # Last line
        assert "Line 0" not in result  # First line removed

    @patch("web.newsserver.services.log_service.settings")
    def test_get_log_content_invalid_path(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test getting log content with invalid path."""
        mock_settings.BASE_DIR = tmp_path

        result = LogService.get_log_content("../../etc/passwd")

        assert "Error: Invalid log file path" in result

    @patch("web.newsserver.services.log_service.settings")
    def test_is_safe_log_path_valid(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test that valid log paths are safe."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        assert LogService.is_safe_log_path("app.log") is True
        assert LogService.is_safe_log_path("app.log.2025-12-20") is True

    @patch("web.newsserver.services.log_service.settings")
    def test_is_safe_log_path_traversal(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test that path traversal attempts are detected."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        assert LogService.is_safe_log_path("../../etc/passwd") is False
        assert LogService.is_safe_log_path("../other_dir/file.log") is False

    @patch("web.newsserver.services.log_service.settings")
    def test_validate_log_path_valid(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test validating a valid log path."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        log_file = logs_dir / "app.log"
        log_file.write_text("Test\n")

        result = LogService.validate_log_path("app.log")

        assert result is not None
        assert isinstance(result, Path)
        assert result.name == "app.log"

    @patch("web.newsserver.services.log_service.settings")
    def test_validate_log_path_nonexistent(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test validating a non-existent log path."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        result = LogService.validate_log_path("nonexistent.log")

        assert result is None

    @patch("web.newsserver.services.log_service.settings")
    def test_validate_log_path_traversal(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test that path traversal is rejected."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        result = LogService.validate_log_path("../../etc/passwd")

        assert result is None

    @patch("web.newsserver.services.log_service.settings")
    def test_get_config_tabs(
        self,
        mock_settings,
        tmp_path,
    ):
        """Test building config tabs from active logs."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        (logs_dir / "technology.log").write_text("Tech log\n")
        (logs_dir / "world_politics.log").write_text("Politics log\n")
        (logs_dir / "technology.log.2025-12-20").write_text("Old tech\n")

        result = LogService.get_config_tabs()

        assert len(result) == 2  # Only active logs
        assert all("name" in tab for tab in result)
        assert all("display_name" in tab for tab in result)
        assert all("filename" in tab for tab in result)
        # Should be sorted by display name
        assert result[0]["name"] == "technology"
        assert result[1]["name"] == "world_politics"

    def test_get_active_tab_for_log_active(self):
        """Test getting active tab for active log file."""
        result = LogService.get_active_tab_for_log("technology.log")
        assert result == "technology"

    def test_get_active_tab_for_log_rotated(self):
        """Test getting active tab for rotated log file."""
        result = LogService.get_active_tab_for_log("technology.log.2025-12-20")
        assert result == "technology"

    def test_can_stream_log_active(self):
        """Test that active logs can be streamed."""
        assert LogService.can_stream_log("app.log") is True

    def test_can_stream_log_rotated(self):
        """Test that rotated logs cannot be streamed."""
        assert LogService.can_stream_log("app.log.2025-12-20") is False
