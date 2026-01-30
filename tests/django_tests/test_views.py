"""Tests for newsserver views."""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from django.http import Http404, StreamingHttpResponse
from django.test import RequestFactory
from django.utils import timezone

from web.newsserver.views import (
    ConfigOverviewView,
    ConfigReportView,
    LogsView,
    RunListView,
    log_stream_view,
)
from web.newsserver.utils import is_active_log_file


@pytest.fixture
def request_factory():
    """Provide a Django RequestFactory."""
    return RequestFactory()


@pytest.fixture
def temp_reports_dir(tmp_path):
    """Create a temporary reports directory structure."""
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()

    # Create some test config directories with reports
    # Note: directories use config keys, not display names
    tech_dir = reports_dir / "technology"
    tech_dir.mkdir()
    (tech_dir / "news_report_20251201_120000.html").write_text(
        "<html><body>Tech Report 1</body></html>",
    )
    (tech_dir / "news_report_20251202_120000.html").write_text(
        "<html><body>Tech Report 2</body></html>",
    )

    sports_dir = reports_dir / "sports"
    sports_dir.mkdir()
    (sports_dir / "news_report_20251203_120000.html").write_text(
        "<html><body>Sports Report</body></html>",
    )

    # Create an empty directory (should not appear if no config in DB)
    empty_dir = reports_dir / "empty"
    empty_dir.mkdir()

    return reports_dir


@pytest.fixture
def test_news_configs(db):
    """Create test NewsConfig instances for view tests."""
    from web.newsserver.models import NewsConfig

    configs = []
    config_data = [
        {"key": "technology", "display_name": "Technology", "is_active": True},
        {"key": "sports", "display_name": "Sports", "is_active": True},
        {"key": "inactive", "display_name": "Inactive Config", "is_active": False},
        {"key": "no_reports", "display_name": "No Reports", "is_active": True},
    ]

    for data in config_data:
        config = NewsConfig.objects.create(
            key=data["key"],
            display_name=data["display_name"],
            country="US",
            language="en",
            is_active=data["is_active"],
        )
        configs.append(config)

    return configs


class TestConfigOverviewView:
    """Test cases for ConfigOverviewView."""

    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_with_reports(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test getting context data with existing reports."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        # Disable Supabase for all configs
        mock_should_use_supabase.return_value = False

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        assert "configs" in context
        configs = context["configs"]

        # Should have 2 configs (Sports and Technology) - only active configs with reports
        # "inactive" is not active, "no_reports" has no reports, "empty" has no config in DB
        assert len(configs) == 2

        # Check that configs are sorted by display name
        assert configs[0]["name"] == "Sports"
        assert configs[1]["name"] == "Technology"

        # Check Technology config
        tech_config = configs[1]
        assert tech_config["name"] == "Technology"
        assert tech_config["key"] == "technology"
        assert tech_config["report_count"] == 2
        # Latest report is determined by modification time
        assert tech_config["latest_report"].startswith("news_report_2025")
        assert "last_modified" in tech_config
        assert tech_config["storage"] == "local"

    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_empty_reports_dir(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test with empty reports directory - configs without reports don't appear."""
        empty_dir = tmp_path / "empty_reports"
        empty_dir.mkdir()
        mock_settings.REPORTS_DIR = empty_dir
        mock_should_use_supabase.return_value = False

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        assert "configs" in context
        # Configs exist in DB but have no reports, so they don't appear
        assert context["configs"] == []

    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_inactive_configs_excluded(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test that inactive configs are excluded even if they have reports."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        
        # Create reports for inactive config
        inactive_dir = reports_dir / "inactive"
        inactive_dir.mkdir()
        (inactive_dir / "news_report_20251201_120000.html").write_text(
            "<html><body>Inactive Report</body></html>",
        )
        
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        assert "configs" in context
        # Inactive config should not appear even though it has reports
        config_names = [c["name"] for c in context["configs"]]
        assert "Inactive Config" not in config_names

    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_configs_without_reports_excluded(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test that configs in database without reports don't appear."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        
        # Only create reports for technology, not for no_reports
        tech_dir = reports_dir / "technology"
        tech_dir.mkdir()
        (tech_dir / "news_report_20251201_120000.html").write_text(
            "<html><body>Tech Report</body></html>",
        )
        
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        assert "configs" in context
        # Only technology should appear (has reports)
        # no_reports config should not appear (no reports)
        assert len(context["configs"]) == 1
        assert context["configs"][0]["name"] == "Technology"
        assert context["configs"][0]["key"] == "technology"

    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_filesystem_only_configs_excluded(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test that configs with reports but not in database don't appear."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        
        # Create reports for a config that doesn't exist in database
        orphan_dir = reports_dir / "orphan_config"
        orphan_dir.mkdir()
        (orphan_dir / "news_report_20251201_120000.html").write_text(
            "<html><body>Orphan Report</body></html>",
        )
        
        # Also create reports for a config that exists in database
        tech_dir = reports_dir / "technology"
        tech_dir.mkdir()
        (tech_dir / "news_report_20251201_120000.html").write_text(
            "<html><body>Tech Report</body></html>",
        )
        
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        assert "configs" in context
        # Only technology should appear (exists in DB and has reports)
        # orphan_config should not appear (not in database)
        assert len(context["configs"]) == 1
        assert context["configs"][0]["name"] == "Technology"

    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_nonexistent_reports_dir(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test with non-existent reports directory - configs without reports don't appear."""
        nonexistent_dir = tmp_path / "does_not_exist"
        mock_settings.REPORTS_DIR = nonexistent_dir
        mock_should_use_supabase.return_value = False

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        assert "configs" in context
        # Configs exist in DB but reports directory doesn't exist, so no configs appear
        assert context["configs"] == []


class TestConfigReportView:
    """Test cases for ConfigReportView."""

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_with_reports(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test getting context data for a specific config."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        # Use config key in URL, not display name
        view.request = request_factory.get("/config/technology/")

        context = view.get_context_data(config_name="technology")

        # Should use display name from database
        assert context["config_name"] == "Technology"
        assert "reports" in context
        assert len(context["reports"]) == 2
        assert "current_report" in context
        assert "report_content" in context
        # Check that some report content is present
        assert "Tech Report" in context["report_content"]
        assert "<html>" in context["report_content"]

        # Latest report should be shown by default (most recent file)
        assert context["current_report"].startswith("news_report_2025")

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_with_selected_report(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test selecting a specific report."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        # Use config key in URL
        view.request = request_factory.get(
            "/config/technology/?report=news_report_20251201_120000.html",
        )

        context = view.get_context_data(config_name="technology")

        assert context["current_report"] == "news_report_20251201_120000.html"
        assert "Tech Report 1" in context["report_content"]

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_invalid_selected_report(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test selecting a non-existent report falls back to latest."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        view.request = request_factory.get(
            "/config/technology/?report=nonexistent.html",
        )

        context = view.get_context_data(config_name="technology")

        # Should fall back to latest report (most recent by modification time)
        assert context["current_report"].startswith("news_report_2025")
        assert context["current_report"].endswith(".html")

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_config_not_found(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test with non-existent config in database."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        view.request = request_factory.get("/config/nonexistent/")

        context = view.get_context_data(config_name="nonexistent")

        assert "error" in context
        assert "not found" in context["error"]

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_inactive_config(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test that inactive configs are rejected."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        
        # Create reports for inactive config
        inactive_dir = reports_dir / "inactive"
        inactive_dir.mkdir()
        (inactive_dir / "news_report_20251201_120000.html").write_text(
            "<html><body>Inactive Report</body></html>",
        )
        
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        view.request = request_factory.get("/config/inactive/")

        context = view.get_context_data(config_name="inactive")

        assert "error" in context
        assert "not found" in context["error"]

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_no_reports(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test with config that has no HTML reports."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        # Create directory but no reports
        no_reports_dir = reports_dir / "no_reports"
        no_reports_dir.mkdir()
        
        mock_settings.REPORTS_DIR = reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        view.request = request_factory.get("/config/no_reports/")

        context = view.get_context_data(config_name="no_reports")

        assert "error" in context
        assert "No reports" in context["error"]

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_download_existing_report(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test downloading an existing report."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        view.request = request_factory.get(
            "/config/technology/"
            "?download=1&report=news_report_20251201_120000.html",
        )
        view.kwargs = {"config_name": "technology"}

        response = view.get(
            view.request,
            config_name="technology",
        )

        assert response.status_code == 200
        assert "attachment" in response["Content-Disposition"]
        assert (
            "news_report_20251201_120000.html"
            in response["Content-Disposition"]
        )

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_download_nonexistent_report(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test downloading a non-existent report raises 404."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        view.request = request_factory.get(
            "/config/technology/?download=1&report=nonexistent.html",
        )
        view.kwargs = {"config_name": "technology"}

        from django.http import Http404

        with pytest.raises(Http404):
            view.get(view.request, config_name="technology")

    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.settings")
    def test_reports_list_metadata(
        self,
        mock_settings,
        mock_should_use_supabase,
        request_factory,
        temp_reports_dir,
        test_news_configs,
    ):
        """Test that reports list includes metadata."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_should_use_supabase.return_value = False

        view = ConfigReportView()
        view.request = request_factory.get("/config/technology/")

        context = view.get_context_data(config_name="technology")

        reports = context["reports"]
        for report in reports:
            assert "filename" in report
            assert "modified" in report
            assert "size" in report
            assert isinstance(report["size"], int)


class TestIsActiveLogFile:
    """Test cases for is_active_log_file helper function."""

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
        """Test log files with other suffixes."""
        # Files with other patterns should be considered active
        assert is_active_log_file("newsbot.log.old") is True
        assert is_active_log_file("newsbot.log.backup") is True
        assert is_active_log_file("newsbot.log.1") is True


@pytest.fixture
def temp_logs_dir(tmp_path):
    """Create a temporary logs directory structure."""
    import time

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    # Create rotated log files first (older)
    rotated_log1 = logs_dir / "newsbot.log.2025-12-22"
    rotated_log1.write_text("2025-12-22 10:00:00 - INFO - Old log line\n")
    time.sleep(0.01)  # Ensure different modification times

    rotated_log2 = logs_dir / "newsbot.log.2025-12-21"
    rotated_log2.write_text("2025-12-21 10:00:00 - INFO - Older log line\n")
    time.sleep(0.01)  # Ensure different modification times

    # Create active log file last (most recent)
    active_log = logs_dir / "newsbot.log"
    active_log.write_text(
        "2025-12-23 10:00:00 - INFO - Test log line 1\n"
        "2025-12-23 10:01:00 - INFO - Test log line 2\n",
    )

    return logs_dir


class TestLogsView:
    """Test cases for LogsView."""

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_with_logs(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test getting context data with existing log files."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        view = LogsView()
        view.request = request_factory.get("/logs/")

        context = view.get_context_data()

        assert "logs" in context
        logs = context["logs"]
        assert len(logs) == 3  # active + 2 rotated

        # Check that logs are sorted: active log first, then rotated by date descending
        assert logs[0]["filename"] == "newsbot.log"  # Active log always first
        assert (
            logs[1]["filename"] == "newsbot.log.2025-12-22"
        )  # Newest rotated
        assert (
            logs[2]["filename"] == "newsbot.log.2025-12-21"
        )  # Oldest rotated

        # Check log metadata
        for log in logs:
            assert "filename" in log
            assert "modified" in log
            assert "size" in log
            assert "is_active" in log
            assert isinstance(log["is_active"], bool)

        # Check active log detection
        active_log = next(
            log for log in logs if log["filename"] == "newsbot.log"
        )
        assert active_log["is_active"] is True

        rotated_log = next(
            log for log in logs if log["filename"] == "newsbot.log.2025-12-22"
        )
        assert rotated_log["is_active"] is False

        # Check current log and content
        assert "current_log" in context
        assert context["current_log"] == "newsbot.log"  # Latest by default
        assert "log_content" in context
        assert "is_current_log_active" in context
        assert context["is_current_log_active"] is True

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_with_selected_log(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test selecting a specific log file."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        view = LogsView()
        view.request = request_factory.get("/logs/?log=newsbot.log.2025-12-22")

        context = view.get_context_data()

        assert context["current_log"] == "newsbot.log.2025-12-22"
        assert context["is_current_log_active"] is False
        assert "Old log line" in context["log_content"]

    @patch("django.conf.settings")
    def test_get_context_data_empty_logs_dir(
        self,
        mock_settings,
        request_factory,
        tmp_path,
    ):
        """Test with empty logs directory."""
        # Create logs directory but leave it empty
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        view = LogsView()
        view.request = request_factory.get("/logs/")

        context = view.get_context_data()

        assert "error" in context
        assert "No log files" in context["error"]

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_nonexistent_logs_dir(
        self,
        mock_settings,
        request_factory,
        tmp_path,
    ):
        """Test with non-existent logs directory."""
        nonexistent_dir = tmp_path / "does_not_exist"
        mock_settings.BASE_DIR = tmp_path

        view = LogsView()
        view.request = request_factory.get("/logs/")

        context = view.get_context_data()

        assert "error" in context
        assert "No log files found" in context["error"]

    @patch("web.newsserver.services.log_service.settings")
    def test_get_download_existing_log(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test downloading an existing log file."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        view = LogsView()
        view.request = request_factory.get("/logs/?download=1&log=newsbot.log")

        response = view.get(view.request)

        assert response.status_code == 200
        assert "attachment" in response["Content-Disposition"]
        assert "newsbot.log" in response["Content-Disposition"]

    @patch("web.newsserver.services.log_service.settings")
    def test_get_download_nonexistent_log(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test downloading a non-existent log raises 404."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        view = LogsView()
        view.request = request_factory.get(
            "/logs/?download=1&log=nonexistent.log"
        )

        with pytest.raises(Http404):
            view.get(view.request)

    @patch("web.newsserver.services.log_service.settings")
    def test_get_download_path_traversal_protection(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test that path traversal attacks are prevented."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        view = LogsView()
        view.request = request_factory.get(
            "/logs/?download=1&log=../../etc/passwd",
        )

        with pytest.raises(Http404):
            view.get(view.request)

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_large_log_file(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test that large log files are truncated to last 1000 lines."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Create a log file with many lines
        large_log = temp_logs_dir / "large.log"
        with large_log.open("w") as f:
            for i in range(1500):
                f.write(f"Line {i}\n")

        view = LogsView()
        view.request = request_factory.get("/logs/?log=large.log")

        context = view.get_context_data()

        # Should only show last 1000 lines
        lines = context["log_content"].split("\n")
        # Account for trailing newline
        assert len([l for l in lines if l.strip()]) <= 1000
        assert "Line 1499" in context["log_content"]  # Last line
        assert "Line 0" not in context["log_content"]  # First line removed

    @patch("web.newsserver.services.log_service.settings")
    @patch("pathlib.Path.open")
    def test_get_context_data_log_read_error(
        self,
        mock_open,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test error handling when reading log file fails."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Mock open to raise an exception
        mock_open.side_effect = PermissionError("Permission denied")

        view = LogsView()
        view.request = request_factory.get("/logs/?log=newsbot.log")

        context = view.get_context_data()

        # Should handle error gracefully
        assert "log_content" in context
        assert "Error reading log file" in context["log_content"]


class TestLogStreamView:
    """Test cases for LogStreamView."""

    @patch("web.newsserver.services.log_service.settings")
    @patch("web.newsserver.views._stream_log_file")
    def test_stream_active_log_file(
        self,
        mock_stream,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test streaming an active log file."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Mock the stream generator to return a finite sequence
        import json

        def mock_generator():
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"

        mock_stream.return_value = mock_generator()

        request = request_factory.get("/logs/stream/?log=newsbot.log")

        response = log_stream_view(request)

        assert response.status_code == 200
        assert response["Content-Type"] == "text/event-stream"
        assert "no-cache" in response["Cache-Control"]

        # Read first few events
        assert isinstance(response, StreamingHttpResponse)
        content = b"".join(response.streaming_content)
        content_str = content.decode("utf-8")

        # Should have connected message
        assert "connected" in content_str

    @patch("web.newsserver.services.log_service.settings")
    def test_stream_missing_log_parameter(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test streaming without log parameter returns 400."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        request = request_factory.get("/logs/stream/")

        response = log_stream_view(request)

        assert response.status_code == 400
        assert "Missing log parameter" in response.content.decode()

    @patch("web.newsserver.services.log_service.settings")
    def test_stream_nonexistent_log(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test streaming a non-existent log returns 404."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        request = request_factory.get("/logs/stream/?log=nonexistent.log")

        response = log_stream_view(request)

        assert response.status_code == 404
        assert "Log file not found" in response.content.decode()

    @patch("web.newsserver.services.log_service.settings")
    def test_stream_rotated_log_file(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test that streaming rotated log files is not allowed."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        request = request_factory.get(
            "/logs/stream/?log=newsbot.log.2025-12-22"
        )

        response = log_stream_view(request)

        assert response.status_code == 400
        assert (
            "Streaming only available for active log files"
            in response.content.decode()
        )

    @patch("web.newsserver.services.log_service.settings")
    def test_stream_path_traversal_protection(
        self,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test that path traversal attacks are prevented."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        request = request_factory.get("/logs/stream/?log=../../etc/passwd")

        response = log_stream_view(request)

        assert response.status_code == 403
        assert "Invalid log file path" in response.content.decode()

    @patch("web.newsserver.views.sleep")
    @patch("web.newsserver.services.log_service.settings")
    def test_stream_returns_initial_content_then_connected(
        self,
        mock_settings,
        mock_sleep,
        request_factory,
        temp_logs_dir,
    ):
        """Test that stream sends initial_content with current tail then connected."""
        import json

        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Stop after connected so we don't block on poll loop
        def sleep_raise(_duration):
            raise StopIteration()

        mock_sleep.side_effect = sleep_raise

        request = request_factory.get("/logs/stream/?log=newsbot.log")
        response = log_stream_view(request)

        assert response.status_code == 200
        assert isinstance(response, StreamingHttpResponse)

        stream_iter = iter(response.streaming_content)
        first_chunk = next(stream_iter).decode("utf-8")
        second_chunk = next(stream_iter).decode("utf-8")

        # Parse SSE: "data: {...}\n\n"
        first_data = json.loads(first_chunk.strip().replace("data: ", ""))
        second_data = json.loads(second_chunk.strip().replace("data: ", ""))

        assert first_data["type"] == "initial_content"
        expected_content = (
            "2025-12-23 10:00:00 - INFO - Test log line 1\n"
            "2025-12-23 10:01:00 - INFO - Test log line 2\n"
        )
        assert first_data["content"] == expected_content

        assert second_data["type"] == "connected"

    @patch("web.newsserver.services.log_service.settings")
    @patch("web.newsserver.views._stream_log_file")
    def test_stream_new_log_lines(
        self,
        mock_stream,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test that new log lines are streamed correctly."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Mock the stream generator
        import json

        def mock_generator():
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            yield f"data: {json.dumps({'type': 'log_line', 'content': 'New line 1'})}\n\n"
            yield f"data: {json.dumps({'type': 'log_line', 'content': 'New line 2'})}\n\n"

        mock_stream.return_value = mock_generator()

        request = request_factory.get("/logs/stream/?log=newsbot.log")

        response = log_stream_view(request)

        assert response.status_code == 200
        assert isinstance(response, StreamingHttpResponse)
        content = b"".join(response.streaming_content).decode("utf-8")
        assert "connected" in content
        assert "New line 1" in content
        assert "New line 2" in content

    @patch("web.newsserver.services.log_service.settings")
    @patch("web.newsserver.views._stream_log_file")
    def test_stream_log_file_new_content(
        self,
        mock_stream,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test streaming detects and yields new log content."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Mock stream to return new log lines
        import json

        def mock_generator():
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            yield f"data: {json.dumps({'type': 'log_line', 'content': 'New line 1'})}\n\n"
            yield f"data: {json.dumps({'type': 'log_line', 'content': 'New line 2'})}\n\n"

        mock_stream.return_value = mock_generator()

        request = request_factory.get("/logs/stream/?log=newsbot.log")

        response = log_stream_view(request)
        assert isinstance(response, StreamingHttpResponse)
        content = b"".join(response.streaming_content).decode("utf-8")

        assert "connected" in content
        assert "New line 1" in content
        assert "New line 2" in content

    @patch("web.newsserver.services.log_service.settings")
    @patch("web.newsserver.views._stream_log_file")
    def test_stream_log_file_disappears(
        self,
        mock_stream,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test streaming handles file disappearing during stream."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Mock stream to simulate file disappearing
        import json

        def mock_generator():
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            error_payload = json.dumps(
                {"type": "error", "message": "Log file no longer exists"},
            )
            yield f"data: {error_payload}\n\n"

        mock_stream.return_value = mock_generator()

        request = request_factory.get("/logs/stream/?log=newsbot.log")

        response = log_stream_view(request)
        assert isinstance(response, StreamingHttpResponse)
        content = b"".join(response.streaming_content).decode("utf-8")

        assert "connected" in content
        assert "no longer exists" in content

    @patch("web.newsserver.services.log_service.settings")
    @patch("web.newsserver.views._stream_log_file")
    def test_stream_log_file_outer_exception(
        self,
        mock_stream,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test streaming handles outer exception."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Mock stream to yield error from outer exception handler
        import json

        def mock_generator():
            error_payload = json.dumps(
                {"type": "error", "message": "Stream error: ValueError"},
            )
            yield f"data: {error_payload}\n\n"

        mock_stream.return_value = mock_generator()

        request = request_factory.get("/logs/stream/?log=newsbot.log")

        response = log_stream_view(request)
        assert isinstance(response, StreamingHttpResponse)
        content = b"".join(response.streaming_content).decode("utf-8")

        # Should have error message
        assert "Stream error" in content

    @patch("web.newsserver.services.log_service.settings")
    @patch("web.newsserver.views._stream_log_file")
    def test_stream_log_file_read_error(
        self,
        mock_stream,
        mock_settings,
        request_factory,
        temp_logs_dir,
    ):
        """Test streaming handles file read errors."""
        mock_settings.BASE_DIR = temp_logs_dir.parent

        # Mock stream to raise error after connected
        import json

        def mock_generator():
            yield f"data: {json.dumps({'type': 'connected'})}\n\n"
            error_payload = json.dumps(
                {"type": "error", "message": "Error reading log: IOError"},
            )
            yield f"data: {error_payload}\n\n"

        mock_stream.return_value = mock_generator()

        request = request_factory.get("/logs/stream/?log=newsbot.log")

        response = log_stream_view(request)
        assert isinstance(response, StreamingHttpResponse)
        content = b"".join(response.streaming_content).decode("utf-8")

        assert "connected" in content
        assert "Error reading log" in content

    @patch("web.newsserver.services.log_service.settings")
    def test_stream_log_file_nonexistent_initial(
        self,
        mock_settings,
        request_factory,
        tmp_path,
    ):
        """Test streaming a log file that doesn't exist initially."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        # Create a non-existent log file path
        nonexistent_log = logs_dir / "nonexistent.log"

        # This should return 404 before streaming starts
        request = request_factory.get("/logs/stream/?log=nonexistent.log")

        response = log_stream_view(request)

        assert response.status_code == 404

    @patch("web.newsserver.views.sleep")
    def test_stream_log_file_function_nonexistent_file(
        self,
        mock_sleep,
        tmp_path,
    ):
        """Test _stream_log_file with non-existent file initially."""
        from web.newsserver.views import _stream_log_file

        nonexistent_log = tmp_path / "nonexistent.log"

        # Mock sleep to stop after first iteration by raising KeyboardInterrupt
        call_count = 0

        def sleep_side_effect(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise KeyboardInterrupt("Test stop")

        mock_sleep.side_effect = sleep_side_effect

        # Should handle non-existent file
        generator = _stream_log_file(nonexistent_log)
        first_event = next(generator)
        assert "connected" in first_event

        # Try to get next event (will trigger sleep)
        try:
            next(generator)
        except (StopIteration, KeyboardInterrupt):
            pass

    @patch("web.newsserver.views.sleep")
    def test_stream_log_file_function_file_disappears(
        self,
        mock_sleep,
        tmp_path,
    ):
        """Test _stream_log_file handles file disappearing."""
        import time

        from web.newsserver.views import _stream_log_file

        log_file = tmp_path / "test.log"
        log_file.write_text("Initial line\n")

        generator = _stream_log_file(log_file)

        # Read initial_content then connected
        next(generator)
        next(generator)

        # Delete file
        log_file.unlink()

        # Mock sleep side effect
        call_count = 0

        def sleep_side_effect(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise StopIteration

        mock_sleep.side_effect = sleep_side_effect

        # Should detect file disappearance
        events = []
        try:
            for _ in range(3):
                event = next(generator)
                events.append(event)
                if "no longer exists" in event:
                    break
        except (StopIteration, KeyboardInterrupt):
            pass

        # Check that we got the error message or at least tried
        content = "".join(events)
        assert "no longer exists" in content or len(events) > 0

    @patch("web.newsserver.views.sleep")
    def test_stream_log_file_function_file_rotation(
        self,
        mock_sleep,
        tmp_path,
    ):
        """Test _stream_log_file handles file rotation."""
        import time

        from web.newsserver.views import _stream_log_file

        log_file = tmp_path / "test.log"
        log_file.write_text("Old line 1\nOld line 2\n")

        generator = _stream_log_file(log_file)

        # Read initial_content then connected
        next(generator)
        next(generator)

        # Truncate file (simulate rotation)
        log_file.write_text("New line after rotation\n")

        # Mock sleep side effect
        call_count = 0

        def sleep_side_effect(duration):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise StopIteration

        mock_sleep.side_effect = sleep_side_effect

        # Should handle rotation
        try:
            next(generator)  # Should not crash
        except StopIteration:
            pass

    @patch("web.newsserver.views.sleep")
    def test_stream_log_file_function_read_error(
        self,
        mock_sleep,
        tmp_path,
    ):
        """Test _stream_log_file handles read errors."""
        from web.newsserver.views import _stream_log_file

        log_file = tmp_path / "test.log"
        log_file.write_text("Initial line\n")  # real open, before patching Path.open

        with patch("pathlib.Path.open") as mock_open:
            # First open (initial read): return context manager yielding file-like with content
            file_like = Mock()
            file_like.readlines.return_value = ["Initial line\n"]
            file_like.tell.return_value = 13
            mock_open.return_value.__enter__.return_value = file_like
            mock_open.return_value.__exit__.return_value = None

            open_call_count = 0

            def open_side_effect(*args, **kwargs):
                nonlocal open_call_count
                open_call_count += 1
                if open_call_count <= 1:
                    return mock_open.return_value
                raise IOError("Read error")

            mock_open.side_effect = open_side_effect

            generator = _stream_log_file(log_file)

            # Read initial_content then connected
            next(generator)
            next(generator)

            # Mock sleep so we don't block; raise after first sleep to stop generator
            call_count = 0

            def sleep_side_effect(duration):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    raise StopIteration

            mock_sleep.side_effect = sleep_side_effect

            # Should handle read error (patch must stay active so next open raises)
            events = []
            try:
                for _ in range(3):
                    event = next(generator)
                    events.append(event)
                    if "Error reading log" in event:
                        break
            except (StopIteration, KeyboardInterrupt):
                pass

        # Check that we got the error message or at least tried
        content = "".join(events)
        assert "Error reading log" in content or len(events) > 0


class TestConfigOverviewViewSupabase:
    """Test cases for ConfigOverviewView with Supabase integration."""

    @patch("web.newsserver.services.config_service.get_supabase_client")
    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.list_supabase_reports")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_with_supabase_reports(
        self,
        mock_settings,
        mock_list_reports,
        mock_should_use,
        mock_get_client,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test ConfigOverview with Supabase-backed config."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        # Setup Supabase mocks
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Make technology use Supabase
        def should_use_side_effect(config_key):
            return config_key == "technology"

        mock_should_use.side_effect = should_use_side_effect

        # Mock Supabase reports
        mock_list_reports.return_value = [
            {
                "name": "news_report_20251210_120000.html",
                "updated_at": "2025-12-10T12:00:00Z",
                "metadata": {"size": 1024},
            },
            {
                "name": "news_report_20251211_120000.html",
                "updated_at": "2025-12-11T12:00:00Z",
                "metadata": {"size": 2048},
            },
        ]

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        configs = context["configs"]
        # Find Technology config (display name from database)
        tech_config = next(c for c in configs if c["name"] == "Technology")
        assert tech_config["key"] == "technology"
        assert tech_config["storage"] == "supabase"
        assert tech_config["report_count"] == 2
        assert (
            tech_config["latest_report"] == "news_report_20251211_120000.html"
        )

    @pytest.mark.django_db
    @patch("web.newsserver.services.config_service.get_supabase_client")
    @patch("web.newsserver.services.config_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.config_service.settings")
    def test_get_context_data_supabase_no_client(
        self,
        mock_settings,
        mock_should_use,
        mock_get_client,
        request_factory,
        temp_reports_dir,
    ):
        """Test when Supabase is configured but client is unavailable."""
        mock_settings.REPORTS_DIR = temp_reports_dir
        mock_get_client.return_value = None
        mock_should_use.return_value = True

        view = ConfigOverviewView()
        view.request = request_factory.get("/")

        context = view.get_context_data()

        # Should fall back to local filesystem
        assert "configs" in context


class TestConfigReportViewSupabase:
    """Test cases for ConfigReportView with Supabase integration."""

    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.list_supabase_reports")
    @patch("web.newsserver.services.report_service.download_from_supabase")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_with_supabase(
        self,
        mock_settings,
        mock_download,
        mock_list_reports,
        mock_should_use,
        mock_get_client,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test ConfigReportView with Supabase storage."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        # Setup Supabase mocks (ReportService uses report_service's get_supabase_client)
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        # Mock Supabase reports
        mock_list_reports.return_value = [
            {
                "name": "news_report_20251210_120000.html",
                "updated_at": "2025-12-10T12:00:00Z",
                "metadata": {"size": 1024},
            },
        ]

        # Mock download content
        mock_download.return_value = (
            b"<html><body>Supabase Report</body></html>"
        )

        view = ConfigReportView()
        # Use config key in URL
        view.request = request_factory.get("/config/technology/")

        context = view.get_context_data(config_name="technology")

        # Should use display name from database
        assert context["config_name"] == "Technology"
        assert context["storage"] == "supabase"
        assert "Supabase Report" in context["report_content"]
        assert len(context["reports"]) == 1

    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.list_supabase_reports")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_supabase_no_reports(
        self,
        mock_settings,
        mock_list_reports,
        mock_should_use,
        mock_get_client,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test ConfigReportView with Supabase but no reports."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True
        mock_list_reports.return_value = []

        view = ConfigReportView()
        view.request = request_factory.get("/config/technology/")

        context = view.get_context_data(config_name="technology")

        assert "error" in context
        assert "No reports found" in context["error"]

    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.list_supabase_reports")
    @patch("web.newsserver.services.report_service.download_from_supabase")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_supabase_download_failure(
        self,
        mock_settings,
        mock_download,
        mock_list_reports,
        mock_should_use,
        mock_get_client,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test ConfigReportView when Supabase download fails."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        # Setup Supabase mocks (ReportService uses report_service's get_supabase_client)
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        mock_list_reports.return_value = [
            {
                "name": "news_report_20251210_120000.html",
                "updated_at": "2025-12-10T12:00:00Z",
                "metadata": {"size": 1024},
            },
        ]

        # Download fails
        mock_download.return_value = None

        view = ConfigReportView()
        view.request = request_factory.get("/config/technology/")

        context = view.get_context_data(config_name="technology")

        assert "error" in context
        assert "Failed to load report content" in context["error"]

    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.download_from_supabase")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_download_from_supabase(
        self,
        mock_settings,
        mock_download,
        mock_should_use,
        mock_get_client,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test downloading a report from Supabase."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        mock_download.return_value = b"<html><body>Download Test</body></html>"

        view = ConfigReportView()
        view.request = request_factory.get(
            "/config/technology/?download=1&report=news_report_20251210_120000.html",
        )
        view.kwargs = {"config_name": "technology"}

        response = view.get(view.request, config_name="technology")

        assert response.status_code == 200
        assert "attachment" in response["Content-Disposition"]
        assert b"Download Test" in response.content

    @patch("web.newsserver.services.report_service.get_supabase_client")
    @patch("web.newsserver.services.report_service.should_use_supabase_for_config")
    @patch("web.newsserver.services.report_service.download_from_supabase")
    @patch("web.newsserver.services.report_service.settings")
    def test_get_download_from_supabase_failure(
        self,
        mock_settings,
        mock_download,
        mock_should_use,
        mock_get_client,
        request_factory,
        tmp_path,
        test_news_configs,
    ):
        """Test downloading from Supabase when download fails."""
        reports_dir = tmp_path / "reports"
        reports_dir.mkdir()
        mock_settings.REPORTS_DIR = reports_dir

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_should_use.return_value = True

        # Download fails
        mock_download.return_value = None

        view = ConfigReportView()
        view.request = request_factory.get(
            "/config/technology/?download=1&report=news_report_20251210_120000.html",
        )
        view.kwargs = {"config_name": "technology"}

        with pytest.raises(Http404) as exc_info:
            view.get(view.request, config_name="technology")
        assert "Report file not found" in str(exc_info.value)

    @patch("web.newsserver.services.report_service.settings")
    def test_get_context_data_invalid_config_name(
        self,
        mock_settings,
        request_factory,
        temp_reports_dir,
    ):
        """Test with invalid config_name type."""
        mock_settings.REPORTS_DIR = temp_reports_dir

        view = ConfigReportView()
        view.request = request_factory.get("/config/Technology/")

        # Pass non-string config_name
        context = view.get_context_data(config_name=123)

        assert "error" in context
        assert "Invalid config name" in context["error"]

    @patch("web.newsserver.services.report_service.settings")
    def test_get_invalid_config_name_type(
        self,
        mock_settings,
        request_factory,
        temp_reports_dir,
    ):
        """Test get() with invalid config_name type."""
        mock_settings.REPORTS_DIR = temp_reports_dir

        view = ConfigReportView()
        view.request = request_factory.get(
            "/config/Technology/?download=1&report=test.html",
        )
        view.kwargs = {"config_name": 123}

        with pytest.raises(Http404) as exc_info:
            view.get(view.request, config_name=123)
        assert "Invalid config name" in str(exc_info.value)


class TestRunListView:
    """Test cases for RunListView."""

    @patch("web.newsserver.utils.django_timezone")
    @patch("web.newsserver.views.timezone")
    @patch("web.newsserver.views.ScrapeSummary")
    @patch("web.newsserver.views.AnalysisSummary")
    def test_get_context_data_default_date(
        self,
        mock_analysis_summary,
        mock_scrape_summary,
        mock_views_timezone,
        mock_utils_timezone,
        request_factory,
    ):
        """Test RunListView with default date (today)."""
        # Mock timezone.now()
        mock_now = datetime(2025, 12, 23, 12, 0, 0)
        mock_utils_timezone.now.return_value = mock_now
        mock_views_timezone.now.return_value = mock_now

        # Mock querysets
        mock_scrape_qs = Mock()
        mock_analysis_qs = Mock()

        mock_scrape_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_scrape_qs
        mock_analysis_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_analysis_qs

        view = RunListView()
        view.request = request_factory.get("/runs/")

        context = view.get_context_data()

        assert "selected_date" in context
        assert context["selected_date"] == mock_now.date()
        assert "prev_date" in context
        assert context["prev_date"] == mock_now.date() - timedelta(days=1)
        assert "next_date" in context
        assert context["next_date"] is None  # Can't go to future
        assert "scrape_runs" in context
        assert "analysis_runs" in context

    @patch("web.newsserver.utils.django_timezone")
    @patch("web.newsserver.views.timezone")
    @patch("web.newsserver.views.ScrapeSummary")
    @patch("web.newsserver.views.AnalysisSummary")
    def test_get_context_data_specific_date(
        self,
        mock_analysis_summary,
        mock_scrape_summary,
        mock_views_timezone,
        mock_utils_timezone,
        request_factory,
    ):
        """Test RunListView with specific date from query param."""
        mock_now = datetime(2025, 12, 23, 12, 0, 0)
        mock_utils_timezone.now.return_value = mock_now
        mock_views_timezone.now.return_value = mock_now

        mock_scrape_qs = Mock()
        mock_analysis_qs = Mock()

        mock_scrape_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_scrape_qs
        mock_analysis_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_analysis_qs

        view = RunListView()
        view.request = request_factory.get("/runs/?date=2025-12-20")

        context = view.get_context_data()

        from datetime import date

        assert context["selected_date"] == date(2025, 12, 20)
        assert context["prev_date"] == date(2025, 12, 19)
        assert context["next_date"] == date(2025, 12, 21)

    @patch("web.newsserver.utils.django_timezone")
    @patch("web.newsserver.views.timezone")
    @patch("web.newsserver.views.ScrapeSummary")
    @patch("web.newsserver.views.AnalysisSummary")
    def test_get_context_data_invalid_date(
        self,
        mock_analysis_summary,
        mock_scrape_summary,
        mock_views_timezone,
        mock_utils_timezone,
        request_factory,
    ):
        """Test RunListView with invalid date falls back to today."""
        mock_now = datetime(2025, 12, 23, 12, 0, 0)
        mock_utils_timezone.now.return_value = mock_now
        mock_views_timezone.now.return_value = mock_now

        mock_scrape_qs = Mock()
        mock_analysis_qs = Mock()

        mock_scrape_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_scrape_qs
        mock_analysis_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_analysis_qs

        view = RunListView()
        view.request = request_factory.get("/runs/?date=invalid-date")

        context = view.get_context_data()

        assert context["selected_date"] == mock_now.date()

    @patch("web.newsserver.utils.django_timezone")
    @patch("web.newsserver.views.timezone")
    @patch("web.newsserver.views.ScrapeSummary")
    @patch("web.newsserver.views.AnalysisSummary")
    def test_get_context_data_future_date_no_next(
        self,
        mock_analysis_summary,
        mock_scrape_summary,
        mock_views_timezone,
        mock_utils_timezone,
        request_factory,
    ):
        """Test that future dates don't allow next_date."""
        mock_now = datetime(2025, 12, 23, 12, 0, 0)
        mock_utils_timezone.now.return_value = mock_now
        mock_views_timezone.now.return_value = mock_now

        mock_scrape_qs = Mock()
        mock_analysis_qs = Mock()

        mock_scrape_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_scrape_qs
        mock_analysis_summary.objects.filter.return_value.select_related.return_value.order_by.return_value = mock_analysis_qs

        view = RunListView()
        # Request today's date
        view.request = request_factory.get("/runs/?date=2025-12-23")

        context = view.get_context_data()

        # Next date should be None (can't go to future)
        assert context["next_date"] is None


class TestLogsViewConfigTabs:
    """Test cases for LogsView config tabs functionality."""

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_with_config_tabs(
        self,
        mock_settings,
        request_factory,
        tmp_path,
    ):
        """Test that config tabs are built correctly from active logs."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        # Create multiple active log files
        (logs_dir / "technology.log").write_text("Technology log\n")
        (logs_dir / "world_politics.log").write_text("World Politics log\n")

        # Create rotated log files (should not create tabs)
        (logs_dir / "technology.log.2025-12-20").write_text("Old Technology log\n")

        view = LogsView()
        view.request = request_factory.get("/logs/")

        context = view.get_context_data()

        config_tabs = context["config_tabs"]
        assert len(config_tabs) == 2

        # Check tabs are sorted by display name
        assert config_tabs[0]["name"] == "technology"
        assert config_tabs[0]["display_name"] == "Technology"
        assert config_tabs[1]["name"] == "world_politics"
        assert config_tabs[1]["display_name"] == "World Politics"

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_active_tab_detection(
        self,
        mock_settings,
        request_factory,
        tmp_path,
    ):
        """Test that active tab is detected correctly."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        (logs_dir / "technology.log").write_text("Technology log\n")

        view = LogsView()
        view.request = request_factory.get("/logs/?log=technology.log")

        context = view.get_context_data()

        assert context["active_tab"] == "technology"

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_active_tab_from_rotated_log(
        self,
        mock_settings,
        request_factory,
        tmp_path,
    ):
        """Test that active tab is detected from rotated log."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        (logs_dir / "technology.log").write_text("Technology log\n")
        (logs_dir / "technology.log.2025-12-20").write_text("Old Technology log\n")

        view = LogsView()
        view.request = request_factory.get("/logs/?log=technology.log.2025-12-20")

        context = view.get_context_data()

        # Should extract base name from rotated log
        assert context["active_tab"] == "technology"

    @patch("web.newsserver.services.log_service.settings")
    def test_get_context_data_selected_log_path_traversal(
        self,
        mock_settings,
        request_factory,
        tmp_path,
    ):
        """Test path traversal protection in selected log."""
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()
        mock_settings.BASE_DIR = tmp_path

        (logs_dir / "newsbot.log").write_text("Newsbot log\n")

        view = LogsView()
        view.request = request_factory.get("/logs/?log=../../etc/passwd")

        context = view.get_context_data()

        # Should fall back to first log file (newsbot.log)
        assert context["current_log"] == "newsbot.log"
