"""
Tests for scheduler script functionality.
"""

import sys
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

# Add scripts directory to path
scripts_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))

import scheduler


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler."""
    scheduler_instance = Mock(spec=BlockingScheduler)
    scheduler_instance.get_jobs = Mock(return_value=[])
    scheduler_instance.add_job = Mock()
    scheduler_instance.remove_job = Mock()
    return scheduler_instance


@pytest.fixture
def sample_schedules_response():
    """Sample API schedules response."""
    return {
        "configs": [
            {
                "key": "test_config",
                "name": "Test Config",
                "daily_scrape": {
                    "enabled": True,
                    "hour": 2,
                    "minute": 0,
                    "cron": "0 2 * * *",
                },
                "weekly_analysis": {
                    "enabled": True,
                    "day_of_week": "mon",
                    "hour": 9,
                    "minute": 0,
                    "lookback_days": 7,
                    "cron": "0 9 ? * MON",
                },
            },
            {
                "key": "another_config",
                "name": "Another Config",
                "daily_scrape": {
                    "enabled": True,
                    "hour": 3,
                    "minute": 30,
                    "cron": "30 3 * * *",
                },
                "weekly_analysis": {
                    "enabled": False,
                },
            },
        ],
    }


class TestMakeApiRequest:
    """Tests for make_api_request function."""

    @patch("scheduler.requests.request")
    def test_successful_get_request(self, mock_request):
        """Test successful GET request."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        result = scheduler.make_api_request(
            "GET", "http://localhost:8000/test"
        )

        assert result == {"status": "ok"}
        mock_request.assert_called_once()
        mock_response.raise_for_status.assert_called_once()

    @patch("scheduler.requests.request")
    def test_successful_post_request(self, mock_request):
        """Test successful POST request with JSON data."""
        mock_response = Mock()
        mock_response.json.return_value = {"status": "started"}
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        json_data = {"force": False}
        result = scheduler.make_api_request(
            "POST",
            "http://localhost:8000/run/test",
            json_data=json_data,
        )

        assert result == {"status": "started"}
        call_args = mock_request.call_args
        assert call_args[0][0] == "POST"
        assert "json" in call_args[1]
        assert call_args[1]["json"] == json_data

    @patch("scheduler.requests.request")
    def test_connection_error(self, mock_request):
        """Test handling of connection error."""
        import requests

        mock_request.side_effect = requests.exceptions.ConnectionError(
            "Connection refused",
        )

        result = scheduler.make_api_request(
            "GET", "http://localhost:8000/test"
        )

        assert result is None

    @patch("scheduler.requests.request")
    def test_timeout_error(self, mock_request):
        """Test handling of timeout error."""
        import requests

        mock_request.side_effect = requests.exceptions.Timeout(
            "Request timeout"
        )

        result = scheduler.make_api_request(
            "GET", "http://localhost:8000/test"
        )

        assert result is None

    @patch("scheduler.requests.request")
    def test_http_error(self, mock_request):
        """Test handling of HTTP error."""
        import requests

        mock_response = Mock()
        mock_response.status_code = 404
        mock_request.side_effect = requests.exceptions.HTTPError(
            response=mock_response
        )

        result = scheduler.make_api_request(
            "GET", "http://localhost:8000/test"
        )

        assert result is None

    @patch("scheduler.requests.request")
    def test_invalid_json_response(self, mock_request):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.raise_for_status = Mock()
        mock_request.return_value = mock_response

        result = scheduler.make_api_request(
            "GET", "http://localhost:8000/test"
        )

        assert result is None


class TestScheduleDailyScrape:
    """Tests for schedule_daily_scrape function."""

    def test_schedules_job_correctly(self, mock_scheduler):
        """Test that daily scrape job is scheduled correctly."""
        api_base_url = "http://localhost:8000"
        config_key = "test_config"
        hour = 2
        minute = 30

        scheduler.schedule_daily_scrape(
            mock_scheduler,
            api_base_url,
            config_key,
            hour,
            minute,
        )

        mock_scheduler.add_job.assert_called_once()
        call_args = mock_scheduler.add_job.call_args
        assert call_args[0][0] == scheduler.trigger_daily_scrape
        assert (
            call_args[1]["id"]
            == f"{scheduler.JOB_PREFIX}daily_scrape_{config_key}"
        )
        assert call_args[1]["name"] == f"Daily Scrape: {config_key}"
        assert call_args[1]["args"] == [api_base_url, config_key]
        assert call_args[1]["replace_existing"] is True


class TestScheduleWeeklyAnalysis:
    """Tests for schedule_weekly_analysis function."""

    def test_schedules_job_correctly(self, mock_scheduler):
        """Test that weekly analysis job is scheduled correctly."""
        api_base_url = "http://localhost:8000"
        config_key = "test_config"
        day_of_week = "mon"
        hour = 9
        minute = 0
        scheduler.schedule_weekly_analysis(
            mock_scheduler,
            api_base_url,
            config_key,
            day_of_week,
            hour,
            minute,
        )

        mock_scheduler.add_job.assert_called_once()
        call_args = mock_scheduler.add_job.call_args
        assert call_args[0][0] == scheduler.trigger_weekly_analysis
        assert (
            call_args[1]["id"]
            == f"{scheduler.JOB_PREFIX}weekly_analysis_{config_key}"
        )
        assert call_args[1]["name"] == f"Weekly Analysis: {config_key}"
        assert call_args[1]["args"] == [api_base_url, config_key]
        assert call_args[1]["replace_existing"] is True


class TestRemoveStaleJobs:
    """Tests for remove_stale_jobs function."""

    def test_removes_stale_jobs(self, mock_scheduler):
        """Test that stale jobs are removed."""
        # Create mock jobs - one active, one stale
        active_job = Mock()
        active_job.id = f"{scheduler.JOB_PREFIX}daily_scrape_active_config"

        stale_job = Mock()
        stale_job.id = f"{scheduler.JOB_PREFIX}daily_scrape_stale_config"

        other_job = Mock()
        other_job.id = "some_other_job_id"

        mock_scheduler.get_jobs.return_value = [
            active_job,
            stale_job,
            other_job,
        ]

        active_config_keys = {"active_config"}

        scheduler.remove_stale_jobs(mock_scheduler, active_config_keys)

        # Should only remove stale_job
        mock_scheduler.remove_job.assert_called_once_with(stale_job.id)

    def test_does_not_remove_refresh_job(self, mock_scheduler):
        """Test that refresh job is not removed."""
        refresh_job = Mock()
        refresh_job.id = scheduler.REFRESH_JOB_ID

        daily_job = Mock()
        daily_job.id = f"{scheduler.JOB_PREFIX}daily_scrape_test_config"

        mock_scheduler.get_jobs.return_value = [refresh_job, daily_job]

        scheduler.remove_stale_jobs(mock_scheduler, set())

        # Should not remove refresh job, only daily job
        assert mock_scheduler.remove_job.call_count == 1
        assert mock_scheduler.remove_job.call_args[0][0] == daily_job.id

    def test_does_not_remove_non_scheduler_jobs(self, mock_scheduler):
        """Test that non-scheduler jobs are not removed."""
        other_job = Mock()
        other_job.id = "external_job_id"

        mock_scheduler.get_jobs.return_value = [other_job]

        scheduler.remove_stale_jobs(mock_scheduler, set())

        mock_scheduler.remove_job.assert_not_called()


class TestRefreshAndScheduleTasks:
    """Tests for refresh_and_schedule_tasks function."""

    @patch("scheduler.make_api_request")
    def test_fetches_and_schedules_tasks(
        self, mock_api_request, mock_scheduler, sample_schedules_response
    ):
        """Test that schedules are fetched and tasks are scheduled."""
        mock_api_request.return_value = sample_schedules_response

        api_base_url = "http://localhost:8000"

        scheduler.refresh_and_schedule_tasks(mock_scheduler, api_base_url)

        # Verify API was called
        mock_api_request.assert_called_once_with(
            "GET", f"{api_base_url}/schedules"
        )

        # Verify jobs were added (2 daily scrapes + 1 weekly analysis)
        assert mock_scheduler.add_job.call_count == 3

    @patch("scheduler.make_api_request")
    def test_handles_api_failure(self, mock_api_request, mock_scheduler):
        """Test handling of API request failure."""
        mock_api_request.return_value = None

        scheduler.refresh_and_schedule_tasks(
            mock_scheduler, "http://localhost:8000"
        )

        # Should not add any jobs
        mock_scheduler.add_job.assert_not_called()

    @patch("scheduler.make_api_request")
    def test_handles_empty_configs(self, mock_api_request, mock_scheduler):
        """Test handling of empty configs list."""
        mock_api_request.return_value = {"configs": []}

        scheduler.refresh_and_schedule_tasks(
            mock_scheduler, "http://localhost:8000"
        )

        # Should not add any jobs
        mock_scheduler.add_job.assert_not_called()

    @patch("scheduler.make_api_request")
    def test_skips_configs_missing_key(self, mock_api_request, mock_scheduler):
        """Test that configs without key are skipped."""
        mock_api_request.return_value = {
            "configs": [
                {
                    "name": "Config without key",
                    "daily_scrape": {"enabled": True},
                },
            ],
        }

        scheduler.refresh_and_schedule_tasks(
            mock_scheduler, "http://localhost:8000"
        )

        # Should not add any jobs
        mock_scheduler.add_job.assert_not_called()

    @patch("scheduler.make_api_request")
    def test_schedules_only_enabled_tasks(
        self, mock_api_request, mock_scheduler
    ):
        """Test that only enabled tasks are scheduled."""
        mock_api_request.return_value = {
            "configs": [
                {
                    "key": "test_config",
                    "daily_scrape": {"enabled": False},
                    "weekly_analysis": {
                        "enabled": True,
                        "day_of_week": "mon",
                        "hour": 9,
                        "minute": 0,
                    },
                },
            ],
        }

        scheduler.refresh_and_schedule_tasks(
            mock_scheduler, "http://localhost:8000"
        )

        # Should only schedule weekly analysis
        assert mock_scheduler.add_job.call_count == 1
        call_args = mock_scheduler.add_job.call_args
        assert call_args[0][0] == scheduler.trigger_weekly_analysis


class TestTriggerFunctions:
    """Tests for trigger functions."""

    @patch("scheduler.make_api_request")
    @patch("scheduler.logger")
    def test_trigger_daily_scrape_success(self, mock_logger, mock_api_request):
        """Test successful daily scrape trigger."""
        mock_api_request.return_value = {"status": "started"}

        scheduler.trigger_daily_scrape("http://localhost:8000", "test_config")

        mock_api_request.assert_called_once_with(
            "POST",
            "http://localhost:8000/run/test_config",
            json_data={"force": False},
        )

    @patch("scheduler.make_api_request")
    @patch("scheduler.logger")
    def test_trigger_weekly_analysis_success(
        self, mock_logger, mock_api_request
    ):
        """Test successful weekly analysis trigger."""
        mock_api_request.return_value = {"status": "started"}

        scheduler.trigger_weekly_analysis(
            "http://localhost:8000", "test_config"
        )

        mock_api_request.assert_called_once_with(
            "POST",
            "http://localhost:8000/analyze/test_config",
            json_data={},
        )

    @patch("scheduler.make_api_request")
    @patch("scheduler.logger")
    def test_trigger_handles_failure(self, mock_logger, mock_api_request):
        """Test handling of trigger failures."""
        mock_api_request.return_value = None

        scheduler.trigger_daily_scrape("http://localhost:8000", "test_config")

        # Should log error
        error_calls = [
            call for call in mock_logger.error.call_args_list if call
        ]
        assert len(error_calls) > 0


class TestPrintSchedulerInfo:
    """Tests for print_scheduler_info function."""

    def test_prints_job_info(self, mock_scheduler, capsys):
        """Test that scheduler info is printed correctly."""
        from datetime import datetime, tzinfo

        job1 = Mock()
        job1.name = "Test Job 1"
        job1.next_run_time = datetime(
            2024, 1, 1, 12, 0, 0, tzinfo=cast(tzinfo | None, pytz.UTC)
        )

        job2 = Mock()
        job2.name = "Test Job 2"
        job2.next_run_time = None

        mock_scheduler.get_jobs.return_value = [job1, job2]

        scheduler.print_scheduler_info(mock_scheduler)

        captured = capsys.readouterr()
        assert "NEWSBOT SCHEDULER SERVICE" in captured.out
        assert "Test Job 1" in captured.out
        assert "Test Job 2" in captured.out
        assert "Jobs scheduled: 2" in captured.out
