"""
Tests for scheduler config refresh functionality.
"""

import importlib
from unittest.mock import Mock, patch

import pytest
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from newsbot.constants import TZ
from utilities import models as config_models

main_module = importlib.import_module("newsbot.main")


@pytest.fixture
def mock_config():
    """Create a mock configuration."""
    return config_models.ConfigModel(
        name="test",
        database=config_models.DatabaseConfigModel(url="sqlite:///memory"),
        scheduler=config_models.SchedulerConfigModel(
            daily_scrape=config_models.DailyScrapeConfigModel(
                enabled=True,
                hour=9,
                minute=0,
            ),
            weekly_analysis=config_models.WeeklyAnalysisConfigModel(
                enabled=True,
                day_of_week="mon",
                hour=10,
                minute=30,
                lookback_days=7,
            ),
        ),
        report=config_models.ReportConfigModel(lookback_days=7),
    )


@pytest.fixture
def mock_orchestrator(mock_config):
    """Create a mock orchestrator."""
    orchestrator = Mock()
    orchestrator.config = mock_config
    orchestrator._news_config = Mock()
    orchestrator.news_config = Mock()
    
    def set_news_config(news_config):
        orchestrator._news_config = news_config
    
    orchestrator.set_news_config = set_news_config
    return orchestrator


@pytest.fixture
def mock_scheduler():
    """Create a mock scheduler."""
    scheduler = Mock(spec=BlockingScheduler)
    scheduler.get_job = Mock(return_value=None)
    scheduler.add_job = Mock()
    scheduler.reschedule_job = Mock()
    scheduler.remove_job = Mock()
    return scheduler


@pytest.fixture
def mock_email_handler():
    """Create a mock email error handler."""
    handler = Mock()
    handler.flush = Mock()
    return handler


def test_update_daily_scrape_job_reschedules_when_enabled(
    mock_scheduler, mock_orchestrator, mock_email_handler, mock_config
):
    """Test that enabled daily scrape job gets rescheduled."""
    # Setup: existing job
    existing_job = Mock()
    mock_scheduler.get_job.return_value = existing_job

    # Create new config with different time
    daily_config = config_models.DailyScrapeConfigModel(
        enabled=True,
        hour=14,  # Changed from 9 to 14
        minute=30,  # Changed from 0 to 30
    )

    # Execute
    # Execute
    main_module._update_daily_scrape_job(
        mock_scheduler,
        daily_config,
        mock_email_handler,
        "test_config",
    )

    # Verify: job was rescheduled
    mock_scheduler.reschedule_job.assert_called_once()
    call_args = mock_scheduler.reschedule_job.call_args
    assert call_args[0][0] == "daily_scrape_job"
    trigger = call_args[1]["trigger"]
    assert isinstance(trigger, CronTrigger)


def test_update_daily_scrape_job_removes_when_disabled(
    mock_scheduler, mock_orchestrator, mock_email_handler, mock_config
):
    """Test that disabled daily scrape job gets removed."""
    # Setup: existing job
    existing_job = Mock()
    mock_scheduler.get_job.return_value = existing_job

    # Create disabled config
    daily_config = config_models.DailyScrapeConfigModel(
        enabled=False,
        hour=9,
        minute=0,
    )

    # Execute
    # Execute
    main_module._update_daily_scrape_job(
        mock_scheduler,
        daily_config,
        mock_email_handler,
        "test_config",
    )

    # Verify: job was removed
    mock_scheduler.remove_job.assert_called_once_with("daily_scrape_job")
    mock_scheduler.reschedule_job.assert_not_called()


def test_update_weekly_analysis_job_reschedules_when_enabled(
    mock_scheduler, mock_orchestrator, mock_email_handler, mock_config
):
    """Test that enabled weekly analysis job gets rescheduled."""
    # Setup: existing job
    existing_job = Mock()
    mock_scheduler.get_job.return_value = existing_job

    # Create new config with different schedule
    weekly_config = config_models.WeeklyAnalysisConfigModel(
        enabled=True,
        day_of_week="fri",  # Changed from mon
        hour=16,  # Changed from 10
        minute=0,  # Changed from 30
        lookback_days=7,
    )

    # Execute
    # Execute
    main_module._update_weekly_analysis_job(
        mock_scheduler,
        weekly_config,
        mock_email_handler,
        "test_config",
    )

    # Verify: job was rescheduled
    mock_scheduler.reschedule_job.assert_called_once()
    call_args = mock_scheduler.reschedule_job.call_args
    assert call_args[0][0] == "weekly_analysis_job"
    trigger = call_args[1]["trigger"]
    assert isinstance(trigger, CronTrigger)


def test_update_weekly_analysis_job_removes_when_disabled(
    mock_scheduler, mock_orchestrator, mock_email_handler, mock_config
):
    """Test that disabled weekly analysis job gets removed."""
    # Setup: existing job
    existing_job = Mock()
    mock_scheduler.get_job.return_value = existing_job

    # Create disabled config
    weekly_config = config_models.WeeklyAnalysisConfigModel(
        enabled=False,
        day_of_week="mon",
        hour=10,
        minute=30,
        lookback_days=7,
    )

    # Execute
    # Execute
    main_module._update_weekly_analysis_job(
        mock_scheduler,
        weekly_config,
        mock_email_handler,
        "test_config",
    )

    # Verify: job was removed
    mock_scheduler.remove_job.assert_called_once_with("weekly_analysis_job")
    mock_scheduler.reschedule_job.assert_not_called()


@patch.object(main_module, "load_config")
@patch.object(main_module, "_setup_scheduler_config")
@patch.object(main_module, "_update_daily_scrape_job")
@patch.object(main_module, "_update_weekly_analysis_job")
def test_refresh_config_and_jobs_reloads_config(
    mock_update_weekly,
    mock_update_daily,
    mock_setup_scheduler,
    mock_load_config,
    mock_scheduler,
    mock_orchestrator,
    mock_email_handler,
    mock_config,
):
    """Test that refresh_config_and_jobs reloads config from database."""
    # Setup
    new_config = config_models.ConfigModel(
        name="test",
        database=config_models.DatabaseConfigModel(url="sqlite:///memory"),
        scheduler=config_models.SchedulerConfigModel(
            daily_scrape=config_models.DailyScrapeConfigModel(
                enabled=True, hour=15, minute=45
            ),
            weekly_analysis=config_models.WeeklyAnalysisConfigModel(
                enabled=True,
                day_of_week="wed",
                hour=12,
                minute=0,
                lookback_days=14,
            ),
        ),
        report=config_models.ReportConfigModel(lookback_days=14),
    )
    new_news_config = Mock()

    mock_load_config.return_value = (new_config, new_news_config)

    mock_setup_scheduler.return_value = (
        new_config.scheduler.daily_scrape,
        new_config.scheduler.weekly_analysis,
    )

    # Execute
    # Execute
    main_module._refresh_config_and_jobs(
        mock_scheduler,
        mock_email_handler,
        "test_config",
    )

    # Verify
    mock_load_config.assert_called_once_with("test_config")
    mock_setup_scheduler.assert_called_once_with(new_config)
    mock_update_daily.assert_called_once()
    mock_update_weekly.assert_called_once()

    # Orchestrator mutation checks removed as orchestrator is no longer passed


@patch.object(main_module, "load_config")
def test_refresh_config_and_jobs_handles_errors_gracefully(
    mock_load_config,
    mock_scheduler,
    mock_orchestrator,
    mock_email_handler,
    mock_config,
):
    """Test that refresh handles database errors gracefully."""
    # Setup: load_config raises an exception
    mock_load_config.side_effect = Exception("Database connection error")

    # Execute - should not raise
    main_module._refresh_config_and_jobs(
        mock_scheduler,
        mock_email_handler,
        "test_config",
    )

    # Verify: error was logged but execution continued
    # (no exception raised)
    mock_load_config.assert_called_once_with("test_config")
