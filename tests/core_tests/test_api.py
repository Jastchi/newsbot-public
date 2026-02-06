"""
Comprehensive tests for the API module.

Tests cover:
- handlers.py: handle_run, handle_analyze, cron helpers, get_all_schedules
- app.py: FastAPI endpoints /run, /analyze, /schedules, /health
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from api.handlers import (
    daily_scrape_to_cron,
    get_all_schedules,
    handle_analyze,
    handle_run,
    weekly_analysis_to_cron,
)
from utilities import ConfigNotFoundError


@pytest.fixture(autouse=True)
def _quiet_logs():
    """Quiet down logs during tests."""
    logging.getLogger("newsbot").setLevel(logging.CRITICAL)
    logging.getLogger("api").setLevel(logging.CRITICAL)


# ============================================================================
# Tests for cron helper functions
# ============================================================================


class TestDailyScrapeToCron:
    """Tests for daily_scrape_to_cron helper function."""

    def test_basic_conversion(self):
        """Test basic hour/minute to cron conversion."""
        assert daily_scrape_to_cron(2, 0) == "0 2 * * *"
        assert daily_scrape_to_cron(14, 30) == "30 14 * * *"

    def test_midnight(self):
        """Test midnight conversion."""
        assert daily_scrape_to_cron(0, 0) == "0 0 * * *"

    def test_end_of_day(self):
        """Test end of day conversion."""
        assert daily_scrape_to_cron(23, 59) == "59 23 * * *"

    def test_various_times(self):
        """Test various time combinations."""
        test_cases = [
            ((6, 15), "15 6 * * *"),
            ((12, 0), "0 12 * * *"),
            ((18, 45), "45 18 * * *"),
            ((3, 5), "5 3 * * *"),
        ]
        for (hour, minute), expected in test_cases:
            assert daily_scrape_to_cron(hour, minute) == expected


class TestWeeklyAnalysisToCron:
    """Tests for weekly_analysis_to_cron helper function."""

    def test_monday(self):
        """Test Monday conversion."""
        assert weekly_analysis_to_cron("mon", 9, 0) == "0 9 * * 1"

    def test_all_days_of_week(self):
        """Test all days of week."""
        expected_days = {
            "sun": 0,
            "mon": 1,
            "tue": 2,
            "wed": 3,
            "thu": 4,
            "fri": 5,
            "sat": 6,
        }
        for day, num in expected_days.items():
            assert weekly_analysis_to_cron(day, 10, 30) == f"30 10 * * {num}"

    def test_case_insensitive(self):
        """Test that day names are case insensitive."""
        assert weekly_analysis_to_cron("MON", 9, 0) == "0 9 * * 1"
        assert weekly_analysis_to_cron("Mon", 9, 0) == "0 9 * * 1"
        assert weekly_analysis_to_cron("mOn", 9, 0) == "0 9 * * 1"

    def test_unknown_day_defaults_to_monday(self):
        """Test that unknown day names default to Monday."""
        assert weekly_analysis_to_cron("invalid", 9, 0) == "0 9 * * 1"
        assert weekly_analysis_to_cron("xyz", 12, 30) == "30 12 * * 1"


# ============================================================================
# Tests for get_all_schedules
# ============================================================================


class TestGetAllSchedules:
    """Tests for get_all_schedules function."""

    @pytest.mark.django_db
    def test_returns_empty_list_when_no_configs(self):
        """Test that empty list is returned when no active configs exist."""
        from utilities.django_models import NewsConfig

        # Delete all existing configs
        NewsConfig.objects.all().delete()

        schedules = get_all_schedules()

        assert schedules == []

    @pytest.mark.django_db
    def test_returns_schedules_for_active_configs(self):
        """Test that schedules are returned for active configs."""
        from utilities.django_models import NewsConfig

        # Create an active config
        config, _ = NewsConfig.objects.get_or_create(
            key="test_schedules",
            defaults={
                "display_name": "Test Schedules Config",
                "is_active": True,
                "scheduler_daily_scrape_enabled": True,
                "scheduler_weekly_analysis_enabled": True,
                "scheduler_weekly_analysis_day_of_week": "mon",
                "scheduler_weekly_analysis_hour": 9,
                "scheduler_weekly_analysis_minute": 0,
                "scheduler_weekly_analysis_lookback_days": 7,
            },
        )

        schedules = get_all_schedules()

        # Find our test config in schedules
        test_schedule = next(
            (s for s in schedules if s["key"] == "test_schedules"),
            None,
        )
        assert test_schedule is not None
        assert test_schedule["name"] == "Test Schedules Config"
        assert test_schedule["daily_scrape"]["enabled"] is True
        assert test_schedule["daily_scrape"]["cron"] == "5 0 * * *"
        assert test_schedule["weekly_analysis"]["enabled"] is True
        assert test_schedule["weekly_analysis"]["cron"] == "0 9 * * 1"

    @pytest.mark.django_db
    def test_excludes_inactive_configs(self):
        """Test that inactive configs are not included."""
        from utilities.django_models import NewsConfig

        # Create an inactive config
        NewsConfig.objects.update_or_create(
            key="test_inactive",
            defaults={
                "display_name": "Inactive Config",
                "is_active": False,
            },
        )

        schedules = get_all_schedules()

        # Verify inactive config is not in schedules
        inactive_schedule = next(
            (s for s in schedules if s["key"] == "test_inactive"),
            None,
        )
        assert inactive_schedule is None

    @pytest.mark.django_db
    def test_disabled_schedules_have_null_cron(self):
        """Test that disabled schedules have null cron expressions."""
        from utilities.django_models import NewsConfig

        # Create config with disabled schedules
        config, _ = NewsConfig.objects.update_or_create(
            key="test_disabled_schedules",
            defaults={
                "display_name": "Disabled Schedules Config",
                "is_active": True,
                "scheduler_daily_scrape_enabled": False,
                "scheduler_weekly_analysis_enabled": False,
            },
        )

        schedules = get_all_schedules()

        test_schedule = next(
            (s for s in schedules if s["key"] == "test_disabled_schedules"),
            None,
        )
        assert test_schedule is not None
        assert test_schedule["daily_scrape"]["enabled"] is False
        assert test_schedule["daily_scrape"]["cron"] is None
        assert test_schedule["weekly_analysis"]["enabled"] is False
        assert test_schedule["weekly_analysis"]["cron"] is None

    @pytest.mark.django_db
    def test_schedules_returned_in_dependency_order(self):
        """Configs with exclude_articles_from_configs come after those configs."""
        from utilities.django_models import NewsConfig

        NewsConfig.objects.all().delete()

        config_b, _ = NewsConfig.objects.get_or_create(
            key="dep_config",
            defaults={
                "display_name": "Dependency Config",
                "is_active": True,
                "scheduler_daily_scrape_enabled": True,
                "scheduler_weekly_analysis_enabled": False,
            },
        )
        config_a, _ = NewsConfig.objects.get_or_create(
            key="excluder_config",
            defaults={
                "display_name": "Excluder Config",
                "is_active": True,
                "scheduler_daily_scrape_enabled": True,
                "scheduler_weekly_analysis_enabled": False,
            },
        )
        config_a.exclude_articles_from_configs.add(config_b)

        schedules = get_all_schedules()

        keys = [s["key"] for s in schedules]
        assert keys.index("dep_config") < keys.index("excluder_config")


# ============================================================================
# Tests for handle_run
# ============================================================================


@pytest.mark.django_db
class TestHandleRun:
    """Tests for handle_run handler function."""

    @pytest.fixture
    def mock_dependencies(self, monkeypatch):
        """Mock all dependencies for handle_run."""
        # Mock load_config
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_config.report.lookback_days = 7
        mock_news_config = Mock()

        mock_load_config = Mock(return_value=(mock_config, mock_news_config))
        monkeypatch.setattr("api.handlers.load_config", mock_load_config)

        # Mock email error handler
        mock_email_handler = Mock()
        mock_get_email_handler = Mock(return_value=mock_email_handler)
        monkeypatch.setattr(
            "api.handlers.get_email_error_handler",
            mock_get_email_handler,
        )

        # Mock setup_logging and validate_environment
        monkeypatch.setattr("api.handlers.setup_logging", Mock())
        monkeypatch.setattr("api.handlers.validate_environment", Mock())

        # Mock PipelineOrchestrator
        mock_results = Mock()
        mock_results.success = True
        mock_results.articles_count = 10
        mock_results.saved_to_db = 8
        mock_results.duration = 5.5
        mock_results.errors = []

        mock_orchestrator = Mock()
        mock_orchestrator.get_pipeline_status.return_value = {"status": "ok"}
        mock_orchestrator.run_daily_scrape.return_value = mock_results

        mock_orchestrator_class = Mock(return_value=mock_orchestrator)
        monkeypatch.setattr(
            "api.handlers.PipelineOrchestrator",
            mock_orchestrator_class,
        )

        # Mock SummaryWriter
        mock_summary_writer = Mock()
        mock_summary_writer_class = Mock(return_value=mock_summary_writer)
        monkeypatch.setattr(
            "api.handlers.SummaryWriter",
            mock_summary_writer_class,
        )

        return {
            "load_config": mock_load_config,
            "config": mock_config,
            "news_config": mock_news_config,
            "email_handler": mock_email_handler,
            "orchestrator": mock_orchestrator,
            "orchestrator_class": mock_orchestrator_class,
            "results": mock_results,
            "summary_writer": mock_summary_writer,
        }

    def test_successful_run(self, mock_dependencies):
        """Test successful run operation."""
        result = handle_run("test_config")

        assert result["success"] is True
        assert result["config_key"] == "test_config"
        assert result["config_name"] == "Test Config"
        assert result["articles_scraped"] == 10
        assert result["articles_saved"] == 8
        assert result["errors"] == []
        assert "duration" in result

    def test_run_with_force_flag(self, mock_dependencies):
        """Test run with force flag."""
        handle_run("test_config", force=True)

        mock_dependencies[
            "orchestrator"
        ].run_daily_scrape.assert_called_once_with(
            force=True,
        )

    def test_run_config_not_found(self, monkeypatch):
        """Test run with non-existent config raises error."""

        def mock_load_config(config_key):
            raise ConfigNotFoundError(f"Config not found: {config_key}")

        monkeypatch.setattr("api.handlers.load_config", mock_load_config)

        with pytest.raises(ConfigNotFoundError, match="Config not found"):
            handle_run("nonexistent")

    def test_run_handles_pipeline_exception(self, mock_dependencies):
        """Test that pipeline exceptions are handled gracefully."""
        mock_dependencies[
            "orchestrator"
        ].run_daily_scrape.side_effect = RuntimeError("Pipeline failed")

        result = handle_run("test_config")

        assert result["success"] is False
        assert "Pipeline failed" in result["error"]
        assert result["articles_scraped"] == 0
        assert result["articles_saved"] == 0

    def test_run_flushes_email_handler(self, mock_dependencies):
        """Test that email handler is flushed after run."""
        handle_run("test_config")

        mock_dependencies["email_handler"].flush.assert_called_once()

    def test_run_saves_summary_to_db(self, mock_dependencies):
        """Test that scrape summary is saved to database."""
        handle_run("test_config")

        mock_dependencies[
            "summary_writer"
        ].save_scrape_summary.assert_called_once()


# ============================================================================
# Tests for handle_analyze
# ============================================================================


@pytest.mark.django_db
class TestHandleAnalyze:
    """Tests for handle_analyze handler function."""

    @pytest.fixture
    def mock_dependencies(self, monkeypatch):
        """Mock all dependencies for handle_analyze."""
        # Mock load_config
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_config.report.lookback_days = 7
        mock_news_config = Mock()

        mock_load_config = Mock(return_value=(mock_config, mock_news_config))
        monkeypatch.setattr("api.handlers.load_config", mock_load_config)

        # Mock email error handler
        mock_email_handler = Mock()
        mock_get_email_handler = Mock(return_value=mock_email_handler)
        monkeypatch.setattr(
            "api.handlers.get_email_error_handler",
            mock_get_email_handler,
        )

        # Mock setup_logging and validate_environment
        monkeypatch.setattr("api.handlers.setup_logging", Mock())
        monkeypatch.setattr("api.handlers.validate_environment", Mock())

        # Mock PipelineOrchestrator
        mock_story = SimpleNamespace(
            title="Test Story",
            sources=["Source A", "Source B"],
            article_count=5,
        )
        mock_results = Mock()
        mock_results.success = True
        mock_results.articles_count = 50
        mock_results.stories_count = 3
        mock_results.top_stories = [mock_story]
        mock_results.duration = 120.5
        mock_results.errors = []

        mock_orchestrator = Mock()
        mock_orchestrator.run_weekly_analysis.return_value = mock_results

        mock_orchestrator_class = Mock(return_value=mock_orchestrator)
        monkeypatch.setattr(
            "api.handlers.PipelineOrchestrator",
            mock_orchestrator_class,
        )

        # Mock SummaryWriter
        mock_summary_writer = Mock()
        mock_summary_writer_class = Mock(return_value=mock_summary_writer)
        monkeypatch.setattr(
            "api.handlers.SummaryWriter",
            mock_summary_writer_class,
        )

        # Mock insert_test_articles
        mock_insert_test = Mock(return_value=5)
        monkeypatch.setattr(
            "api.handlers.insert_test_articles",
            mock_insert_test,
        )

        return {
            "load_config": mock_load_config,
            "config": mock_config,
            "news_config": mock_news_config,
            "email_handler": mock_email_handler,
            "orchestrator": mock_orchestrator,
            "orchestrator_class": mock_orchestrator_class,
            "results": mock_results,
            "summary_writer": mock_summary_writer,
            "insert_test_articles": mock_insert_test,
        }

    def test_successful_analyze(self, mock_dependencies):
        """Test successful analysis operation."""
        result = handle_analyze("test_config")

        assert result["success"] is True
        assert result["config_key"] == "test_config"
        assert result["config_name"] == "Test Config"
        assert result["articles_analyzed"] == 50
        assert result["stories_identified"] == 3
        assert result["lookback_days"] == 7
        assert len(result["top_stories"]) == 1
        assert result["top_stories"][0]["title"] == "Test Story"
        assert result["errors"] == []

    def test_analyze_with_custom_days(self, mock_dependencies):
        """Test analysis with custom days parameter."""
        result = handle_analyze("test_config", days=14)

        mock_dependencies[
            "orchestrator"
        ].run_weekly_analysis.assert_called_once_with(
            days_back=14,
        )
        assert result["lookback_days"] == 14

    def test_analyze_with_email_receivers_override(self, mock_dependencies):
        """Test analysis with email receivers override."""
        receivers = ["test@example.com"]

        handle_analyze("test_config", email_receivers=receivers)

        mock_dependencies[
            "orchestrator"
        ].set_email_receivers_override.assert_called_once_with(receivers)

    def test_analyze_test_flag_requires_test_config(self, mock_dependencies):
        """Test that test flag only works with test configs."""
        result = handle_analyze("production_config", test=True)

        assert result["success"] is False
        assert "--test requires a test config" in result["error"]
        mock_dependencies["insert_test_articles"].assert_not_called()

    def test_analyze_test_flag_with_test_config(self, mock_dependencies):
        """Test that test flag works with test configs."""
        handle_analyze("test_config", test=True)

        mock_dependencies["insert_test_articles"].assert_called_once_with(
            "test_config",
        )

    def test_analyze_config_not_found(self, monkeypatch):
        """Test analysis with non-existent config raises error."""

        def mock_load_config(config_key):
            raise ConfigNotFoundError(f"Config not found: {config_key}")

        monkeypatch.setattr("api.handlers.load_config", mock_load_config)

        with pytest.raises(ConfigNotFoundError, match="Config not found"):
            handle_analyze("nonexistent")

    def test_analyze_handles_pipeline_exception(self, mock_dependencies):
        """Test that pipeline exceptions are handled gracefully."""
        mock_dependencies[
            "orchestrator"
        ].run_weekly_analysis.side_effect = RuntimeError("Analysis failed")

        result = handle_analyze("test_config")

        assert result["success"] is False
        assert "Analysis failed" in result["error"]
        assert result["articles_analyzed"] == 0
        assert result["stories_identified"] == 0

    def test_analyze_flushes_email_handler(self, mock_dependencies):
        """Test that email handler is flushed after analysis."""
        handle_analyze("test_config")

        mock_dependencies["email_handler"].flush.assert_called_once()

    def test_analyze_saves_summary_to_db(self, mock_dependencies):
        """Test that analysis summary is saved to database."""
        handle_analyze("test_config")

        mock_dependencies[
            "summary_writer"
        ].save_analysis_summary.assert_called_once()

    def test_analyze_top_stories_limited_to_five(self, mock_dependencies):
        """Test that top_stories in response is limited to 5."""
        # Create 10 mock stories
        mock_stories = [
            SimpleNamespace(
                title=f"Story {i}",
                sources=["Source A"],
                article_count=i,
            )
            for i in range(10)
        ]
        mock_dependencies["results"].top_stories = mock_stories

        result = handle_analyze("test_config")

        assert len(result["top_stories"]) == 5


# ============================================================================
# Tests for FastAPI endpoints
# ============================================================================


class TestFastAPIEndpoints:
    """Tests for FastAPI application endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        # Import here to ensure Django is set up
        from api.app import app

        return TestClient(app)

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["message"] == "NewsBot API is running"

    @pytest.mark.django_db
    def test_run_endpoint_success(self, client, monkeypatch):
        """Test /run/{config_key} endpoint with successful response."""
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_load_config = Mock(return_value=(mock_config, Mock()))
        monkeypatch.setattr("api.app.load_config", mock_load_config)
        monkeypatch.setattr("api.app.handle_run", Mock())

        response = client.post("/run/test_config")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["config_key"] == "test_config"
        assert data["config_name"] == "Test Config"
        assert "started" in data["message"].lower()
        assert "job_id" in data

    @pytest.mark.django_db
    def test_run_endpoint_with_request_body(self, client, monkeypatch):
        """Test /run endpoint with request body."""
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_load_config = Mock(return_value=(mock_config, Mock()))
        monkeypatch.setattr("api.app.load_config", mock_load_config)
        mock_handle_run = Mock()
        monkeypatch.setattr("api.app.handle_run", mock_handle_run)

        response = client.post(
            "/run/test_config",
            json={
                "force": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        # Note: handle_run is called in BackgroundTasks, which TestClient
        # doesn't execute, so we can't verify it was called here

    @pytest.mark.django_db
    def test_run_endpoint_config_not_found(self, client, monkeypatch):
        """Test /run endpoint with non-existent config."""
        monkeypatch.setattr(
            "api.app.load_config",
            Mock(side_effect=ConfigNotFoundError("Config not found: invalid")),
        )

        response = client.post("/run/invalid")

        assert response.status_code == 404
        assert "Config not found" in response.json()["detail"]

    @pytest.mark.django_db
    def test_run_endpoint_internal_error(self, client, monkeypatch):
        """Test /run endpoint with internal server error."""
        monkeypatch.setattr(
            "api.app.load_config",
            Mock(side_effect=RuntimeError("Internal error")),
        )

        response = client.post("/run/test_config")

        assert response.status_code == 500
        assert "Internal error" in response.json()["detail"]

    @pytest.mark.django_db
    def test_analyze_endpoint_success(self, client, monkeypatch):
        """Test /analyze/{config_key} endpoint with successful response."""
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_load_config = Mock(return_value=(mock_config, Mock()))
        monkeypatch.setattr("api.app.load_config", mock_load_config)
        monkeypatch.setattr("api.app.handle_analyze", Mock())

        response = client.post("/analyze/test_config")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["config_key"] == "test_config"
        assert data["config_name"] == "Test Config"
        assert "started" in data["message"].lower()

    @pytest.mark.django_db
    def test_analyze_endpoint_with_request_body(self, client, monkeypatch):
        """Test /analyze endpoint with request body."""
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_load_config = Mock(return_value=(mock_config, Mock()))
        monkeypatch.setattr("api.app.load_config", mock_load_config)
        mock_handle_analyze = Mock()
        monkeypatch.setattr("api.app.handle_analyze", mock_handle_analyze)

        response = client.post(
            "/analyze/test_config",
            json={
                "days": 14,
                "test": False,
                "email_receivers": ["test@example.com"],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        # Note: handle_analyze is called in BackgroundTasks, which TestClient
        # doesn't execute, so we can't verify it was called here

    @pytest.mark.django_db
    def test_analyze_endpoint_config_not_found(self, client, monkeypatch):
        """Test /analyze endpoint with non-existent config."""
        monkeypatch.setattr(
            "api.app.load_config",
            Mock(side_effect=ConfigNotFoundError("Config not found: invalid")),
        )

        response = client.post("/analyze/invalid")

        assert response.status_code == 404
        assert "Config not found" in response.json()["detail"]

    @pytest.mark.django_db
    def test_schedules_endpoint_success(self, client, monkeypatch):
        """Test /schedules endpoint."""
        mock_schedules = [
            {
                "key": "test_config",
                "name": "Test Config",
                "is_active": True,
                "daily_scrape": {
                    "enabled": True,
                    "cron": "5 0 * * *",
                },
                "weekly_analysis": {
                    "enabled": True,
                    "day_of_week": "mon",
                    "hour": 9,
                    "minute": 0,
                    "lookback_days": 7,
                    "cron": "0 9 * * 1",
                },
            },
        ]
        monkeypatch.setattr(
            "api.app.get_all_schedules",
            Mock(return_value=mock_schedules),
        )

        response = client.get("/schedules")

        assert response.status_code == 200
        data = response.json()
        assert "configs" in data
        assert len(data["configs"]) == 1
        assert data["configs"][0]["key"] == "test_config"

    @pytest.mark.django_db
    def test_schedules_endpoint_error(self, client, monkeypatch):
        """Test /schedules endpoint with error."""
        monkeypatch.setattr(
            "api.app.get_all_schedules",
            Mock(side_effect=RuntimeError("Database error")),
        )

        response = client.get("/schedules")

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    @pytest.mark.django_db
    def test_run_endpoint_without_request_body(self, client, monkeypatch):
        """Test /run endpoint without request body uses defaults."""
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_load_config = Mock(return_value=(mock_config, Mock()))
        monkeypatch.setattr("api.app.load_config", mock_load_config)
        monkeypatch.setattr("api.app.handle_run", Mock())

        response = client.post("/run/test_config")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["config_key"] == "test_config"

    @pytest.mark.django_db
    def test_analyze_endpoint_without_request_body(self, client, monkeypatch):
        """Test /analyze endpoint without request body uses defaults."""
        mock_config = Mock()
        mock_config.name = "Test Config"
        mock_load_config = Mock(return_value=(mock_config, Mock()))
        monkeypatch.setattr("api.app.load_config", mock_load_config)
        monkeypatch.setattr("api.app.handle_analyze", Mock())

        response = client.post("/analyze/test_config")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "pending"
        assert data["config_key"] == "test_config"


# ============================================================================
# Integration tests
# ============================================================================


class TestAPIIntegration:
    """Integration tests for API module with real database."""

    @pytest.fixture
    def test_news_config(self):
        """Create a test NewsConfig for integration tests."""
        from utilities.django_models import NewsConfig

        config, _ = NewsConfig.objects.update_or_create(
            key="test_api_integration",
            defaults={
                "display_name": "API Integration Test",
                "is_active": True,
                "llm_provider": "ollama",
                "llm_model": "llama2",
                "scheduler_daily_scrape_enabled": True,
                "scheduler_weekly_analysis_enabled": True,
                "scheduler_weekly_analysis_day_of_week": "fri",
                "scheduler_weekly_analysis_hour": 10,
                "scheduler_weekly_analysis_minute": 15,
                "scheduler_weekly_analysis_lookback_days": 14,
            },
        )
        return config

    @pytest.mark.django_db
    def test_get_all_schedules_integration(self, test_news_config):
        """Integration test for get_all_schedules with real database."""
        schedules = get_all_schedules()

        # Find our test config
        test_schedule = next(
            (s for s in schedules if s["key"] == "test_api_integration"),
            None,
        )

        assert test_schedule is not None
        assert test_schedule["name"] == "API Integration Test"
        assert test_schedule["daily_scrape"]["enabled"] is True
        assert test_schedule["daily_scrape"]["cron"] == "5 0 * * *"
        assert test_schedule["weekly_analysis"]["day_of_week"] == "fri"
        assert test_schedule["weekly_analysis"]["hour"] == 10
        assert test_schedule["weekly_analysis"]["minute"] == 15
        assert test_schedule["weekly_analysis"]["cron"] == "15 10 * * 5"
        assert test_schedule["weekly_analysis"]["lookback_days"] == 14


# ============================================================================
# Pydantic model tests
# ============================================================================


class TestPydanticModels:
    """Tests for Pydantic request/response models."""

    def test_run_request_defaults(self):
        """Test RunRequest model defaults."""
        from api.app import RunRequest

        request = RunRequest()

        assert request.force is False

    def test_run_request_with_values(self):
        """Test RunRequest model with values."""
        from api.app import RunRequest

        request = RunRequest(
            force=True,
        )

        assert request.force is True

    def test_analyze_request_defaults(self):
        """Test AnalyzeRequest model defaults."""
        from api.app import AnalyzeRequest

        request = AnalyzeRequest()

        assert request.days is None
        assert request.test is False
        assert request.email_receivers is None

    def test_analyze_request_with_values(self):
        """Test AnalyzeRequest model with values."""
        from api.app import AnalyzeRequest

        request = AnalyzeRequest(
            days=14,
            test=True,
            email_receivers=["test@example.com"],
        )

        assert request.days == 14
        assert request.test is True
        assert request.email_receivers == ["test@example.com"]

    def test_job_start_response_model(self):
        """Test JobStartResponse model."""
        from api.app import JobStartResponse

        response = JobStartResponse(
            job_id="123",
            status="pending",
            message="Job started",
            config_key="test",
            config_name="Test",
        )

        assert response.job_id == "123"
        assert response.status == "pending"
        assert response.config_key == "test"
        assert response.config_name == "Test"
        assert "started" in response.message.lower()

    def test_schedule_config_model(self):
        """Test ScheduleConfig model."""
        from api.app import ScheduleConfig

        schedule = ScheduleConfig(
            key="test",
            name="Test",
            is_active=True,
            daily_scrape={"enabled": True, "cron": "5 0 * * *"},
            weekly_analysis={"enabled": True, "cron": "0 9 * * 1"},
        )

        assert schedule.key == "test"
        assert schedule.daily_scrape["enabled"] is True

    def test_health_response_model(self):
        """Test HealthResponse model."""
        from api.app import HealthResponse

        response = HealthResponse(
            status="ok",
            message="Running",
            scheduler_enabled=True,
            scheduler_running=True,
            scheduler_pid=123,
        )

        assert response.status == "ok"
        assert response.message == "Running"

    def test_error_response_model(self):
        """Test ErrorResponse model."""
        from api.app import ErrorResponse

        response = ErrorResponse(error="Something went wrong")

        assert response.success is False
        assert response.error == "Something went wrong"
        assert response.errors == []
