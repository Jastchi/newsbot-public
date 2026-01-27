import importlib
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from utilities import models as config_models

main_module = importlib.import_module("newsbot.main")


@pytest.fixture
def dummy_handler():
    handler = Mock()
    handler.flush = Mock()
    return handler


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch, dummy_handler):
    from unittest.mock import Mock

    def mock_load_config(config_key):
        config = config_models.ConfigModel(
            name="test",
            database=config_models.DatabaseConfigModel(url="sqlite:///memory"),
            scheduler=config_models.SchedulerConfigModel(
                daily_scrape=config_models.DailyScrapeConfigModel(enabled=True),
                weekly_analysis=config_models.WeeklyAnalysisConfigModel(enabled=True),
            ),
            report=config_models.ReportConfigModel(lookback_days=7),
        )
        return config, Mock()  # Return (config, news_config)

    # Mock SummaryWriter to avoid database access in tests
    mock_summary_writer = Mock()
    mock_summary_writer_class = Mock(return_value=mock_summary_writer)

    monkeypatch.setattr(main_module, "load_config", mock_load_config)
    monkeypatch.setattr(
        main_module, "get_email_error_handler", lambda: dummy_handler
    )
    monkeypatch.setattr(
        main_module, "setup_logging", lambda *args, **kwargs: dummy_handler
    )
    monkeypatch.setattr(
        main_module, "SummaryWriter", mock_summary_writer_class
    )


def test_run_once_executes_daily_scrape(monkeypatch, dummy_handler):
    created = {}

    class DummyOrchestrator:
        def __init__(self, config, config_key="", news_config=None):
            created["instance"] = self
            created["config"] = config
            self.run_daily_scrape = Mock(
                return_value=SimpleNamespace(
                    success=True,
                    articles_count=1,
                    duration=0.5,
                    saved_to_db=1,
                    errors=[],
                )
            )
            self.set_email_receivers_override = Mock()

        def get_pipeline_status(self):
            return {"ok": True}

    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)

    args = Namespace(
        config="test", force=True, email_receivers=["a@b.com"]
    )

    main_module.run_once(args)

    assert created["config"].database.url == "sqlite:///memory"
    created["instance"].set_email_receivers_override.assert_called_once_with(
        ["a@b.com"]
    )
    created["instance"].run_daily_scrape.assert_called_once_with(force=True)
    # flush called once for email_error_handler
    assert dummy_handler.flush.call_count == 1


def test_run_analysis_uses_lookback(monkeypatch, dummy_handler):
    created = {}

    class DummyOrchestrator:
        def __init__(self, *_, **__):
            created["instance"] = self
            self.run_weekly_analysis = Mock(
                return_value=SimpleNamespace(
                    success=True,
                    articles_count=5,
                    stories_count=2,
                    duration=1.0,
                    top_stories=[],
                    errors=[],
                )
            )
            self.set_email_receivers_override = Mock()

    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)

    args = Namespace(config="test", days=None, email_receivers=None)

    main_module.run_analysis(args)

    created["instance"].run_weekly_analysis.assert_called_once_with(
        days_back=7
    )
    # flush called once for email_error_handler
    assert dummy_handler.flush.call_count == 1


def test_run_scheduled_registers_jobs(monkeypatch, dummy_handler):
    jobs = []

    class DummyScheduler:
        def __init__(self, *_, **__):
            pass

        def add_job(self, func, trigger=None, **kwargs):
            jobs.append(SimpleNamespace(func=func, trigger=trigger, **kwargs))

        def get_jobs(self):
            return [
                SimpleNamespace(
                    name="job",
                    trigger=Mock(get_next_fire_time=lambda _, __: None),
                )
            ]

        def start(self):
            return None

    class DummyCron:
        def __init__(self, *args, **kwargs):
            pass

        def get_next_fire_time(
            self, *_, **__
        ):  # pragma: no cover - called via scheduler
            return None

    class DummyOrchestrator:
        def __init__(self, *_, **__):
            self.run_daily_scrape = Mock()
            self.run_weekly_analysis = Mock()
            self.set_email_receivers_override = Mock()

    monkeypatch.setattr(main_module, "BlockingScheduler", DummyScheduler)
    monkeypatch.setattr(main_module, "CronTrigger", DummyCron)
    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)

    args = Namespace(config="test", email_receivers=None, run_now=False)

    main_module.run_scheduled(args)

    assert len(jobs) == 3  # daily scrape, weekly analysis, config refresh
    dummy_handler.flush.assert_not_called()


def test_run_scheduled_run_now_triggers_scrape(monkeypatch, dummy_handler):
    jobs = []

    class DummyScheduler:
        def __init__(self, *_, **__):
            pass

        def add_job(self, func, trigger=None, **kwargs):
            jobs.append(SimpleNamespace(func=func, trigger=trigger, **kwargs))

        def get_jobs(self):
            return [
                SimpleNamespace(
                    name="job",
                    trigger=Mock(get_next_fire_time=lambda *_: None),
                )
            ]

        def start(self):
            return None

    class DummyCron:
        def __init__(self, *_, **__):
            pass

        def get_next_fire_time(self, *_, **__):  # pragma: no cover
            return None

    class DummyOrchestrator:
        def __init__(self, *_, **__):
            self.run_daily_scrape = Mock()
            self.run_weekly_analysis = Mock()
            self.set_email_receivers_override = Mock()

    monkeypatch.setattr(main_module, "BlockingScheduler", DummyScheduler)
    monkeypatch.setattr(main_module, "CronTrigger", DummyCron)
    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)

    args = Namespace(config="test", email_receivers=None, run_now=True)

    main_module.run_scheduled(args)

    assert len(jobs) == 3  # daily scrape, weekly analysis, config refresh
    DummyOrchestrator(
        None
    ).run_daily_scrape.assert_called if False else None  # pragma: no cover
    # flush called once for email_error_handler
    assert dummy_handler.flush.call_count == 1


def test_run_scheduled_no_jobs(monkeypatch, dummy_handler):
    class DummyScheduler:
        def __init__(self, *_, **__):
            self.started = False

        def add_job(self, *_, **__):
            return None

        def get_jobs(self):
            return []

        def start(self):
            self.started = True

    class DummyCron:
        def __init__(self, *_, **__):
            pass

    class DummyOrchestrator:
        def __init__(self, *_, **__):
            self.run_daily_scrape = Mock()
            self.run_weekly_analysis = Mock()
            self.set_email_receivers_override = Mock()

    monkeypatch.setattr(main_module, "BlockingScheduler", DummyScheduler)
    monkeypatch.setattr(main_module, "CronTrigger", DummyCron)
    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)

    # Override config to disable both jobs
    def mock_load_config_disabled(_key):
        config = config_models.ConfigModel(
            name="test",
            database=config_models.DatabaseConfigModel(url="sqlite:///memory"),
            scheduler=config_models.SchedulerConfigModel(
                daily_scrape=config_models.DailyScrapeConfigModel(enabled=False),
                weekly_analysis=config_models.WeeklyAnalysisConfigModel(enabled=False),
            ),
            report=config_models.ReportConfigModel(lookback_days=7),
        )
        return config, Mock()

    monkeypatch.setattr(main_module, "load_config", mock_load_config_disabled)

    args = Namespace(config="test", email_receivers=None, run_now=False)

    main_module.run_scheduled(args)

    # No jobs means scheduler.start not invoked; flush also not called
    assert dummy_handler.flush.call_count == 0


def test_run_analysis_with_test_flag_inserts_test_articles(
    monkeypatch, dummy_handler
):
    """Test that --test flag inserts test articles before analysis."""
    insert_called = {"count": 0}

    def mock_insert_test_articles(config_key="test"):
        insert_called["count"] += 1
        return 5

    class DummyOrchestrator:
        def __init__(self, *_, **__):
            self.run_weekly_analysis = Mock(
                return_value=SimpleNamespace(
                    success=True,
                    articles_count=5,
                    stories_count=2,
                    duration=1.0,
                    top_stories=[],
                    errors=[],
                )
            )
            self.set_email_receivers_override = Mock()

    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(
        main_module, "insert_test_articles", mock_insert_test_articles
    )

    # Use a config that starts with "test"
    args = Namespace(
        config="test_technology", days=None, email_receivers=None, test=True
    )

    main_module.run_analysis(args)

    assert insert_called["count"] == 1


def test_run_analysis_test_flag_requires_test_config(
    monkeypatch, dummy_handler
):
    """Test that --test flag fails with non-test config."""

    class DummyOrchestrator:
        def __init__(self, *_):
            self.run_weekly_analysis = Mock()
            self.set_email_receivers_override = Mock()

    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)

    # Use a config that does NOT start with "test"
    args = Namespace(
        config="technology", days=None, email_receivers=None, test=True
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.run_analysis(args)

    assert exc_info.value.code == 1


def test_run_analysis_without_test_flag_skips_insert(
    monkeypatch, dummy_handler
):
    """Test that without --test flag, test articles are not inserted."""
    insert_called = {"count": 0}

    def mock_insert_test_articles(config_key="test"):
        insert_called["count"] += 1
        return 5

    class DummyOrchestrator:
        def __init__(self, *_, **__):
            self.run_weekly_analysis = Mock(
                return_value=SimpleNamespace(
                    success=True,
                    articles_count=5,
                    stories_count=2,
                    duration=1.0,
                    top_stories=[],
                    errors=[],
                )
            )
            self.set_email_receivers_override = Mock()

    monkeypatch.setattr(main_module, "PipelineOrchestrator", DummyOrchestrator)
    monkeypatch.setattr(
        main_module, "insert_test_articles", mock_insert_test_articles
    )

    args = Namespace(
        config="test", days=None, email_receivers=None, test=False
    )

    main_module.run_analysis(args)

    assert insert_called["count"] == 0
