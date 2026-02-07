"""Pytest configuration for Django tests."""

import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add the src directory to the path FIRST
# This ensures 'web' resolves to src/web/ (our package)
# Do NOT add src/web to path - this causes duplicate module imports
# where the same module can be imported as both 'web.newsserver' and 'newsserver'
src_path = Path(__file__).resolve().parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Ensure DJANGO_SETTINGS_MODULE is set before any Django imports
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web.web.settings")


@pytest.fixture(scope="session")
def django_db_setup(django_db_setup, django_db_blocker):
    """Set up the test database with migrations."""
    with django_db_blocker.unblock():
        from django.core.management import call_command

        call_command("migrate", "--run-syncdb", verbosity=0)


@pytest.fixture
def admin_user(db):
    """Create or get a superuser (Subscriber with is_staff/is_superuser) for admin access."""
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user, created = User.objects.get_or_create(
        email="admin@example.com",
        defaults={
            "is_staff": True,
            "is_superuser": True,
            "first_name": "",
            "last_name": "",
        },
    )
    if created:
        user.set_password("admin123")
        user.save()
    return user


@pytest.fixture
def sample_news_configs(db):
    """Create sample NewsConfig instances."""
    from web.newsserver.models import NewsConfig

    configs = []
    config_data = [
        {"key": "technology", "display_name": "Technology News", "country": "US"},
        {"key": "world", "display_name": "World News", "country": "US"},
    ]

    for data in config_data:
        config = NewsConfig.objects.create(
            key=data["key"],
            display_name=data["display_name"],
            country=data["country"],
            language="en",
            is_active=True,
        )
        configs.append(config)

    return configs


@pytest.fixture
def sample_news_sources(db, sample_news_configs):
    """Create sample NewsSource instances linked to configs."""
    from web.newsserver.models import NewsSource

    sources = []
    source_data = [
        {
            "name": "Example News Source 1",
            "url": "https://example.com/feed1",
        },
        {
            "name": "Example News Source 2",
            "url": "https://example.com/feed2",
        },
        {
            "name": "Example News Source 3",
            "url": "https://example.com/feed3",
        },
        {
            "name": "Example News Source 4",
            "url": "https://example.com/feed4",
        },
        {
            "name": "Example News Source 5",
            "url": "https://example.com/feed5",
        },
    ]

    for data in source_data:
        source = NewsSource.objects.create(
            name=data["name"],
            url=data["url"],
            type="rss",
        )
        sources.append(source)

    # Link sources to configs
    technology_config = sample_news_configs[0]
    world_config = sample_news_configs[1]

    technology_config.news_sources.add(sources[0], sources[1], sources[2], sources[3])
    world_config.news_sources.add(sources[4])

    return sources


@pytest.fixture
def sample_articles(db, sample_news_configs):
    """Create sample Article instances."""
    from django.utils import timezone

    from web.newsserver.models import Article

    articles = []
    now = timezone.now()

    article_data = [
        {
            "config_file": "technology",
            "title": "AI Startup Raises $100M in Series B",
            "source": "Example Source 1",
            "sentiment_label": "positive",
            "sentiment_score": 0.8,
        },
        {
            "config_file": "technology",
            "title": "New CPU Architecture Promises 50% Speed Boost",
            "source": "Example Source 2",
            "sentiment_label": "positive",
            "sentiment_score": 0.65,
        },
        {
            "config_file": "technology",
            "title": "Quantum Computing Breakthrough Achieved",
            "source": "Example Source 1",
            "sentiment_label": "positive",
            "sentiment_score": 0.75,
        },
        {
            "config_file": "world",
            "title": "Global Summit Addresses Climate Change",
            "source": "Example Source 5",
            "sentiment_label": "neutral",
            "sentiment_score": 0.1,
        },
    ]

    for i, data in enumerate(article_data):
        article = Article.objects.create(
            config_file=data["config_file"],
            title=data["title"],
            content=f"Full content for article: {data['title']}",
            summary=f"Summary of {data['title']}",
            source=data["source"],
            url=f"https://example.com/article-{i+1}",
            published_date=now - timedelta(days=i),
            sentiment_label=data["sentiment_label"],
            sentiment_score=data["sentiment_score"],
        )
        articles.append(article)

    return articles


@pytest.fixture
def sample_scrape_summaries(db, sample_news_configs):
    """Create sample ScrapeSummary instances."""
    from django.utils import timezone

    from web.newsserver.models import ScrapeSummary

    summaries = []
    now = timezone.now()

    for config in sample_news_configs:
        for days_ago in [0, 1, 2]:
            summary = ScrapeSummary.objects.create(
                config=config,
                success=True,
                duration=45.5 + days_ago * 10,
                articles_scraped=15 - days_ago * 2,
                articles_saved=12 - days_ago * 2,
                error_count=0,
                errors="",
            )
            # Manually set timestamp for variety
            summary.timestamp = now - timedelta(days=days_ago)
            summary.save()
            summaries.append(summary)

    return summaries


@pytest.fixture
def sample_analysis_summaries(db, sample_news_configs):
    """Create sample AnalysisSummary instances."""
    from django.utils import timezone

    from web.newsserver.models import AnalysisSummary

    summaries = []
    now = timezone.now()

    for config in sample_news_configs:
        summary = AnalysisSummary.objects.create(
            config=config,
            success=True,
            duration=120.5,
            articles_analyzed=25,
            stories_identified=5,
            top_stories=json.dumps([
                "Story 1: Major development",
                "Story 2: Economic impact",
                "Story 3: Regional update",
            ]),
            error_count=0,
            errors="",
        )
        summary.timestamp = now - timedelta(days=7)
        summary.save()
        summaries.append(summary)

    return summaries


@pytest.fixture
def sample_subscribers(db, sample_news_configs):
    """Create sample Subscriber instances."""
    from web.newsserver.models import Subscriber

    subscribers = []
    subscriber_data = [
        {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
        },
        {
            "first_name": "Jane",
            "last_name": "Smith",
            "email": "jane.smith@example.com",
        },
        {
            "first_name": "Bob",
            "last_name": "Johnson",
            "email": "bob.johnson@example.com",
        },
    ]

    for i, data in enumerate(subscriber_data):
        subscriber = Subscriber.objects.create_user(
            email=data["email"],
            first_name=data["first_name"],
            last_name=data["last_name"],
            password="testpass",
            is_staff=False,
        )
        # Subscribe to different configs
        if i == 0:
            subscriber.configs.add(*sample_news_configs)
        elif i == 1:
            subscriber.configs.add(sample_news_configs[0])
        else:
            subscriber.configs.add(sample_news_configs[1])
        subscribers.append(subscriber)

    return subscribers


@pytest.fixture
def sample_log_files(tmp_path):
    """Create sample log files for the Logs view."""
    from django.conf import settings

    logs_dir = settings.BASE_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_files = []
    log_content_template = """2026-01-17 10:00:00 [INFO - newsbot.pipeline] Starting pipeline for {config}
2026-01-17 10:00:01 [INFO - newsbot.pipeline] Scraping sources...
2026-01-17 10:00:05 [INFO - newsbot.pipeline] Processing story 1/5
2026-01-17 10:00:10 [INFO - newsbot.pipeline] Processing story 2/5
2026-01-17 10:00:15 [INFO - newsbot.pipeline] Processing story 3/5
2026-01-17 10:00:20 [INFO - newsbot.pipeline] Analysis complete
2026-01-17 10:00:21 [INFO - newsbot.pipeline] Report generated successfully
"""

    for config_name in ["technology", "world"]:
        log_file = logs_dir / f"{config_name}.log"
        log_file.write_text(log_content_template.format(config=config_name))
        log_files.append(log_file)

    yield log_files

    # Cleanup
    for log_file in log_files:
        if log_file.exists():
            log_file.unlink()


@pytest.fixture
def sample_reports(tmp_path):
    """Create sample report files for the Configs view."""
    from django.conf import settings

    reports_dir = settings.REPORTS_DIR
    reports_dir.mkdir(exist_ok=True)

    report_dirs = []
    report_content = """<!DOCTYPE html>
<html>
<head><title>News Report</title></head>
<body>
<h1>Sample News Report</h1>
<p>This is a sample report for testing.</p>
</body>
</html>
"""

    for config_name in ["technology", "world"]:
        config_dir = reports_dir / config_name
        config_dir.mkdir(exist_ok=True)
        report_file = config_dir / "news_report_20260117_100000.html"
        report_file.write_text(report_content)
        report_dirs.append(config_dir)

    yield report_dirs

    # Cleanup
    for config_dir in report_dirs:
        if config_dir.exists():
            shutil.rmtree(config_dir)


@pytest.fixture
def all_sample_data(
    admin_user,
    sample_news_configs,
    sample_news_sources,
    sample_articles,
    sample_scrape_summaries,
    sample_analysis_summaries,
    sample_subscribers,
    sample_log_files,
    sample_reports,
):
    """Combine all sample data fixtures."""
    return {
        "admin_user": admin_user,
        "news_configs": sample_news_configs,
        "news_sources": sample_news_sources,
        "articles": sample_articles,
        "scrape_summaries": sample_scrape_summaries,
        "analysis_summaries": sample_analysis_summaries,
        "subscribers": sample_subscribers,
        "log_files": sample_log_files,
        "report_dirs": sample_reports,
    }


@pytest.fixture
def all_sample_data_idempotent(db, admin_user):
    """
    Same as all_sample_data but uses get_or_create so it is safe to use
    with live_server when the test DB is reused (TransactionTestCase, no rollback).
    """
    from django.utils import timezone

    from web.newsserver.models import (
        AnalysisSummary,
        Article,
        NewsConfig,
        NewsSource,
        ScrapeSummary,
        Subscriber,
        SubscriberRequest,
    )
    config_data = [
        {
            "key": "technology",
            "display_name": "Technology News",
            "country": "US",
            "scheduler_weekly_analysis_day_of_week": NewsConfig.DayOfWeek.MONDAY,
            "scheduler_weekly_analysis_hour": 12,
            "scheduler_weekly_analysis_minute": 0,
        },
        {
            "key": "world",
            "display_name": "World News",
            "country": "US",
            "scheduler_weekly_analysis_day_of_week": NewsConfig.DayOfWeek.TUESDAY,
            "scheduler_weekly_analysis_hour": 12,
            "scheduler_weekly_analysis_minute": 0,
        },
    ]
    configs = []
    for data in config_data:
        day = data.pop("scheduler_weekly_analysis_day_of_week")
        hour = data.pop("scheduler_weekly_analysis_hour")
        minute = data.pop("scheduler_weekly_analysis_minute")
        obj, _ = NewsConfig.objects.get_or_create(
            key=data["key"],
            defaults={
                "display_name": data["display_name"],
                "country": data["country"],
                "language": "en",
                "is_active": True,
                "published_for_subscription": True,
                "scheduler_weekly_analysis_enabled": True,
                "scheduler_weekly_analysis_day_of_week": day,
                "scheduler_weekly_analysis_hour": hour,
                "scheduler_weekly_analysis_minute": minute,
            },
        )
        if not obj.published_for_subscription:
            obj.published_for_subscription = True
            obj.save()
        # Ensure schedule: noon on different days (idempotent)
        if (
            not obj.scheduler_weekly_analysis_enabled
            or obj.scheduler_weekly_analysis_hour != 12
            or obj.scheduler_weekly_analysis_minute != 0
        ):
            obj.scheduler_weekly_analysis_enabled = True
            obj.scheduler_weekly_analysis_day_of_week = day
            obj.scheduler_weekly_analysis_hour = 12
            obj.scheduler_weekly_analysis_minute = 0
            obj.save()
        configs.append(obj)

    source_data = [
        ("Example News Source 1", "https://example.com/feed1"),
        ("Example News Source 2", "https://example.com/feed2"),
        ("Example News Source 3", "https://example.com/feed3"),
        ("Example News Source 4", "https://example.com/feed4"),
        ("Example News Source 5", "https://example.com/feed5"),
    ]
    sources = []
    for name, url in source_data:
        obj, _ = NewsSource.objects.get_or_create(
            url=url,
            defaults={"name": name, "type": "rss"},
        )
        sources.append(obj)
    if configs:
        configs[0].news_sources.add(sources[0], sources[1], sources[2], sources[3])
        if len(configs) > 1:
            configs[1].news_sources.add(sources[4])

    now = timezone.now()
    article_data = [
        ("technology", "AI Startup Raises $100M in Series B", 0),
        ("technology", "New CPU Architecture Promises 50% Speed Boost", 1),
        ("technology", "Quantum Computing Breakthrough Achieved", 2),
        ("world", "Global Summit Addresses Climate Change", 3),
    ]
    articles = []
    for i, (cfg_key, title, idx) in enumerate(article_data):
        config = next((c for c in configs if c.key == cfg_key), configs[0])
        url = f"https://example.com/article-idempotent-{idx + 1}"
        article, _ = Article.objects.get_or_create(
            url=url,
            config=config,
            defaults={
                "config_file": cfg_key,
                "title": title,
                "content": f"Full content for article: {title}",
                "summary": f"Summary of {title}",
                "source": f"Example Source {min(idx + 1, 5)}",
                "published_date": now - timedelta(days=idx),
                "sentiment_label": "positive" if cfg_key == "technology" else "neutral",
                "sentiment_score": 0.8 if cfg_key == "technology" else 0.1,
            },
        )
        articles.append(article)

    summaries_scrape = []
    for config in configs:
        for days_ago in [0, 1, 2]:
            ts = now - timedelta(days=days_ago)
            summary = ScrapeSummary.objects.create(
                config=config,
                success=True,
                duration=45.5 + days_ago * 10,
                articles_scraped=15 - days_ago * 2,
                articles_saved=12 - days_ago * 2,
                error_count=0,
                errors="",
            )
            summary.timestamp = ts
            summary.save()
            summaries_scrape.append(summary)

    summaries_analysis = []
    for config in configs:
        ts = now - timedelta(days=7)
        summary = AnalysisSummary.objects.create(
            config=config,
            success=True,
            duration=120.5,
            articles_analyzed=25,
            stories_identified=5,
            top_stories=json.dumps([
                "Story 1: Major development",
                "Story 2: Economic impact",
                "Story 3: Regional update",
            ]),
            error_count=0,
            errors="",
        )
        summary.timestamp = ts
        summary.save()
        summaries_analysis.append(summary)

    subscriber_data = [
        ("John", "Doe", "john.doe@example.com"),
        ("Jane", "Smith", "jane.smith@example.com"),
        ("Bob", "Johnson", "bob.johnson@example.com"),
    ]
    subscribers = []
    for i, (first_name, last_name, email) in enumerate(subscriber_data):
        sub, created = Subscriber.objects.get_or_create(
            email=email,
            defaults={
                "first_name": first_name,
                "last_name": last_name,
                "is_staff": False,
            },
        )
        if created:
            sub.set_password("testpass")
            sub.save()
        # Two published configs (technology, world); only technology is subscribed
        if configs:
            if i == 0:
                sub.configs.set([configs[0]])  # John: technology only
            elif i == 1:
                sub.configs.set([configs[0]])  # Jane: technology only
            else:
                sub.configs.set([])  # Bob: no configs (world is published but unsubscribed)
        subscribers.append(sub)

    log_files = []
    try:
        from django.conf import settings as django_settings
        logs_dir = django_settings.BASE_DIR / "logs"
        logs_dir.mkdir(exist_ok=True)
        for config_name in ["technology", "world"]:
            log_file = logs_dir / f"{config_name}.log"
            log_file.write_text(f"2026-01-17 10:00:00 [INFO] Test log for {config_name}\n")
            log_files.append(log_file)
    except Exception:
        pass

    report_dirs = []
    try:
        from django.conf import settings as django_settings
        reports_dir = getattr(django_settings, "REPORTS_DIR", Path(django_settings.BASE_DIR) / "reports")
        reports_dir = Path(reports_dir)
        reports_dir.mkdir(exist_ok=True)
        for config_name in ["technology", "world"]:
            config_dir = reports_dir / config_name
            config_dir.mkdir(exist_ok=True)
            (config_dir / "news_report_20260117_100000.html").write_text("<html><body>Report</body></html>")
            report_dirs.append(config_dir)
    except Exception:
        pass

    # One pending subscriber request (no admin_notified_at / included_in_daily_email_at)
    pending_request, _ = SubscriberRequest.objects.get_or_create(
        email="pending.subscriber@example.com",
        defaults={
            "first_name": "Pending",
            "last_name": "Subscriber",
        },
    )

    # Logged-in user (admin) is subscribed to one config so schedule screenshot shows one subscribed
    if configs:
        admin_user.configs.set([configs[0]])

    return {
        "admin_user": admin_user,
        "news_configs": configs,
        "news_sources": sources,
        "articles": articles,
        "scrape_summaries": summaries_scrape,
        "analysis_summaries": summaries_analysis,
        "subscribers": subscribers,
        "subscriber_requests": [pending_request],
        "log_files": log_files,
        "report_dirs": report_dirs,
    }


@pytest.fixture
def django_live_server(db, all_sample_data_idempotent, live_server):
    """
    Start a live Django development server for screenshot tests.
    
    Uses idempotent sample data (get_or_create) so the test works when
    the DB is reused with TransactionTestCase (no rollback).
    """
    yield {
        "url": live_server.url,
        "sample_data": all_sample_data_idempotent,
    }


@pytest.fixture
def screenshot_dir():
    """Create and return the screenshot directory with timestamp."""
    base_dir = Path(__file__).resolve().parent.parent / "screenshots"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_path = base_dir / timestamp
    screenshot_path.mkdir(parents=True, exist_ok=True)
    return screenshot_path
