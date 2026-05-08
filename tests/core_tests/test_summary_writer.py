"""Tests for the SummaryWriter module."""

import json
from datetime import datetime

import pytest

from newsbot.constants import TZ
from newsbot.models import Article, Story
from newsbot.summary_writer import SummaryWriter


@pytest.fixture
def news_config(db):
    """Create a test NewsConfig in the Django database."""
    from web.newsserver.models import NewsConfig

    return NewsConfig.objects.create(
        key="test_config",
        display_name="Test Config",
    )


@pytest.mark.django_db
def test_save_scrape_summary(news_config):
    """Test saving a scrape summary."""
    from web.newsserver.models import ScrapeSummary

    writer = SummaryWriter()

    writer.save_scrape_summary(
        config_key="test_config",
        success=True,
        duration=10.5,
        articles_scraped=100,
        articles_saved=50,
        errors=["Error 1", "Error 2"],
    )

    summary = ScrapeSummary.objects.first()

    assert summary is not None
    assert summary.success is True
    assert summary.duration == 10.5
    assert summary.articles_scraped == 100
    assert summary.articles_saved == 50
    assert summary.error_count == 2
    assert json.loads(summary.errors) == ["Error 1", "Error 2"]
    assert summary.config == news_config


@pytest.mark.django_db
def test_save_scrape_summary_unknown_config():
    """Test saving a scrape summary for an unknown config."""
    from web.newsserver.models import ScrapeSummary

    writer = SummaryWriter()

    writer.save_scrape_summary(
        config_key="unknown_config",
        success=True,
        duration=10.5,
        articles_scraped=100,
        articles_saved=50,
        errors=[],
    )

    summary = ScrapeSummary.objects.first()

    assert summary is not None
    assert summary.config is None


@pytest.mark.django_db
def test_save_analysis_summary(news_config):
    """Test saving an analysis summary."""
    from web.newsserver.models import AnalysisSummary

    writer = SummaryWriter()

    now = datetime.now(TZ)
    article = Article(
        title="A1",
        content="content",
        source="Source A",
        url="https://example.com/a1",
        published_date=now,
        scraped_date=now,
    )
    top_stories: list[Story] = [
        Story(
            story_id="s1",
            title="Story 1",
            articles=[article],
            earliest_date=now,
            latest_date=now,
        ),
    ]

    writer.save_analysis_summary(
        config_key="test_config",
        success=True,
        duration=20.0,
        articles_analyzed=200,
        stories_identified=10,
        top_stories=top_stories,
        errors=[],
    )

    summary = AnalysisSummary.objects.first()

    assert summary is not None
    assert summary.success is True
    assert summary.duration == 20.0
    assert summary.articles_analyzed == 200
    assert summary.stories_identified == 10

    saved_stories = json.loads(summary.top_stories)
    assert len(saved_stories) == 1
    assert saved_stories[0]["title"] == "Story 1"
    assert saved_stories[0]["sources"] == ["Source A"]

    assert summary.error_count == 0
    assert summary.errors == ""
    assert summary.config == news_config


@pytest.mark.django_db
def test_save_scrape_summary_empty_errors(news_config):
    """Test saving a scrape summary with no errors."""
    from web.newsserver.models import ScrapeSummary

    writer = SummaryWriter()

    writer.save_scrape_summary(
        config_key="test_config",
        success=True,
        duration=5.0,
        articles_scraped=50,
        articles_saved=50,
        errors=[],
    )

    summary = ScrapeSummary.objects.first()

    assert summary is not None
    assert summary.error_count == 0
    assert summary.errors == ""
