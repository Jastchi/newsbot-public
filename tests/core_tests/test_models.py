"""Comprehensive tests for newsbot.models module."""

from datetime import datetime

import pytest

from newsbot.constants import TZ
from newsbot.models import (
    AnalysisData,
    Article,
    Config,
    DatabaseConfig,
    LLMConfig,
    LoggingConfig,
    NewsSource,
    PipelineStatus,
    PipelineStatusAgents,
    PipelineStatusConfig,
    ReportConfig,
    Results,
    SchedulerConfig,
    SentimentAnalysisDict,
    SentimentConfig,
    SentimentDifference,
    SentimentResult,
    SentimentSummary,
    Story,
    StoryAnalysis,
    StoryClusteringConfig,
    SummarizationConfig,
    SummaryItem,
)


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_sentiment_result_creation(self):
        """Test creating a SentimentResult instance."""
        result = SentimentResult(
            article_url="https://example.com",
            source="Test Source",
            polarity=0.5,
            subjectivity=0.3,
            compound=0.6,
            label="positive",
        )

        assert result.article_url == "https://example.com"
        assert result.source == "Test Source"
        assert result.polarity == 0.5
        assert result.subjectivity == 0.3
        assert result.compound == 0.6
        assert result.label == "positive"

    def test_sentiment_result_negative_values(self):
        """Test SentimentResult with negative sentiment."""
        result = SentimentResult(
            article_url="https://example.com",
            source="Test Source",
            polarity=-0.7,
            subjectivity=0.8,
            compound=-0.9,
            label="negative",
        )

        assert result.polarity == -0.7
        assert result.compound == -0.9
        assert result.label == "negative"

    def test_sentiment_result_neutral(self):
        """Test SentimentResult with neutral sentiment."""
        result = SentimentResult(
            article_url="https://example.com",
            source="Test Source",
            polarity=0.0,
            subjectivity=0.0,
            compound=0.0,
            label="neutral",
        )

        assert result.polarity == 0.0
        assert result.compound == 0.0
        assert result.label == "neutral"


class TestArticle:
    """Tests for Article dataclass."""

    def test_article_creation(self):
        """Test creating an Article instance."""
        now = datetime.now(TZ)
        article = Article(
            title="Test Title",
            content="Test content here",
            source="Test Source",
            url="https://example.com/article",
            published_date=now,
            scraped_date=now,
        )

        assert article.title == "Test Title"
        assert article.content == "Test content here"
        assert article.source == "Test Source"
        assert article.url == "https://example.com/article"
        assert article.summary is None
        assert article.sentiment is None

    def test_article_with_summary_and_sentiment(self):
        """Test Article with optional fields."""
        now = datetime.now(TZ)
        sentiment = SentimentResult(
            article_url="https://example.com",
            source="Test",
            polarity=0.5,
            subjectivity=0.3,
            compound=0.6,
            label="positive",
        )

        article = Article(
            title="Test",
            content="Content",
            source="Source",
            url="https://example.com",
            published_date=now,
            scraped_date=now,
            summary="This is a summary",
            sentiment=sentiment,
        )

        assert article.summary == "This is a summary"
        assert article.sentiment == sentiment

    def test_article_hash(self):
        """Test Article hash is based on URL."""
        now = datetime.now(TZ)
        article1 = Article(
            title="Title 1",
            content="Content 1",
            source="Source 1",
            url="https://example.com/1",
            published_date=now,
            scraped_date=now,
        )
        article2 = Article(
            title="Title 2",
            content="Content 2",
            source="Source 2",
            url="https://example.com/1",  # Same URL
            published_date=now,
            scraped_date=now,
        )

        assert hash(article1) == hash(article2)

    def test_article_equality(self):
        """Test Article equality is based on URL."""
        now = datetime.now(TZ)
        article1 = Article(
            title="Title 1",
            content="Content 1",
            source="Source 1",
            url="https://example.com/1",
            published_date=now,
            scraped_date=now,
        )
        article2 = Article(
            title="Different Title",
            content="Different Content",
            source="Different Source",
            url="https://example.com/1",  # Same URL
            published_date=now,
            scraped_date=now,
        )
        article3 = Article(
            title="Title 1",
            content="Content 1",
            source="Source 1",
            url="https://example.com/2",  # Different URL
            published_date=now,
            scraped_date=now,
        )

        assert article1 == article2
        assert article1 != article3
        assert article1 != "not an article"

    def test_article_in_set(self):
        """Test that Articles can be used in sets (using hash)."""
        now = datetime.now(TZ)
        article1 = Article(
            title="A",
            content="C",
            source="S",
            url="https://example.com/1",
            published_date=now,
            scraped_date=now,
        )
        article2 = Article(
            title="B",
            content="D",
            source="T",
            url="https://example.com/1",  # Same URL
            published_date=now,
            scraped_date=now,
        )

        articles_set = {article1, article2}
        assert len(articles_set) == 1  # Only one unique URL


class TestStory:
    """Tests for Story dataclass."""

    def test_story_creation(self):
        """Test creating a Story instance."""
        now = datetime.now(TZ)
        articles = [
            Article(
                title="Article 1",
                content="Content 1",
                source="Source A",
                url="https://example.com/1",
                published_date=now,
                scraped_date=now,
            ),
            Article(
                title="Article 2",
                content="Content 2",
                source="Source B",
                url="https://example.com/2",
                published_date=now,
                scraped_date=now,
            ),
        ]

        story = Story(
            story_id="story_1",
            title="Main Story Title",
            articles=articles,
            sources=["Source A", "Source B"],
            article_count=2,
            earliest_date=now,
            latest_date=now,
        )

        assert story.story_id == "story_1"
        assert story.title == "Main Story Title"
        assert len(story.articles) == 2
        assert story.article_count == 2
        assert story.story_summary is None

    def test_story_post_init(self):
        """Test Story post_init sets derived fields."""
        now = datetime.now(TZ)
        articles = [
            Article(
                title="Article 1",
                content="Content 1",
                source="Source A",
                url="https://example.com/1",
                published_date=now,
                scraped_date=now,
            ),
            Article(
                title="Article 2",
                content="Content 2",
                source="Source B",
                url="https://example.com/2",
                published_date=now,
                scraped_date=now,
            ),
            Article(
                title="Article 3",
                content="Content 3",
                source="Source A",  # Duplicate source
                url="https://example.com/3",
                published_date=now,
                scraped_date=now,
            ),
        ]

        story = Story(
            story_id="story_1",
            title="Main Story Title",
            articles=articles,
            sources=[],  # Will be set by post_init
            article_count=0,  # Will be set by post_init
            earliest_date=now,
            latest_date=now,
        )

        assert story.article_count == 3
        assert set(story.sources) == {"Source A", "Source B"}

    def test_story_with_summary(self):
        """Test Story with summary."""
        now = datetime.now(TZ)
        articles = [
            Article(
                title="Article 1",
                content="Content 1",
                source="Source A",
                url="https://example.com/1",
                published_date=now,
                scraped_date=now,
            ),
        ]

        story = Story(
            story_id="story_1",
            title="Main Story Title",
            articles=articles,
            sources=["Source A"],
            article_count=1,
            earliest_date=now,
            latest_date=now,
            story_summary="This is the story summary",
        )

        assert story.story_summary == "This is the story summary"


class TestTypedDicts:
    """Tests for TypedDict classes."""

    def test_summary_item(self):
        """Test SummaryItem TypedDict."""
        now = datetime.now(TZ)
        article = Article(
            title="Test",
            content="Content",
            source="Source",
            url="https://example.com",
            published_date=now,
            scraped_date=now,
        )

        item: SummaryItem = {
            "article": article,
            "summary": "Summary text",
        }

        assert item["article"] == article
        assert item["summary"] == "Summary text"

    def test_sentiment_summary(self):
        """Test SentimentSummary TypedDict."""
        sentiments = [
            SentimentResult(
                article_url="https://example.com",
                source="Test",
                polarity=0.5,
                subjectivity=0.3,
                compound=0.6,
                label="positive",
            )
        ]

        summary: SentimentSummary = {
            "avg_sentiment": 0.6,
            "label": "positive",
            "article_count": 1,
            "sentiments": sentiments,
        }

        assert summary["avg_sentiment"] == 0.6
        assert summary["label"] == "positive"
        assert summary["article_count"] == 1
        assert len(summary["sentiments"]) == 1

    def test_sentiment_analysis_dict(self):
        """Test SentimentAnalysisDict TypedDict."""
        analysis: SentimentAnalysisDict = {
            "polarity": 0.5,
            "compound": 0.6,
            "label": "positive",
            "subjectivity": 0.3,
            "positive": 0.7,
            "negative": 0.1,
            "neutral": 0.2,
            "probas": {"pos": 0.7, "neg": 0.1, "neu": 0.2},
        }

        assert analysis["polarity"] == 0.5
        assert analysis["label"] == "positive"

    def test_sentiment_difference(self):
        """Test SentimentDifference TypedDict."""
        diff: SentimentDifference = {
            "source1": "Source A",
            "source2": "Source B",
            "difference": 0.4,
            "source1_avg": 0.6,
            "source2_avg": 0.2,
        }

        assert diff["source1"] == "Source A"
        assert diff["difference"] == 0.4

    def test_story_analysis(self):
        """Test StoryAnalysis TypedDict."""
        now = datetime.now(TZ)
        article = Article(
            title="Test",
            content="Content",
            source="Source",
            url="https://example.com",
            published_date=now,
            scraped_date=now,
        )
        story = Story(
            story_id="1",
            title="Story",
            articles=[article],
            sources=["Source"],
            article_count=1,
            earliest_date=now,
            latest_date=now,
        )

        analysis: StoryAnalysis = {
            "story": story,
            "source_summaries": {},
            "source_sentiments": {},
        }

        assert analysis["story"] == story

    def test_news_source(self):
        """Test NewsSource TypedDict."""
        source: NewsSource = {
            "name": "Test News",
            "rss_url": "https://example.com/feed",
            "type": "rss",
        }

        assert source["name"] == "Test News"
        assert source["type"] == "rss"

    def test_config_types(self):
        """Test configuration TypedDicts."""
        llm: LLMConfig = {
            "provider": "ollama",
            "model": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        summarization: SummarizationConfig = {
            "two_pass_enabled": True,
            "max_articles_batch": 10,
            "article_order": "chronological",
        }

        sentiment: SentimentConfig = {
            "method": "vader",
            "comparison_threshold": 0.3,
        }

        clustering: StoryClusteringConfig = {
            "top_stories_count": 5,
            "min_sources": 2,
            "similarity_threshold": 0.3,
            "embedding_model": "Xenova/all-MiniLM-L6-v2",
        }

        report: ReportConfig = {
            "format": "html",
            "include_summaries": True,
            "lookback_days": 7,
        }

        scheduler: SchedulerConfig = {
            "weekly_analysis": {"day": 0, "hour": 9},
        }

        logging: LoggingConfig = {
            "level": "INFO",
            "format": "%(asctime)s - %(message)s",
        }

        database: DatabaseConfig = {
            "type": "sqlite",
            "path": "newsbot.db",
        }

        assert llm["model"] == "llama2"
        assert summarization["two_pass_enabled"] is True
        assert sentiment["method"] == "vader"
        assert clustering["min_sources"] == 2
        assert report["format"] == "html"
        assert scheduler["weekly_analysis"]["day"] == 0
        assert logging["level"] == "INFO"
        assert database["type"] == "sqlite"

    def test_full_config(self):
        """Test full Config TypedDict."""
        config: Config = {
            "name": "TestConfig",
            "country": "US",
            "language": "en",
            "news_sources": [
                {
                    "name": "Test News",
                    "rss_url": "https://example.com/feed",
                    "type": "rss",
                }
            ],
            "llm": {
                "model": "llama2",
                "base_url": "http://localhost:11434",
            },
            "report": {
                "format": "html",
                "lookback_days": 7,
            },
        }

        assert config["name"] == "TestConfig"
        assert config["country"] == "US"
        assert len(config["news_sources"]) == 1

    def test_analysis_data(self):
        """Test AnalysisData TypedDict."""
        now = datetime.now(TZ)
        data: AnalysisData = {
            "success": True,
            "articles_count": 50,
            "stories_count": 10,
            "duration": 123.45,
            "timestamp": "2025-12-08 10:00:00",
            "format": "html",
            "config_name": "TestConfig",
            "from_date": now,
            "to_date": now,
            "email_receivers_override": ["test@example.com"],
        }

        assert data["success"] is True
        assert data["articles_count"] == 50
        assert data["email_receivers_override"] == ["test@example.com"]

    def test_pipeline_status(self):
        """Test PipelineStatus TypedDicts."""
        agents: PipelineStatusAgents = {
            "scraper": "ready",
            "summarizer": "ready",
            "sentiment_analyzer": "ready",
            "report_generator": "ready",
        }

        config: PipelineStatusConfig = {
            "country": "US",
            "sources": 5,
            "llm_model": "llama2",
        }

        status: PipelineStatus = {
            "agents": agents,
            "config": config,
        }

        assert status["agents"]["scraper"] == "ready"
        assert status["config"]["sources"] == 5


class TestResults:
    """Tests for Results class."""

    def test_results_initialization(self):
        """Test Results object initialization."""
        results = Results()

        assert results.success is False
        assert results.duration == 0.0
        assert results.articles_count == 0
        assert results.stories_count == 0
        assert results.saved_to_db == 0
        assert results.top_stories == []
        assert results.story_analyses == []
        assert results.errors == []
        assert results.report_path == ""
        assert isinstance(results.start_time, datetime)

    def test_results_can_be_modified(self):
        """Test Results object can be modified."""
        results = Results()

        results.success = True
        results.articles_count = 42
        results.stories_count = 7
        results.duration = 120.5
        results.saved_to_db = 42
        results.report_path = "/path/to/report.html"
        results.errors = ["Error 1", "Error 2"]

        assert results.success is True
        assert results.articles_count == 42
        assert results.stories_count == 7
        assert results.duration == 120.5
        assert results.saved_to_db == 42
        assert results.report_path == "/path/to/report.html"
        assert len(results.errors) == 2

    def test_results_with_stories(self):
        """Test Results with story data."""
        now = datetime.now(TZ)
        article = Article(
            title="Test",
            content="Content",
            source="Source",
            url="https://example.com",
            published_date=now,
            scraped_date=now,
        )
        story = Story(
            story_id="1",
            title="Story",
            articles=[article],
            sources=["Source"],
            article_count=1,
            earliest_date=now,
            latest_date=now,
        )

        results = Results()
        results.top_stories = [story]
        results.stories_count = 1

        assert len(results.top_stories) == 1
        assert results.stories_count == 1
        assert results.top_stories[0].story_id == "1"
