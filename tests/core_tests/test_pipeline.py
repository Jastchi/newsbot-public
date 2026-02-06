import logging
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from newsbot.managers import AgentManager, DatabaseManager
from newsbot.models import Article, SentimentResult
from newsbot.pipeline import PipelineOrchestrator, Results


@pytest.fixture(autouse=True)
def _quiet_logs():
    logging.getLogger("newsbot").setLevel(logging.CRITICAL)
    logging.getLogger("newsbot.pipeline").setLevel(logging.CRITICAL)


@pytest.fixture
def orchestrator(monkeypatch, sample_config):
    def fake_init(self, config, config_key="", news_config=None):
        self.config = config
        self.config_key = config_key
        self.email_receivers_override = None
        self._news_config = Mock()
        # Create mock managers
        self.agent_manager = Mock(spec=AgentManager)
        self.database_manager = Mock(spec=DatabaseManager)

    monkeypatch.setattr(PipelineOrchestrator, "__init__", fake_init)
    return PipelineOrchestrator(sample_config)


def _article(index: int = 1) -> Article:
    return Article(
        title=f"Title {index}",
        content=f"Content {index}",
        source="SourceA",
        url=f"https://example.com/{index}",
        published_date=datetime.now(),
        scraped_date=datetime.now(),
    )


class TestDailyScrape:
    def test_skip_when_already_scraped(self, orchestrator):
        orchestrator.database_manager.has_scraped_today.return_value = True

        result = orchestrator.run_daily_scrape(force=False)

        assert result.success is True
        assert result.articles_count == 0
        assert result.errors == []

    def test_scrape_and_save(self, orchestrator):
        articles = [_article()]
        orchestrator.database_manager.has_scraped_today.return_value = False
        orchestrator.agent_manager.scraper.scrape_all_sources.return_value = (
            articles
        )
        orchestrator.database_manager.save_articles.return_value = 1

        result = orchestrator.run_daily_scrape(force=False)

        assert result.success is True
        assert result.saved_to_db == 1
        orchestrator.agent_manager.scraper.scrape_all_sources.assert_called_once()


class TestWeeklyAnalysis:
    def test_no_articles_in_db(self, orchestrator):
        orchestrator.database_manager.load_articles.side_effect = RuntimeError(
            "db missing",
        )

        out = orchestrator.run_weekly_analysis(days_back=3)

        assert out.success is False
        assert "db missing" in out.errors[0]

    def test_full_weekly_flow(self, orchestrator):
        articles = [_article()]
        top_stories = [SimpleNamespace(title="Story", articles=articles)]
        story_analyses = [
            {
                "story": top_stories[0],
                "source_summaries": {
                    "SourceA": [{"article": articles[0], "summary": "S"}]
                },
            }
        ]

        orchestrator.database_manager.load_articles.return_value = articles
        orchestrator._identify_top_stories = Mock(return_value=top_stories)
        orchestrator._summarize_stories = Mock(return_value=story_analyses)
        orchestrator._analyze_sentiment = Mock()
        orchestrator._generate_report = Mock(return_value="weekly.html")

        with patch("newsbot.pipeline.run_hooks", Mock()):
            out = orchestrator.run_weekly_analysis(days_back=3)

        assert out.success is True
        assert out.report_path == "weekly.html"
        orchestrator.database_manager.update_articles_with_analysis.assert_called_once_with(
            articles
        )


class TestDatabaseManager:
    @pytest.fixture
    def news_config(self):
        """Create a NewsConfig for testing."""
        from utilities.django_models import NewsConfig

        config, _ = NewsConfig.objects.get_or_create(
            key="test_pipeline",
            defaults={"display_name": "Test Pipeline Config"},
        )
        return config

    @pytest.fixture
    def database_manager(self, news_config):
        """Create a DatabaseManager for testing."""
        return DatabaseManager(news_config)

    @pytest.mark.django_db
    def test_save_articles_saves_to_database(self, database_manager):
        """Test that save_articles saves articles to Django database."""
        import uuid

        from newsbot.constants import TZ
        from utilities.django_models import Article as DjangoArticle

        # Create article with unique URL to avoid conflicts
        unique_url = f"https://example.com/{uuid.uuid4()}"
        unique_article = Article(
            title="Test Article",
            content="Test content",
            source="Test Source",
            url=unique_url,
            published_date=datetime.now(TZ),
            scraped_date=datetime.now(TZ),
        )
        articles = [unique_article]
        saved = database_manager.save_articles(articles)

        # Since we use a unique URL, the article should be saved
        assert saved == 1

        # Verify the article was actually saved to the database
        saved_article = DjangoArticle.objects.filter(url=unique_url).first()
        assert saved_article is not None
        assert saved_article.title == "Test Article"
        assert saved_article.content == "Test content"
        assert saved_article.source == "Test Source"
        assert saved_article.config == database_manager.news_config

    @pytest.mark.django_db
    def test_save_articles_handles_exception(
        self, monkeypatch, database_manager
    ):
        """Test that save_articles handles database errors gracefully."""
        from django.db import DatabaseError
        from utilities.django_models import Article as DjangoArticle

        # Mock DjangoArticle.objects.filter to raise an exception
        def mock_filter(*args, **kwargs):
            raise DatabaseError("Database error")

        monkeypatch.setattr(
            DjangoArticle.objects,
            "filter",
            mock_filter,
        )

        saved = database_manager.save_articles([_article()])

        assert saved == 0

    @pytest.mark.django_db
    def test_update_articles_handles_exception(
        self, monkeypatch, database_manager
    ):
        """Test that update_articles_with_analysis handles errors."""
        from django.db import DatabaseError
        from utilities.django_models import Article as DjangoArticle

        # Mock DjangoArticle.objects.filter to raise an exception
        def mock_filter(*args, **kwargs):
            raise DatabaseError("Database error")

        monkeypatch.setattr(
            DjangoArticle.objects,
            "filter",
            mock_filter,
        )

        result = database_manager.update_articles_with_analysis([_article()])
        assert result == 0

    @pytest.mark.django_db
    def test_has_scraped_today_handles_exception(
        self, monkeypatch, database_manager
    ):
        """Test that has_scraped_today handles database errors gracefully."""
        from django.db import DatabaseError
        from utilities.django_models import Article as DjangoArticle

        # Mock DjangoArticle.objects.filter to raise an exception
        def mock_filter(*args, **kwargs):
            raise DatabaseError("Database error")

        monkeypatch.setattr(
            DjangoArticle.objects,
            "filter",
            mock_filter,
        )

        assert database_manager.has_scraped_today() is False


class TestPipelineEdgeCases:
    def test_run_daily_scrape_no_articles(self, orchestrator):
        orchestrator.database_manager.has_scraped_today.return_value = False
        orchestrator.agent_manager.scraper.scrape_all_sources.return_value = []

        result = orchestrator.run_daily_scrape()

        assert result.success is False
        assert result.errors == ["No articles scraped"]

    def test_weekly_analysis_no_articles(self, orchestrator):
        orchestrator.database_manager.load_articles.return_value = []

        result = orchestrator.run_weekly_analysis(days_back=2)

        assert result.success is False
        assert "No articles in database" in result.errors[0]


class TestPipelineInitialization:
    """Tests for PipelineOrchestrator initialization."""

    @pytest.mark.django_db
    def test_init_with_news_config(self, sample_config):
        """Test initialization when news_config is provided."""
        from utilities.django_models import NewsConfig

        news_config, _ = NewsConfig.objects.get_or_create(
            key="test_init",
            defaults={"display_name": "Test Init Config"},
        )

        orchestrator = PipelineOrchestrator(
            sample_config, config_key="test_init", news_config=news_config
        )

        assert orchestrator.config is sample_config
        assert orchestrator.config_key == "test_init"
        assert orchestrator.news_config is news_config

    @pytest.mark.django_db
    def test_init_looks_up_news_config_by_key(self, sample_config):
        """Test that NewsConfig is looked up by config_key when news_config is None."""
        from utilities.django_models import NewsConfig

        # Create NewsConfig with key
        news_config, _ = NewsConfig.objects.get_or_create(
            key="test_technology",
            defaults={"display_name": "Test Technology Config"},
        )

        orchestrator = PipelineOrchestrator(
            sample_config, config_key="test_technology", news_config=None
        )

        assert orchestrator.news_config == news_config

    @pytest.mark.django_db
    def test_init_raises_error_when_news_config_not_found(self, sample_config):
        """Test that ValueError is raised when NewsConfig is not found."""
        with pytest.raises(ValueError, match="NewsConfig not found"):
            PipelineOrchestrator(
                sample_config, config_key="nonexistent", news_config=None
            )

    @pytest.mark.django_db
    def test_init_handles_empty_config_key(self, sample_config):
        """Test initialization with empty config_key."""
        from utilities.django_models import NewsConfig

        # With empty config_key, lookup should return None, causing ValueError
        with pytest.raises(ValueError, match="NewsConfig not found"):
            PipelineOrchestrator(sample_config, config_key="", news_config=None)

    @pytest.mark.django_db
    def test_lookup_news_config_handles_exception(self, sample_config, monkeypatch):
        """Test that _lookup_news_config handles exceptions gracefully."""
        from django.db import DatabaseError
        from utilities.django_models import NewsConfig

        # Mock NewsConfig.objects.filter to raise an exception
        def mock_filter(*args, **kwargs):
            raise DatabaseError("Database error")

        monkeypatch.setattr(NewsConfig.objects, "filter", mock_filter)

        orchestrator = PipelineOrchestrator.__new__(PipelineOrchestrator)
        orchestrator.config = sample_config
        orchestrator.config_key = "test"
        orchestrator.email_receivers_override = None

        # _lookup_news_config should return None on exception
        result = orchestrator._lookup_news_config("test")

        assert result is None

    def test_news_config_property(self, orchestrator):
        """Test that news_config property returns _news_config."""
        mock_config = Mock()
        orchestrator._news_config = mock_config

        assert orchestrator.news_config is mock_config


class TestPipelineInternalMethods:
    """Tests for internal pipeline methods."""

    def test_scrape_articles_logs_when_articles_exist(self, orchestrator, caplog):
        """Test that _scrape_articles logs when articles are scraped."""
        articles = [_article(1), _article(2)]
        orchestrator.agent_manager.scraper.scrape_all_sources.return_value = articles

        with caplog.at_level(logging.INFO, logger="newsbot.pipeline"):
            result = orchestrator._scrape_articles()

        assert len(result) == 2
        assert "Scraped 2 articles from" in caplog.text

    def test_scrape_articles_handles_empty_result(self, orchestrator):
        """Test that _scrape_articles handles empty article list."""
        orchestrator.agent_manager.scraper.scrape_all_sources.return_value = []

        result = orchestrator._scrape_articles()

        assert result == []

    def test_identify_top_stories_logs_and_calls_clustering(self, orchestrator):
        """Test that _identify_top_stories logs and calls clustering agent."""
        articles = [_article(1), _article(2)]
        top_stories = [SimpleNamespace(title="Story 1", articles=articles)]
        orchestrator.agent_manager.story_clustering.identify_top_stories.return_value = (
            top_stories
        )

        result = orchestrator._identify_top_stories(articles, top_n=5)

        assert result == top_stories
        orchestrator.agent_manager.story_clustering.identify_top_stories.assert_called_once_with(
            articles, top_n=5
        )

    def test_summarize_stories_calls_summarizer_and_groups(self, orchestrator):
        """Test that _summarize_stories calls summarizer and groups summaries."""
        articles = [_article(1), _article(2)]
        articles[0].summary = "Summary 1"
        articles[1].summary = "Summary 2"

        story = SimpleNamespace(title="Test Story", articles=articles)
        top_stories = [story]

        orchestrator.agent_manager.summarizer.summarize_story = Mock()

        result = orchestrator._summarize_stories(top_stories)

        # Verify summarizer was called
        orchestrator.agent_manager.summarizer.summarize_story.assert_called_once_with(
            story
        )

        # Verify result structure
        assert len(result) == 1
        assert "story" in result[0]
        assert "source_summaries" in result[0]
        assert result[0]["story"] is story

    def test_analyze_sentiment_analyzes_per_article(self, orchestrator):
        """Test that _analyze_sentiment analyzes sentiment per article."""
        articles = [_article(1), _article(2)]
        story = SimpleNamespace(title="Test Story", articles=articles)
        story_analyses = [{"story": story}]

        # Mock sentiment analyzer
        sentiment1 = SentimentResult(
            article_url=articles[0].url,
            source=articles[0].source,
            polarity=0.5,
            subjectivity=0.6,
            compound=0.7,
            label="positive",
        )
        sentiment2 = SentimentResult(
            article_url=articles[1].url,
            source=articles[1].source,
            polarity=-0.3,
            subjectivity=0.4,
            compound=-0.4,
            label="negative",
        )

        orchestrator.agent_manager.sentiment_analyzer.analyze_article.side_effect = [
            sentiment1,
            sentiment2,
        ]

        orchestrator._analyze_sentiment(story_analyses)

        # Verify sentiment was analyzed for each article
        assert orchestrator.agent_manager.sentiment_analyzer.analyze_article.call_count == 2
        assert articles[0].sentiment is sentiment1
        assert articles[1].sentiment is sentiment2

        # Verify source_sentiments were added to analysis
        assert "source_sentiments" in story_analyses[0]
        source_sentiments = story_analyses[0]["source_sentiments"]
        assert isinstance(source_sentiments, dict)
        assert "SourceA" in source_sentiments

    def test_generate_report_calls_report_generator(self, orchestrator):
        """Test that _generate_report calls report generator."""
        story_analyses = [{"story": SimpleNamespace(title="Story", articles=[])}]
        orchestrator.agent_manager.report_generator.generate_top_stories_report.return_value = (
            "report.html",
            "/path/to/report.html",
        )

        result = orchestrator._generate_report(story_analyses)

        orchestrator.agent_manager.report_generator.generate_top_stories_report.assert_called_once_with(
            story_analyses
        )
        assert result == "/path/to/report.html"

    def test_get_pipeline_status_returns_status_dict(self, orchestrator):
        """Test that get_pipeline_status returns proper status dictionary."""
        orchestrator.agent_manager.get_agent_status.return_value = {
            "scraper": "initialized",
            "story_clustering": "not_initialized",
        }

        status = orchestrator.get_pipeline_status()

        assert "agents" in status
        assert "config" in status
        assert status["agents"]["scraper"] == "initialized"
        assert "country" in status["config"]
        assert "sources" in status["config"]
        assert "llm_model" in status["config"]
