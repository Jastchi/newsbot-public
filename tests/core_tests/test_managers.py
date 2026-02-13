"""Tests for manager classes."""

import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from newsbot.constants import TZ
from newsbot.managers import AgentManager, DatabaseManager
from newsbot.models import Article, SentimentResult

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def _quiet_logs():
    """Suppress logging during tests."""
    logging.getLogger("newsbot.managers").setLevel(logging.CRITICAL)


class TestAgentManager:
    """Tests for AgentManager class."""

    @pytest.fixture
    def agent_manager(self, sample_config):
        """Create an AgentManager instance for testing."""
        return AgentManager(sample_config)

    def test_scraper_lazy_initialization(self, agent_manager, sample_config):
        """Test that scraper agent is lazily initialized."""
        # Initially, scraper should not be initialized
        assert agent_manager._scraper is None

        # Access the scraper property
        scraper = agent_manager.scraper

        # Now it should be initialized
        assert agent_manager._scraper is not None
        assert scraper is agent_manager._scraper

        # Verify it's the correct type
        from newsbot.agents.scraper_agent import NewsScraperAgent

        assert isinstance(scraper, NewsScraperAgent)

        # Verify it was initialized with the config
        assert scraper.config == sample_config

    @pytest.mark.django_db
    def test_scraper_gets_exclude_url_check_when_config_has_exclude_and_db(
        self, sample_config, db
    ):
        """Scraper gets exclude_url_check when config has exclude_articles_from_config_keys and database_manager is set."""
        from utilities.django_models import NewsConfig

        news_config, _ = NewsConfig.objects.get_or_create(
            key="test_managers",
            defaults={"display_name": "Test Managers Config"},
        )
        config_with_exclude = sample_config.model_copy(
            update={"exclude_articles_from_config_keys": ["other_config"]}
        )
        manager = AgentManager(
            config_with_exclude,
            database_manager=DatabaseManager(news_config),
        )
        scraper = manager.scraper
        assert scraper.exclude_url_check is not None

    @pytest.mark.django_db
    def test_scraper_uses_url_exists_with_content_when_database_manager_set(
        self, sample_config, db
    ):
        """Scraper uses url_exists_with_content so we re-fetch when URL exists but has no content."""
        from newsbot.managers.database_manager import DatabaseManager as DM
        from utilities.django_models import NewsConfig

        news_config, _ = NewsConfig.objects.get_or_create(
            key="test_managers",
            defaults={"display_name": "Test Managers Config"},
        )
        db_manager = DatabaseManager(news_config)
        manager = AgentManager(sample_config, database_manager=db_manager)
        scraper = manager.scraper
        assert scraper.url_check is not None
        assert scraper.url_check.__func__ is DM.url_exists_with_content

    def test_scraper_exclude_url_check_none_without_config_list(
        self, sample_config
    ):
        """Scraper has exclude_url_check None when config has empty exclude list."""
        manager = AgentManager(sample_config, database_manager=Mock())
        scraper = manager.scraper
        assert scraper.exclude_url_check is None

    def test_scraper_exclude_url_check_none_without_database_manager(
        self, sample_config
    ):
        """Scraper has exclude_url_check None when database_manager is None."""
        config_with_exclude = sample_config.model_copy(
            update={"exclude_articles_from_config_keys": ["other"]}
        )
        manager = AgentManager(config_with_exclude, database_manager=None)
        scraper = manager.scraper
        assert scraper.exclude_url_check is None

    def test_story_clustering_lazy_initialization(
        self, agent_manager, sample_config
    ):
        """Test that story clustering agent is lazily initialized."""
        # Initially, story clustering should not be initialized
        assert agent_manager._story_clustering is None

        # Access the story_clustering property
        clustering = agent_manager.story_clustering

        # Now it should be initialized
        assert agent_manager._story_clustering is not None
        assert clustering is agent_manager._story_clustering

        # Verify it's the correct type
        from newsbot.agents.story_clustering_agent import StoryClusteringAgent

        assert isinstance(clustering, StoryClusteringAgent)
        # StoryClusteringAgent doesn't store config, it stores config attributes
        assert (
            clustering.story_clustering_config
            is sample_config.story_clustering
        )

    def test_summarizer_lazy_initialization(
        self, agent_manager, sample_config
    ):
        """Test that summarizer agent is lazily initialized."""
        # Initially, summarizer should not be initialized
        assert agent_manager._summarizer is None

        # Access the summarizer property
        summarizer = agent_manager.summarizer

        # Now it should be initialized
        assert agent_manager._summarizer is not None
        assert summarizer is agent_manager._summarizer

        # Verify it's the correct type
        from newsbot.agents.summarization_agent import SummarizationAgent

        assert isinstance(summarizer, SummarizationAgent)
        assert summarizer.config == sample_config

    def test_sentiment_analyzer_lazy_initialization(
        self, agent_manager, sample_config
    ):
        """Test that sentiment analyzer agent is lazily initialized."""
        # Initially, sentiment analyzer should not be initialized
        assert agent_manager._sentiment_analyzer is None

        # Access the sentiment_analyzer property
        sentiment_analyzer = agent_manager.sentiment_analyzer

        # Now it should be initialized
        assert agent_manager._sentiment_analyzer is not None
        assert sentiment_analyzer is agent_manager._sentiment_analyzer

        # Verify it's the correct type
        from newsbot.agents.sentiment_agent import SentimentAnalysisAgent

        assert isinstance(sentiment_analyzer, SentimentAnalysisAgent)
        assert sentiment_analyzer.config == sample_config

    def test_report_generator_lazy_initialization(
        self, agent_manager, sample_config
    ):
        """Test that report generator agent is lazily initialized."""
        # Initially, report generator should not be initialized
        assert agent_manager._report_generator is None

        # Access the report_generator property
        report_generator = agent_manager.report_generator

        # Now it should be initialized
        assert agent_manager._report_generator is not None
        assert report_generator is agent_manager._report_generator

        # Verify it's the correct type
        from newsbot.agents.report_agent import ReportGeneratorAgent

        assert isinstance(report_generator, ReportGeneratorAgent)
        assert report_generator.config == sample_config

    def test_agents_only_initialized_once(self, agent_manager):
        """Test that agents are only initialized once (cached)."""
        # Access scraper multiple times
        scraper1 = agent_manager.scraper
        scraper2 = agent_manager.scraper

        # Should be the same instance
        assert scraper1 is scraper2
        assert agent_manager._scraper is scraper1

    def test_get_agent_status_all_not_initialized(self, agent_manager):
        """Test get_agent_status when no agents are initialized."""
        status = agent_manager.get_agent_status()

        assert status == {
            "scraper": "not_initialized",
            "story_clustering": "not_initialized",
            "summarizer": "not_initialized",
            "sentiment_analyzer": "not_initialized",
            "report_generator": "not_initialized",
        }

    def test_get_agent_status_all_initialized(self, agent_manager):
        """Test get_agent_status when all agents are initialized."""
        # Initialize all agents
        agent_manager.scraper
        agent_manager.story_clustering
        agent_manager.summarizer
        agent_manager.sentiment_analyzer
        agent_manager.report_generator

        status = agent_manager.get_agent_status()

        assert status == {
            "scraper": "initialized",
            "story_clustering": "initialized",
            "summarizer": "initialized",
            "sentiment_analyzer": "initialized",
            "report_generator": "initialized",
        }

    def test_get_agent_status_partial_initialization(self, agent_manager):
        """Test get_agent_status when only some agents are initialized."""
        # Initialize only scraper and sentiment_analyzer
        agent_manager.scraper
        agent_manager.sentiment_analyzer

        status = agent_manager.get_agent_status()

        assert status == {
            "scraper": "initialized",
            "story_clustering": "not_initialized",
            "summarizer": "not_initialized",
            "sentiment_analyzer": "initialized",
            "report_generator": "not_initialized",
        }


class TestDatabaseManager:
    """Tests for DatabaseManager class."""

    @pytest.fixture
    def news_config(self, db):
        """Create a NewsConfig for testing."""
        from utilities.django_models import NewsConfig

        config, _ = NewsConfig.objects.get_or_create(
            key="test_managers",
            defaults={"display_name": "Test Managers Config"},
        )
        return config

    @pytest.fixture
    def database_manager(self, news_config):
        """Create a DatabaseManager for testing."""
        return DatabaseManager(news_config)

    @pytest.mark.django_db
    def test_news_config_property(self, database_manager, news_config):
        """Test that news_config property returns the correct config."""
        assert database_manager.news_config is news_config

    @pytest.mark.django_db
    def test_load_articles_loads_from_database(self, database_manager):
        """Test that load_articles loads articles from database."""
        from utilities.django_models import Article as DjangoArticle

        # Create test articles with different scraped dates
        now = datetime.now(TZ)
        yesterday = now - timedelta(days=1)
        two_days_ago = now - timedelta(days=2)
        three_days_ago = now - timedelta(days=3)

        # Create articles - some within range, some outside
        # Note: scraped_date has auto_now_add=True, so we need to use update() to set it
        # Use slightly more than 1 day ago to ensure it's well within the range
        one_and_half_days_ago = now - timedelta(days=1, hours=12)
        # Use slightly less than 2 days ago to ensure it's included with >= comparison
        almost_two_days_ago = now - timedelta(days=2, hours=-1)
        # Use slightly more than 2 days ago to ensure it's excluded
        more_than_two_days_ago = now - timedelta(days=2, hours=1)

        article1 = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Recent Article 1",
            content="Content 1",
            source="Source 1",
            url="https://example.com/article1",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=article1.id).update(
            scraped_date=one_and_half_days_ago
        )  # Within 2 days

        article2 = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Recent Article 2",
            content="Content 2",
            source="Source 2",
            url="https://example.com/article2",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=article2.id).update(
            scraped_date=almost_two_days_ago
        )  # Just within 2 days

        article3 = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Old Article",
            content="Content 3",
            source="Source 3",
            url="https://example.com/article3",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=article3.id).update(
            scraped_date=more_than_two_days_ago
        )  # Outside 2 days

        # Load articles from last 2 days
        articles = database_manager.load_articles(days_back=2)

        # Should return 2 articles (within 2 days)
        # article1: yesterday (within 2 days) - should be included
        # article2: exactly 2 days ago (with >= comparison, should be included)
        # article3: 3 days ago - should NOT be included
        assert len(articles) == 2

        # Verify articles are converted correctly
        assert all(isinstance(article, Article) for article in articles)

        # Verify article fields are mapped correctly
        # Find article1 by URL (it should definitely be in the results)
        article1_loaded = next(
            (a for a in articles if a.url == "https://example.com/article1"),
            None,
        )
        assert article1_loaded is not None, "Article 1 should be loaded"
        assert article1_loaded.title == "Recent Article 1"
        assert article1_loaded.content == "Content 1"
        assert article1_loaded.source == "Source 1"
        assert article1_loaded.summary == ""  # No summary in test data

    @pytest.mark.django_db
    def test_load_articles_filters_by_config(self, database_manager, db):
        """Test that load_articles only returns articles for the manager's config."""
        from utilities.django_models import Article as DjangoArticle
        from utilities.django_models import NewsConfig

        # Create another config and article
        other_config, _ = NewsConfig.objects.get_or_create(
            key="other_config",
            defaults={"display_name": "Other Config"},
        )

        now = datetime.now(TZ)
        other_article = DjangoArticle.objects.create(
            config=other_config,  # Different config
            title="Other Article",
            content="Content",
            source="Source",
            url="https://example.com/other",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=other_article.id).update(
            scraped_date=now
        )

        # Create article for our config
        our_article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Our Article",
            content="Content",
            source="Source",
            url="https://example.com/ours",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=our_article.id).update(
            scraped_date=now
        )

        # Load articles
        articles = database_manager.load_articles(days_back=7)

        # Should only return article for our config
        assert len(articles) == 1
        assert articles[0].url == "https://example.com/ours"

    @pytest.mark.django_db
    def test_load_articles_handles_empty_content(self, database_manager):
        """Test that load_articles handles articles with empty content."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)

        # Set scraped_date to a clear past time to ensure it's within the 7-day range
        # Use a time that's clearly in the past but well within 7 days
        past_time = now - timedelta(days=1)

        # Create article with empty string content
        # Note: The database has a NOT NULL constraint on content, so we can only test
        # empty strings, not None values
        article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Article with empty content",
            content="",
            source="Source",
            url="https://example.com/article1",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=article.id).update(
            scraped_date=past_time
        )

        # Verify article exists in database with correct scraped_date
        article_db = DjangoArticle.objects.get(id=article.id)

        # Check that scraped_date was set correctly (allowing for minor time drift)
        assert (
            abs((article_db.scraped_date - past_time).total_seconds()) < 1
        ), (
            f"Article scraped_date mismatch: expected ~{past_time}, got {article_db.scraped_date}"
        )

        articles = database_manager.load_articles(days_back=7)

        assert len(articles) == 1, (
            f"Expected 1 article, got {len(articles)}. "
            f"URLs: {[a.url for a in articles]}. "
            f"Expected URL: 'https://example.com/article1'"
        )

        # Verify article is loaded with empty content
        assert articles[0].url == "https://example.com/article1"
        assert articles[0].content == "", (
            f"Article should have empty content, "
            f"got: {repr(articles[0].content)} (type: {type(articles[0].content)})"
        )

    @pytest.mark.django_db
    def test_load_articles_handles_summary(self, database_manager):
        """Test that load_articles correctly loads article summaries."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)

        article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Article with summary",
            content="Content",
            summary="This is a summary",
            source="Source",
            url="https://example.com/article1",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=article.id).update(scraped_date=now)

        articles = database_manager.load_articles(days_back=7)

        assert len(articles) == 1
        assert articles[0].summary == "This is a summary"

    @pytest.mark.django_db
    def test_update_articles_with_analysis_updates_existing(
        self, database_manager
    ):
        """Test that update_articles_with_analysis updates existing articles."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)

        # Create an existing article in the database
        django_article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Test Article",
            content="Original content",
            source="Test Source",
            url="https://example.com/test",
            published_date=now,
            summary="",  # No summary yet
            sentiment_score=None,  # No sentiment yet
            sentiment_label="",
        )
        DjangoArticle.objects.filter(id=django_article.id).update(
            scraped_date=now
        )

        # Create Article object with analysis results
        sentiment = SentimentResult(
            article_url="https://example.com/test",
            source="Test Source",
            polarity=0.5,
            subjectivity=0.6,
            compound=0.7,
            label="positive",
        )

        article = Article(
            title="Test Article",
            content="Original content",
            source="Test Source",
            url="https://example.com/test",
            published_date=now,
            scraped_date=now,
            summary="Updated summary",
            sentiment=sentiment,
        )

        # Update the article
        updated_count = database_manager.update_articles_with_analysis(
            [article]
        )

        assert updated_count == 1

        # Refresh from database
        django_article.refresh_from_db()

        # Verify updates
        assert django_article.summary == "Updated summary"
        assert django_article.sentiment_score == 0.7
        assert django_article.sentiment_label == "positive"

    @pytest.mark.django_db
    def test_update_articles_with_analysis_partial_updates(
        self, database_manager
    ):
        """Test that update_articles_with_analysis handles partial updates."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)

        # Create article with existing summary
        django_article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Test Article",
            content="Content",
            source="Source",
            url="https://example.com/test",
            published_date=now,
            summary="Existing summary",
            sentiment_score=None,
            sentiment_label="",
        )
        DjangoArticle.objects.filter(id=django_article.id).update(
            scraped_date=now
        )

        # Update only sentiment (no summary in Article object)
        sentiment = SentimentResult(
            article_url="https://example.com/test",
            source="Source",
            polarity=0.3,
            subjectivity=0.4,
            compound=-0.5,
            label="negative",
        )

        article = Article(
            title="Test Article",
            content="Content",
            source="Source",
            url="https://example.com/test",
            published_date=now,
            scraped_date=now,
            summary=None,  # No summary update
            sentiment=sentiment,
        )

        updated_count = database_manager.update_articles_with_analysis(
            [article]
        )

        assert updated_count == 1

        django_article.refresh_from_db()

        # Summary should remain unchanged
        assert django_article.summary == "Existing summary"
        # Sentiment should be updated
        assert django_article.sentiment_score == -0.5
        assert django_article.sentiment_label == "negative"

    @pytest.mark.django_db
    def test_update_articles_with_analysis_no_sentiment(
        self, database_manager
    ):
        """Test update_articles_with_analysis when article has no sentiment."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)

        django_article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Test Article",
            content="Content",
            source="Source",
            url="https://example.com/test",
            published_date=now,
            summary="",
            sentiment_score=None,
            sentiment_label="",
        )
        DjangoArticle.objects.filter(id=django_article.id).update(
            scraped_date=now
        )

        # Article with summary but no sentiment
        article = Article(
            title="Test Article",
            content="Content",
            source="Source",
            url="https://example.com/test",
            published_date=now,
            scraped_date=now,
            summary="New summary",
            sentiment=None,  # No sentiment
        )

        updated_count = database_manager.update_articles_with_analysis(
            [article]
        )

        assert updated_count == 1

        django_article.refresh_from_db()

        # Summary should be updated
        assert django_article.summary == "New summary"
        # Sentiment should remain None/empty
        assert django_article.sentiment_score is None
        assert django_article.sentiment_label == ""

    @pytest.mark.django_db
    def test_update_articles_with_analysis_nonexistent_article(
        self, database_manager
    ):
        """Test update_articles_with_analysis when article doesn't exist."""
        now = datetime.now(TZ)

        # Create Article object for non-existent article
        article = Article(
            title="Nonexistent Article",
            content="Content",
            source="Source",
            url="https://example.com/nonexistent",
            published_date=now,
            scraped_date=now,
            summary="Summary",
        )

        updated_count = database_manager.update_articles_with_analysis(
            [article]
        )

        # Should return 0 (no articles updated)
        assert updated_count == 0

    @pytest.mark.django_db
    def test_update_articles_with_analysis_handles_exception(
        self, database_manager, monkeypatch
    ):
        """Test that update_articles_with_analysis handles exceptions gracefully."""
        from django.db import DatabaseError
        from utilities.django_models import Article as DjangoArticle

        # Mock DjangoArticle.objects.filter to raise an exception
        def mock_filter(*args, **kwargs):
            raise DatabaseError("Database error")

        monkeypatch.setattr(DjangoArticle.objects, "filter", mock_filter)

        article = Article(
            title="Test",
            content="Content",
            source="Source",
            url="https://example.com/test",
            published_date=datetime.now(TZ),
            scraped_date=datetime.now(TZ),
        )

        updated_count = database_manager.update_articles_with_analysis(
            [article]
        )

        assert updated_count == 0

    @pytest.mark.django_db
    def test_has_scraped_today_returns_true_when_articles_exist(
        self, database_manager
    ):
        """Test has_scraped_today returns True when articles exist today."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Create article from today
        article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Today's Article",
            content="Content",
            source="Source",
            url="https://example.com/today",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=article.id).update(
            scraped_date=today_start + timedelta(hours=10)
        )  # Today at 10 AM

        assert database_manager.has_scraped_today() is True

    @pytest.mark.django_db
    def test_has_scraped_today_returns_false_when_no_articles_today(
        self, database_manager
    ):
        """Test has_scraped_today returns False when no articles exist today."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)
        yesterday = now - timedelta(days=1)

        # Create article from yesterday
        article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Yesterday's Article",
            content="Content",
            source="Source",
            url="https://example.com/yesterday",
            published_date=yesterday,
        )
        DjangoArticle.objects.filter(id=article.id).update(
            scraped_date=yesterday
        )

        assert database_manager.has_scraped_today() is False

    @pytest.mark.django_db
    def test_has_scraped_today_filters_by_config(self, database_manager, db):
        """Test has_scraped_today only checks articles for the manager's config."""
        from utilities.django_models import Article as DjangoArticle
        from utilities.django_models import NewsConfig

        now = datetime.now(TZ)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Create another config with article from today
        other_config, _ = NewsConfig.objects.get_or_create(
            key="other_config",
            defaults={"display_name": "Other Config"},
        )

        other_article = DjangoArticle.objects.create(
            config=other_config,  # Different config
            title="Other Article",
            content="Content",
            source="Source",
            url="https://example.com/other",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=other_article.id).update(
            scraped_date=today_start + timedelta(hours=10)
        )

        # Should return False because no articles for our config
        assert database_manager.has_scraped_today() is False

        # Create article for our config
        our_article = DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Our Article",
            content="Content",
            source="Source",
            url="https://example.com/ours",
            published_date=now,
        )
        DjangoArticle.objects.filter(id=our_article.id).update(
            scraped_date=today_start + timedelta(hours=11)
        )

        # Now should return True
        assert database_manager.has_scraped_today() is True

    @pytest.mark.django_db
    def test_has_scraped_today_handles_exception(
        self, database_manager, monkeypatch
    ):
        """Test that has_scraped_today handles exceptions gracefully."""
        from django.db import DatabaseError
        from utilities.django_models import Article as DjangoArticle

        # Mock DjangoArticle.objects.filter to raise an exception
        def mock_filter(*args, **kwargs):
            raise DatabaseError("Database error")

        monkeypatch.setattr(DjangoArticle.objects, "filter", mock_filter)

        # Should return False on exception
        assert database_manager.has_scraped_today() is False
        assert database_manager.has_scraped_today() is False

    @pytest.mark.django_db
    def test_url_exists_with_content_returns_false_when_no_article(
        self, database_manager
    ):
        """Test url_exists_with_content returns False when no article exists."""
        assert database_manager.url_exists_with_content(
            "https://example.com/missing"
        ) is False

    @pytest.mark.django_db
    def test_url_exists_with_content_returns_false_when_article_has_no_content(
        self, database_manager
    ):
        """Test url_exists_with_content returns False when article exists but content is empty."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)
        DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="No content",
            content="",
            source="Source",
            url="https://example.com/empty",
            published_date=now,
        )
        DjangoArticle.objects.filter(
            config=database_manager.news_config,
            url="https://example.com/empty",
        ).update(scraped_date=now)
        assert database_manager.url_exists_with_content(
            "https://example.com/empty"
        ) is False

    @pytest.mark.django_db
    def test_url_exists_with_content_returns_true_when_article_has_content(
        self, database_manager
    ):
        """Test url_exists_with_content returns True when article exists with content."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)
        DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="With content",
            content="Some body text",
            source="Source",
            url="https://example.com/full",
            published_date=now,
        )
        DjangoArticle.objects.filter(
            config=database_manager.news_config,
            url="https://example.com/full",
        ).update(scraped_date=now)
        assert database_manager.url_exists_with_content(
            "https://example.com/full"
        ) is True

    @pytest.mark.django_db
    def test_save_articles_backfills_content_for_existing_empty(
        self, database_manager
    ):
        """Test save_articles updates existing rows that had no content."""
        from utilities.django_models import Article as DjangoArticle

        now = datetime.now(TZ)
        url = "https://example.com/backfill"
        DjangoArticle.objects.create(
            config=database_manager.news_config,
            title="Existing",
            content="",
            source="Source",
            url=url,
            published_date=now,
        )
        DjangoArticle.objects.filter(
            config=database_manager.news_config,
            url=url,
        ).update(scraped_date=now)

        new_scraped = now + timedelta(hours=1)
        articles = [
            Article(
                title="Existing",
                content="Fetched full content for backfill.",
                source="Source",
                url=url,
                published_date=now,
                scraped_date=new_scraped,
            ),
        ]
        saved = database_manager.save_articles(articles)
        assert saved == 0  # No new row created

        db_art = DjangoArticle.objects.get(
            config=database_manager.news_config,
            url=url,
        )
        assert db_art.content == "Fetched full content for backfill."
        assert db_art.scraped_date == new_scraped

    @pytest.mark.django_db
    def test_url_exists_in_any_config_returns_false_when_empty(
        self, database_manager
    ):
        """Test url_exists_in_any_config returns False when config_keys is empty."""
        assert database_manager.url_exists_in_any_config(
            "https://example.com/any", []
        ) is False

    @pytest.mark.django_db
    def test_url_exists_in_any_config_returns_true_when_url_in_config(
        self, database_manager, db
    ):
        """Test url_exists_in_any_config returns True when URL exists in one config."""
        from utilities.django_models import Article as DjangoArticle
        from utilities.django_models import NewsConfig

        now = datetime.now(TZ)
        other_config, _ = NewsConfig.objects.get_or_create(
            key="other_exclude_config",
            defaults={"display_name": "Other Exclude Config"},
        )
        DjangoArticle.objects.create(
            config=other_config,
            title="Other Article",
            content="Content",
            source="Source",
            url="https://example.com/shared",
            published_date=now,
        )
        DjangoArticle.objects.filter(
            config=other_config, url="https://example.com/shared"
        ).update(scraped_date=now)

        # Our database_manager is for test_managers config; check across configs
        assert database_manager.url_exists_in_any_config(
            "https://example.com/shared", [database_manager.news_config.key, "other_exclude_config"]
        ) is True

    @pytest.mark.django_db
    def test_url_exists_in_any_config_returns_false_when_url_not_in_any(
        self, database_manager, db
    ):
        """Test url_exists_in_any_config returns False when URL in no config."""
        assert database_manager.url_exists_in_any_config(
            "https://example.com/nowhere", [database_manager.news_config.key]
        ) is False

    @pytest.mark.django_db
    def test_url_exists_in_any_config_handles_exception(
        self, database_manager, monkeypatch
    ):
        """Test that url_exists_in_any_config handles exceptions gracefully."""
        from django.db import DatabaseError
        from utilities.django_models import Article as DjangoArticle

        def mock_filter(*args, **kwargs):
            raise DatabaseError("Database error")

        monkeypatch.setattr(DjangoArticle.objects, "filter", mock_filter)

        assert database_manager.url_exists_in_any_config(
            "https://example.com/any", ["some_config"]
        ) is False
