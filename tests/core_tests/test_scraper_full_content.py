from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from newsbot.agents.scraper_agent import NewsScraperAgent
from newsbot.constants import TZ
from newsbot.models import Article, NewsSource
from typing import cast


class TestScraperFullContent:
    @pytest.fixture
    def sample_config(self):
        from utilities.models import ConfigModel, NewsSourceModel, ReportConfigModel
        return ConfigModel(
            name="test_config",
            news_sources=[
                NewsSourceModel(
                    name="Test Source",
                    rss_url="http://test.com/rss",
                    type="rss",
                )
            ],
            country="US",
            language="en",
            report=ReportConfigModel(lookback_days=7),
        )

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_try_trafilatura_extract")
    def test_fetch_full_content_success(
        self, mock_trafilatura, mock_sleep, sample_config
    ):
        """Test successful fetching of full content using trafilatura"""
        agent = NewsScraperAgent(sample_config)

        # Mock response
        mock_response = Mock()
        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        mock_response.text = "<html><body><p>Article content with enough text to pass the minimum length requirement for HTML validation.</p></body></html>"
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()

        mock_get = Mock(return_value=mock_response)
        with patch.object(agent.session, "get", mock_get):
            # Mock trafilatura extraction
            mock_trafilatura.return_value = "First paragraph. Second paragraph."

            content = agent._fetch_full_content("http://test.com/article")

            assert content == "First paragraph. Second paragraph."
            mock_get.assert_called_once()
            mock_trafilatura.assert_called_once()

    @patch("tenacity.nap.time.sleep")
    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_fetch_full_content_failure(
        self, mock_sleep, mock_tenacity_sleep, sample_config
    ):
        """Test failure handling when fetching content."""
        import requests

        agent = NewsScraperAgent(sample_config)

        # Mock connection error (retried by tenacity, then fails)
        mock_get = Mock(
            side_effect=requests.exceptions.ConnectionError("Connection error")
        )
        with patch.object(agent.session, "get", mock_get):
            content = agent._fetch_full_content("http://test.com/article")

            assert content is None
            # Tenacity retries connection errors
            assert mock_get.call_count == 3

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_try_beautifulsoup_extract")
    @patch.object(NewsScraperAgent, "_try_trafilatura_extract")
    def test_fetch_full_content_trafilatura_returns_none(
        self, mock_trafilatura, mock_beautifulsoup, mock_sleep, sample_config
    ):
        """Test handling when trafilatura cannot extract content"""
        agent = NewsScraperAgent(sample_config)

        # Mock response
        mock_response = Mock()
        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        mock_response.text = "<html><body><p>No article here but enough text to pass validation requirements</p></body></html>"
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()

        mock_get = Mock(return_value=mock_response)
        with patch.object(agent.session, "get", mock_get):
            # Mock trafilatura returning None (no content found)
            mock_trafilatura.return_value = None
            mock_beautifulsoup.return_value = None

            content = agent._fetch_full_content("http://test.com/article")

            assert content is None

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_fetch_full_content")
    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_scrape_rss_uses_full_content(
        self, mock_parse, mock_fetch, mock_sleep, sample_config
    ):
        """Test that scrape_rss_feed uses fetched content via trafilatura"""
        agent = NewsScraperAgent(sample_config)

        # Mock RSS feed using a class that supports .get()
        # Use a recent date (1 day ago) to ensure it's within the 7-day lookback period
        recent_date = datetime.now(TZ) - timedelta(days=1)
        recent_date_tuple = recent_date.timetuple()[:9]

        class MockEntry:
            def __init__(self):
                self.title = "Test Article"
                self.link = "http://test.com/article"
                self.summary = "Summary only"
                self.published_parsed = recent_date_tuple
                self.content = None  # Optional

            def get(self, key, default=None):
                return getattr(self, key, default)

        mock_feed = Mock()
        mock_feed.entries = [MockEntry()]
        mock_parse.return_value = mock_feed

        # Mock fetch to return full content
        mock_fetch.return_value = "Full content fetched."

        source_model = sample_config.news_sources[0]
        source: NewsSource = cast(
            NewsSource,
            {
                "name": source_model.name,
                "rss_url": source_model.rss_url,
                "type": source_model.type,
            },
        )
        articles = agent._scrape_rss_feed(source)

        assert len(articles) == 1
        assert articles[0].content == "Full content fetched."
        mock_fetch.assert_called_once_with("http://test.com/article")

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_fetch_full_content")
    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_scrape_rss_fallback_to_summary(
        self, mock_parse, mock_fetch, mock_sleep, sample_config
    ):
        """Test fallback to summary when fetch fails"""
        agent = NewsScraperAgent(sample_config)

        # Mock RSS feed
        # Use a recent date (1 day ago) to ensure it's within the 7-day lookback period
        recent_date = datetime.now(TZ) - timedelta(days=1)
        recent_date_tuple = recent_date.timetuple()[:9]

        class MockEntry:
            def __init__(self):
                self.title = "Test Article"
                self.link = "http://test.com/article"
                self.summary = "Summary only"
                self.published_parsed = recent_date_tuple
                self.content = None

            def get(self, key, default=None):
                return getattr(self, key, default)

        mock_feed = Mock()
        mock_feed.entries = [MockEntry()]
        mock_parse.return_value = mock_feed

        # Mock fetch failure
        mock_fetch.return_value = None

        source_model = sample_config.news_sources[0]
        source: NewsSource = cast(
            NewsSource,
            {
                "name": source_model.name,
                "rss_url": source_model.rss_url,
                "type": source_model.type,
            },
        )
        articles = agent._scrape_rss_feed(source)

        assert len(articles) == 1
        assert articles[0].content == "Summary only"
