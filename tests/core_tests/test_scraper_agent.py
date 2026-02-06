"""Tests for News Scraper Agent"""

from datetime import datetime, timedelta
from typing import cast
from unittest.mock import Mock, patch

import requests
from feedparser.util import FeedParserDict

from newsbot.agents.scraper_agent import NewsScraperAgent
from newsbot.constants import TZ
from newsbot.models import Article, NewsSource
from utilities import models as config_models


class TestNewsScraperAgent:
    """Test cases for NewsScraperAgent"""

    def test_init(self, sample_config):
        """Test agent initialization"""
        agent = NewsScraperAgent(sample_config)

        assert agent.config == sample_config
        assert agent.sources == [
            {"name": source.name, "rss_url": source.rss_url, "type": source.type}
            for source in sample_config.news_sources
        ]
        assert agent.country == "US"
        assert agent.language == "en"
        assert agent.lookback_days == 7

    def test_init_with_defaults(self):
        """Test agent initialization with minimal config"""
        config = config_models.ConfigModel(news_sources=[])
        agent = NewsScraperAgent(config)

        assert agent.country == "US"
        assert agent.language == "en"
        assert agent.lookback_days == 7

    def test_init_with_exclude_url_check(self, sample_config):
        """Test agent stores exclude_url_check when provided."""
        exclude_check = Mock(return_value=False)
        agent = NewsScraperAgent(
            sample_config, exclude_url_check=exclude_check
        )
        assert agent.exclude_url_check is exclude_check

    def test_init_without_exclude_url_check(self, sample_config):
        """Test agent has exclude_url_check None when not provided."""
        agent = NewsScraperAgent(sample_config)
        assert agent.exclude_url_check is None

    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_scrape_rss_feed_success(
        self, mock_parse, sample_config, mock_rss_feed
    ):
        """Test successful RSS feed scraping"""
        mock_parse.return_value = mock_rss_feed

        agent = NewsScraperAgent(sample_config)
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

        assert len(articles) == 2
        assert all(isinstance(article, Article) for article in articles)
        assert articles[0].title == "Test Article 1"
        assert articles[0].source == "Test News"
        assert articles[0].url == "https://test.com/article1"
        mock_parse.assert_called_once_with(
            source["rss_url"], agent=NewsScraperAgent.REQUEST_AGENT
        )

    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_scrape_rss_feed_filters_old_articles(
        self, mock_parse, sample_config
    ):
        """Test that old articles are filtered out"""
        import time

        now = datetime.now()
        old_date = now - timedelta(days=30)
        recent_date = now - timedelta(days=1)

        class MockEntry:
            def __init__(self, title, link, summary, published_parsed):
                self.title = title
                self.link = link
                self.summary = summary
                self.published_parsed = published_parsed

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockFeed:
            def __init__(self):
                self.entries = [
                    MockEntry(
                        "Old Article",
                        "https://test.com/old",
                        "Old article",
                        time.struct_time(old_date.timetuple()),
                    ),
                    MockEntry(
                        "Recent Article",
                        "https://test.com/recent",
                        "Recent article",
                        time.struct_time(recent_date.timetuple()),
                    ),
                ]

        mock_parse.return_value = MockFeed()

        agent = NewsScraperAgent(sample_config)
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
        assert articles[0].title == "Recent Article"

    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_scrape_rss_feed_handles_missing_dates(
        self, mock_parse, sample_config
    ):
        """Test handling of articles without publication dates"""

        class MockEntry:
            def __init__(self, title, link, summary):
                self.title = title
                self.link = link
                self.summary = summary

            def get(self, key, default=None):
                return getattr(self, key, default)

        class MockFeed:
            def __init__(self):
                self.entries = [
                    MockEntry(
                        "No Date Article",
                        "https://test.com/nodate",
                        "Article without date",
                    ),
                ]

        mock_parse.return_value = MockFeed()

        agent = NewsScraperAgent(sample_config)
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
        assert articles[0].published_date is not None

    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_scrape_rss_feed_error_handling(self, mock_parse, sample_config):
        """Test error handling during RSS feed parsing"""
        from urllib.error import URLError
        mock_parse.side_effect = URLError("Feed parsing error")

        agent = NewsScraperAgent(sample_config)
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

        assert articles == []

    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    @patch.object(NewsScraperAgent, "_fetch_full_content")
    def test_scrape_rss_feed_prefers_content_field(
        self,
        mock_fetch,
        mock_parse,
        sample_config,
    ):
        """Content field should be used when available."""
        mock_fetch.return_value = None  # Fetch fails, should use content field

        class Entry:
            def __init__(self):
                self.title = "Entry"
                self.link = "http://example.com"
                self.summary = "summary"
                self.content = [
                    type("C", (), {"value": "<p>full content</p>"})()
                ]
                self.published_parsed = None

            def get(self, key, default=None):
                return getattr(self, key, default)

        class Feed:
            entries = [Entry()]

        mock_parse.return_value = Feed()

        agent = NewsScraperAgent(sample_config)
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
        assert "full content" in articles[0].content

    @patch.object(NewsScraperAgent, "_scrape_source")
    def test_scrape_all_sources(
        self, mock_scrape_source, sample_config, sample_article
    ):
        """Test scraping multiple sources"""
        mock_scrape_source.return_value = [sample_article]

        agent = NewsScraperAgent(sample_config)
        articles = agent.scrape_all_sources()

        assert len(articles) == len(sample_config.news_sources)
        assert mock_scrape_source.call_count == len(
            sample_config.news_sources
        )

    @patch.object(NewsScraperAgent, "_scrape_source")
    def test_scrape_all_sources_handles_errors(
        self, mock_scrape_source, sample_config
    ):
        """Test that scraping continues even if one source fails"""
        import requests
        mock_scrape_source.side_effect = [
            requests.exceptions.RequestException("Source 1 failed"),
            [],
        ]

        # Add another source
        from utilities.models import NewsSourceModel
        config = sample_config.model_copy(
            update={
                "news_sources": [
                    NewsSourceModel(
                        name="Source1",
                        rss_url="http://source1.com/feed",
                        type="rss",
                    ),
                    NewsSourceModel(
                        name="Source2",
                        rss_url="http://source2.com/feed",
                        type="rss",
                    ),
                ]
            }
        )

        agent = NewsScraperAgent(config)
        articles = agent.scrape_all_sources()

        # Should return empty list but not crash
        assert isinstance(articles, list)

    def test_scrape_source_unknown_type(self, sample_config):
        """Test handling of unknown source types"""
        from newsbot.models import NewsSource

        source: NewsSource = cast(
            NewsSource,
            {
                "name": "Unknown",
                "type": "unknown",
                "url": "http://test.com",
            },
        )

        agent = NewsScraperAgent(sample_config)
        articles = agent._scrape_source(source)

        assert articles == []

    def test_setup_session_headers(self, sample_config):
        """Test that session headers are set correctly"""
        agent = NewsScraperAgent(sample_config)

        assert agent.session.headers["User-Agent"] == NewsScraperAgent.REQUEST_AGENT
        assert "Accept" in agent.session.headers
        assert "Accept-Language" in agent.session.headers
        assert "DNT" in agent.session.headers

    def test_is_valid_html_valid(self, sample_config):
        """Test HTML validation with valid HTML"""
        agent = NewsScraperAgent(sample_config)

        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        valid_html = "<html><body><p>Valid content here with enough text to pass the minimum length requirement</p></body></html>"
        assert agent._is_valid_html(valid_html) is True

        valid_html2 = "<div><article>Content with enough text to pass validation requirements for minimum length</article></div>"
        assert agent._is_valid_html(valid_html2) is True

        valid_html3 = "<main><p>Some text with enough content to pass the minimum length validation check and meet the requirement</p></main>"
        assert agent._is_valid_html(valid_html3) is True

    def test_is_valid_html_invalid(self, sample_config):
        """Test HTML validation with invalid HTML"""
        agent = NewsScraperAgent(sample_config)

        # Too short
        assert agent._is_valid_html("") is False
        assert agent._is_valid_html("x" * 50) is False

        # Error indicators
        assert agent._is_valid_html("Rate limit exceeded") is False
        assert agent._is_valid_html("Too many requests") is False
        assert agent._is_valid_html("Access denied") is False
        assert agent._is_valid_html("403 Forbidden") is False
        assert agent._is_valid_html("429 Too Many") is False

        # No HTML structure
        assert agent._is_valid_html("Just plain text" * 10) is False

    @patch("newsbot.agents.scraper_agent.trafilatura.extract")
    def test_try_trafilatura_extract_success(self, mock_extract, sample_config):
        """Test successful trafilatura extraction"""
        agent = NewsScraperAgent(sample_config)

        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        html = "<html><body><p>Article content here with enough text to pass the minimum length requirement for HTML validation</p></body></html>"
        mock_extract.return_value = "Article content here" * 10  # > MIN_CONTENT_LENGTH

        result = agent._try_trafilatura_extract(html)

        assert result is not None
        assert "Article content here" in result
        mock_extract.assert_called_once()

    @patch("newsbot.agents.scraper_agent.trafilatura.extract")
    def test_try_trafilatura_extract_invalid_html(self, mock_extract, sample_config):
        """Test trafilatura extraction with invalid HTML"""
        agent = NewsScraperAgent(sample_config)

        invalid_html = "Rate limit exceeded"
        result = agent._try_trafilatura_extract(invalid_html)

        assert result is None
        mock_extract.assert_not_called()

    @patch("newsbot.agents.scraper_agent.trafilatura.extract")
    def test_try_trafilatura_extract_short_content(self, mock_extract, sample_config):
        """Test trafilatura extraction with content too short"""
        agent = NewsScraperAgent(sample_config)

        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars) to pass validation
        html = "<html><body><p>Short content but HTML is long enough to pass validation check</p></body></html>"
        mock_extract.return_value = "Short"  # < MIN_CONTENT_LENGTH

        result = agent._try_trafilatura_extract(html)

        assert result is None

    @patch("newsbot.agents.scraper_agent.trafilatura.extract")
    def test_try_trafilatura_extract_exception(self, mock_extract, sample_config):
        """Test trafilatura extraction with exception"""
        agent = NewsScraperAgent(sample_config)

        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        html = "<html><body><p>Content with enough text to pass the minimum length requirement for HTML validation</p></body></html>"
        mock_extract.side_effect = Exception("Extraction failed")

        result = agent._try_trafilatura_extract(html)

        assert result is None

    def test_try_beautifulsoup_extract_article_tag(self, sample_config):
        """Test BeautifulSoup extraction with article tag"""
        agent = NewsScraperAgent(sample_config)

        html = """
        <html>
            <body>
                <article>
                    <p>This is a long article content that should be extracted properly.</p>
                    <p>More content here to make it long enough.</p>
                    <p>Even more content to ensure we exceed the minimum length.</p>
                </article>
            </body>
        </html>
        """

        result = agent._try_beautifulsoup_extract(html)

        assert result is not None
        assert "article content" in result.lower()

    def test_try_beautifulsoup_extract_main_tag(self, sample_config):
        """Test BeautifulSoup extraction with main tag"""
        agent = NewsScraperAgent(sample_config)

        html = """
        <html>
            <body>
                <main>
                    <p>Main content here with enough text to pass validation.</p>
                    <p>Additional paragraphs to ensure minimum length.</p>
                </main>
            </body>
        </html>
        """

        result = agent._try_beautifulsoup_extract(html)

        assert result is not None
        assert "main content" in result.lower()

    def test_try_beautifulsoup_extract_paragraphs_fallback(self, sample_config):
        """Test BeautifulSoup extraction fallback to paragraphs"""
        agent = NewsScraperAgent(sample_config)

        html = """
        <html>
            <body>
                <div>
                    <p>First paragraph with substantial content.</p>
                    <p>Second paragraph with more content.</p>
                    <p>Third paragraph to ensure we have enough text.</p>
                    <p>Fourth paragraph for good measure.</p>
                </div>
            </body>
        </html>
        """

        result = agent._try_beautifulsoup_extract(html)

        assert result is not None
        assert "First paragraph" in result

    def test_try_beautifulsoup_extract_short_content(self, sample_config):
        """Test BeautifulSoup extraction with content too short"""
        agent = NewsScraperAgent(sample_config)

        html = "<html><body><p>Short</p></body></html>"

        result = agent._try_beautifulsoup_extract(html)

        assert result is None

    def test_try_beautifulsoup_extract_exception(self, sample_config):
        """Test BeautifulSoup extraction with exception"""
        agent = NewsScraperAgent(sample_config)

        # Invalid HTML that might cause issues
        html = "<html><body><p>Content</p></body></html>"

        # Mock BeautifulSoup to raise exception
        with patch("newsbot.agents.scraper_agent.BeautifulSoup") as mock_bs:
            mock_bs.side_effect = Exception("Parsing failed")

            result = agent._try_beautifulsoup_extract(html)

            assert result is None

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_try_beautifulsoup_extract")
    @patch.object(NewsScraperAgent, "_try_trafilatura_extract")
    def test_fetch_full_content_success_with_trafilatura(
        self,
        mock_trafilatura,
        mock_beautifulsoup,
        mock_sleep,
        sample_config,
    ):
        """Test successful content fetch using trafilatura"""
        agent = NewsScraperAgent(sample_config)

        mock_response = Mock()
        mock_response.status_code = 200
        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        mock_response.text = "<html><body><p>Content with enough text to pass the minimum length requirement for HTML validation</p></body></html>"
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()

        mock_get = Mock(return_value=mock_response)
        with patch.object(agent.session, "get", mock_get):
            mock_trafilatura.return_value = "Extracted content" * 10

            result = agent._fetch_full_content("http://test.com/article")

            assert result == "Extracted content" * 10
            mock_trafilatura.assert_called_once()
            mock_beautifulsoup.assert_not_called()

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_try_beautifulsoup_extract")
    @patch.object(NewsScraperAgent, "_try_trafilatura_extract")
    def test_fetch_full_content_fallback_to_beautifulsoup(
        self,
        mock_trafilatura,
        mock_beautifulsoup,
        mock_sleep,
        sample_config,
    ):
        """Test fallback to BeautifulSoup when trafilatura fails"""
        agent = NewsScraperAgent(sample_config)

        mock_response = Mock()
        mock_response.status_code = 200
        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        mock_response.text = "<html><body><article><p>Content with enough text to pass the minimum length requirement for HTML validation</p></article></body></html>"
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()

        mock_get = Mock(return_value=mock_response)
        with patch.object(agent.session, "get", mock_get):
            mock_trafilatura.return_value = None
            mock_beautifulsoup.return_value = "BeautifulSoup content" * 10

            result = agent._fetch_full_content("http://test.com/article")

            assert result == "BeautifulSoup content" * 10
            mock_trafilatura.assert_called_once()
            mock_beautifulsoup.assert_called_once()

    @patch("tenacity.nap.time.sleep")
    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_try_trafilatura_extract")
    def test_fetch_full_content_rate_limiting_with_retry(
        self, mock_trafilatura, mock_sleep, mock_tenacity_sleep, sample_config
    ):
        """Test rate limiting triggers retry and eventually succeeds."""
        agent = NewsScraperAgent(sample_config)

        # First response: rate limited
        rate_limited_response = Mock()
        rate_limited_response.status_code = 429
        rate_limited_response.headers = {}

        # Second response: success
        success_response = Mock()
        success_response.status_code = 200
        # HTML must be >= MIN_RESPONSE_LENGTH (100 chars)
        success_response.text = "<html><body><p>Content with enough text to pass the minimum length requirement for HTML validation</p></body></html>"
        success_response.headers = {}
        success_response.raise_for_status = Mock()

        mock_get = Mock(side_effect=[rate_limited_response, success_response])
        with patch.object(agent.session, "get", mock_get):
            mock_trafilatura.return_value = "Content" * 10

            result = agent._fetch_full_content("http://test.com/article")

            assert result == "Content" * 10
            # Tenacity retries after rate limit, so 2 calls total
            assert mock_get.call_count == 2

    @patch("tenacity.nap.time.sleep")
    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_fetch_full_content_rate_limiting_max_retries(
        self, mock_sleep, mock_tenacity_sleep, sample_config
    ):
        """Test rate limiting exhausts retries and returns None."""
        agent = NewsScraperAgent(sample_config)

        # All responses: rate limited
        rate_limited_response = Mock()
        rate_limited_response.status_code = 429
        rate_limited_response.headers = {}

        mock_get = Mock(return_value=rate_limited_response)
        with patch.object(agent.session, "get", mock_get):
            result = agent._fetch_full_content("http://test.com/article")

            assert result is None
            # Should retry max_retries times (3)
            assert mock_get.call_count == 3

    @patch("tenacity.nap.time.sleep")
    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_fetch_full_content_timeout(
        self, mock_sleep, mock_tenacity_sleep, sample_config
    ):
        """Test timeout handling exhausts retries."""
        import requests

        agent = NewsScraperAgent(sample_config)

        mock_get = Mock(side_effect=requests.exceptions.Timeout("Timeout"))
        with patch.object(agent.session, "get", mock_get):
            result = agent._fetch_full_content("http://test.com/article")

            assert result is None
            # Tenacity retries on timeout
            assert mock_get.call_count == 3

    @patch("tenacity.nap.time.sleep")
    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_fetch_full_content_http_error(
        self, mock_sleep, mock_tenacity_sleep, sample_config
    ):
        """Test HTTP error handling (non-retryable errors fail immediately)."""
        import requests

        agent = NewsScraperAgent(sample_config)

        error_response = Mock()
        error_response.status_code = 404
        http_error = requests.exceptions.HTTPError("Not Found")
        http_error.response = error_response

        mock_get = Mock(side_effect=http_error)
        with patch.object(agent.session, "get", mock_get):
            result = agent._fetch_full_content("http://test.com/article")

            assert result is None
            # HTTPError (non-429) is not retried
            assert mock_get.call_count == 1

    @patch("tenacity.nap.time.sleep")
    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_try_trafilatura_extract")
    def test_fetch_full_content_timeout_then_success(
        self, mock_trafilatura, mock_sleep, mock_tenacity_sleep, sample_config
    ):
        """Test recovery from transient timeout on retry."""
        import requests

        agent = NewsScraperAgent(sample_config)

        # First: timeout, Second: success
        success_response = Mock()
        success_response.status_code = 200
        success_response.text = (
            "<html><body><p>" + "x" * 100 + "</p></body></html>"
        )
        success_response.headers = {}
        success_response.raise_for_status = Mock()

        mock_get = Mock(
            side_effect=[
                requests.exceptions.Timeout("Timeout"),
                success_response,
            ]
        )
        with patch.object(agent.session, "get", mock_get):
            mock_trafilatura.return_value = "Recovered content"

            result = agent._fetch_full_content("http://test.com/article")

            assert result == "Recovered content"
            # Retried once after timeout
            assert mock_get.call_count == 2

    @patch("tenacity.nap.time.sleep")
    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_fetch_full_content_too_many_redirects(
        self, mock_sleep, mock_tenacity_sleep, sample_config
    ):
        """Test TooManyRedirects is not retried (fails immediately)."""
        import requests

        agent = NewsScraperAgent(sample_config)

        mock_get = Mock(
            side_effect=requests.exceptions.TooManyRedirects(
                "Too many redirects"
            )
        )
        with patch.object(agent.session, "get", mock_get):
            result = agent._fetch_full_content("http://test.com/article")

            assert result is None
            # Should NOT retry - TooManyRedirects is not in retry list
            assert mock_get.call_count == 1

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_try_trafilatura_extract")
    def test_fetch_full_content_invalid_html(
        self, mock_trafilatura, mock_sleep, sample_config
    ):
        """Test handling of invalid HTML response"""
        agent = NewsScraperAgent(sample_config)

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {}
        mock_response.raise_for_status = Mock()

        mock_get = Mock(return_value=mock_response)
        with patch.object(agent.session, "get", mock_get):
            result = agent._fetch_full_content("http://test.com/article")

            assert result is None
            mock_trafilatura.assert_not_called()

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_fetch_full_content")
    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_process_rss_entry_fetch_fails_with_description(
        self, mock_parse, mock_fetch, mock_sleep, sample_config
    ):
        """Test that entry with description but failed fetch still creates article"""
        import time

        agent = NewsScraperAgent(sample_config)

        recent_date = datetime.now() - timedelta(days=1)

        class MockEntry:
            def __init__(self):
                self.title = "Test Article"
                self.link = "http://test.com/article"
                self.summary = "Article summary with enough content"
                self.published_parsed = time.struct_time(recent_date.timetuple())

            def get(self, key, default=None):
                return getattr(self, key, default)

        mock_feed = Mock()
        mock_feed.entries = [MockEntry()]
        mock_parse.return_value = mock_feed

        mock_fetch.return_value = None  # Fetch fails

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
        assert articles[0].content == "Article summary with enough content"

    @patch("newsbot.agents.scraper_agent.time.sleep")
    @patch.object(NewsScraperAgent, "_fetch_full_content")
    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    def test_process_rss_entry_fetch_fails_no_content(
        self, mock_parse, mock_fetch, mock_sleep, sample_config
    ):
        """Test that entry without description and failed fetch is skipped"""
        import time

        agent = NewsScraperAgent(sample_config)

        recent_date = datetime.now() - timedelta(days=1)

        class MockEntry:
            def __init__(self):
                self.title = "Test Article"
                self.link = "http://test.com/article"
                # No summary or description
                self.published_parsed = time.struct_time(recent_date.timetuple())

            def get(self, key, default=None):
                return getattr(self, key, default)

        mock_feed = Mock()
        mock_feed.entries = [MockEntry()]
        mock_parse.return_value = mock_feed

        mock_fetch.return_value = None  # Fetch fails

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

        assert len(articles) == 0

    def test_process_rss_entry_excludes_when_exclude_url_check_returns_true(
        self, sample_config
    ):
        """Test that _process_rss_entry returns None when exclude_url_check(link) is True."""
        import time

        exclude_check = Mock(return_value=True)
        agent = NewsScraperAgent(
            sample_config, exclude_url_check=exclude_check
        )
        cutoff_date = datetime.now(TZ) - timedelta(days=7)

        class MockEntry:
            def __init__(self):
                self.title = "Excluded Article"
                self.link = "https://example.com/excluded"
                self.summary = "Summary"
                self.published_parsed = time.struct_time(
                    (datetime.now() - timedelta(days=1)).timetuple()
                )

            def get(self, key, default=None):
                return getattr(self, key, default)

        entry = MockEntry()
        result = agent._process_rss_entry(
            cast(FeedParserDict, entry), "Test Source", cutoff_date
        )

        assert result is None
        exclude_check.assert_called_once_with("https://example.com/excluded")

    def test_process_rss_entry_keeps_article_when_exclude_url_check_returns_false(
        self, sample_config
    ):
        """Test that _process_rss_entry returns article when exclude_url_check returns False."""
        import time

        exclude_check = Mock(return_value=False)
        agent = NewsScraperAgent(
            sample_config, exclude_url_check=exclude_check
        )
        cutoff_date = datetime.now(TZ) - timedelta(days=7)

        class MockEntry:
            def __init__(self):
                self.title = "Kept Article"
                self.link = "https://example.com/kept"
                self.summary = "Summary with content"
                self.published_parsed = time.struct_time(
                    (datetime.now() - timedelta(days=1)).timetuple()
                )

            def get(self, key, default=None):
                return getattr(self, key, default)

        entry = MockEntry()
        with patch.object(agent, "_fetch_full_content", return_value=None):
            result = agent._process_rss_entry(
                cast(FeedParserDict, entry), "Test Source", cutoff_date
            )

        assert result is not None
        assert result.title == "Kept Article"
        assert result.url == "https://example.com/kept"
        exclude_check.assert_called_once_with("https://example.com/kept")

    def test_parse_entry_date_published_parsed(self, sample_config):
        """Test parsing date from published_parsed"""
        import time
        from feedparser.util import FeedParserDict

        agent = NewsScraperAgent(sample_config)

        test_date = datetime(2024, 1, 15, 12, 0, 0)
        entry = FeedParserDict()
        entry.published_parsed = time.struct_time(test_date.timetuple())

        result = agent._parse_entry_date(entry)

        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_parse_entry_date_updated_parsed(self, sample_config):
        """Test parsing date from updated_parsed when published_parsed missing"""
        import time
        from feedparser.util import FeedParserDict

        agent = NewsScraperAgent(sample_config)

        test_date = datetime(2024, 2, 20, 10, 0, 0)
        entry = FeedParserDict()
        entry.updated_parsed = time.struct_time(test_date.timetuple())

        result = agent._parse_entry_date(entry)

        assert result is not None
        assert result.year == 2024
        assert result.month == 2
        assert result.day == 20

    def test_parse_entry_date_none(self, sample_config):
        """Test parsing date when both published and updated are missing"""
        from feedparser.util import FeedParserDict

        agent = NewsScraperAgent(sample_config)

        entry = FeedParserDict()

        result = agent._parse_entry_date(entry)

        assert result is None

    def test_extract_content_summary(self, sample_config):
        """Test content extraction from summary"""
        from feedparser.util import FeedParserDict

        agent = NewsScraperAgent(sample_config)

        entry = FeedParserDict()
        entry.summary = "Article summary text"

        description, content = agent._extract_content(entry)

        assert description == "Article summary text"
        assert content == "Article summary text"

    def test_extract_content_description(self, sample_config):
        """Test content extraction from description when summary missing"""
        from feedparser.util import FeedParserDict

        agent = NewsScraperAgent(sample_config)

        entry = FeedParserDict()
        entry.description = "Article description text"

        description, content = agent._extract_content(entry)

        assert description == "Article description text"
        assert content == "Article description text"

    def test_extract_content_prefers_content_field(self, sample_config):
        """Test that content field is preferred over summary"""
        from feedparser.util import FeedParserDict

        agent = NewsScraperAgent(sample_config)

        entry = FeedParserDict()
        entry.summary = "Summary text"
        entry.content = [type("C", (), {"value": "Full content text"})()]

        description, content = agent._extract_content(entry)

        assert description == "Summary text"
        assert content == "Full content text"
