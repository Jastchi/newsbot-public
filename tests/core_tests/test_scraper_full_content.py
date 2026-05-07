from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from newsbot.agents.scraper_agent import (
    MAX_LINKS_PER_LISTING,
    NewsScraperAgent,
)
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


def _make_html_response(text: str) -> Mock:
    """Build a Mock requests.Response with the given body text."""
    response = Mock()
    response.text = text
    response.status_code = 200
    response.headers = {}
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def html_config():
    """ConfigModel with a single HTML source."""
    from utilities.models import (
        ConfigModel,
        NewsSourceModel,
        ReportConfigModel,
    )
    return ConfigModel(
        name="test_html",
        news_sources=[
            NewsSourceModel(
                name="Test HTML Source",
                rss_url="https://news.example.com/section/world",
                type="html",
            ),
        ],
        country="US",
        language="en",
        report=ReportConfigModel(lookback_days=7),
    )


class TestDiscoverArticleLinks:
    def test_filters_off_domain_and_schemes(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = """
        <html><body>
            <a href="/news/article-1">A</a>
            <a href="https://news.example.com/news/article-2">B</a>
            <a href="https://other.example.com/news/article-3">C</a>
            <a href="mailto:foo@example.com">mail</a>
            <a href="tel:+123">call</a>
            <a href="javascript:void(0)">js</a>
            <a href="#section">anchor</a>
            <a href="">empty</a>
        </body></html>
        """
        links = agent._discover_article_links(
            html, "https://news.example.com/section/world",
        )
        assert links == [
            "https://news.example.com/news/article-1",
            "https://news.example.com/news/article-2",
        ]

    def test_dedupes_and_strips_fragments(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = """
        <html><body>
            <a href="/news/article-1">A</a>
            <a href="/news/article-1#top">A again</a>
            <a href="/news/article-1">A again again</a>
            <a href="/news/article-2">B</a>
        </body></html>
        """
        links = agent._discover_article_links(
            html, "https://news.example.com/section/world",
        )
        assert links == [
            "https://news.example.com/news/article-1",
            "https://news.example.com/news/article-2",
        ]

    def test_drops_listing_url_self_link(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = """
        <html><body>
            <a href="https://news.example.com/section/world">self</a>
            <a href="/news/article-1">A</a>
        </body></html>
        """
        links = agent._discover_article_links(
            html, "https://news.example.com/section/world",
        )
        assert links == ["https://news.example.com/news/article-1"]

    def test_caps_at_max_links(self, html_config):
        agent = NewsScraperAgent(html_config)
        anchors = "".join(
            f'<a href="/article-{i}">x</a>'
            for i in range(MAX_LINKS_PER_LISTING + 10)
        )
        html = f"<html><body>{anchors}</body></html>"
        links = agent._discover_article_links(
            html, "https://news.example.com/section/world",
        )
        assert len(links) == MAX_LINKS_PER_LISTING


class TestExtractArticleMetadata:
    def test_prefers_article_published_time(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = """
        <html><head>
            <meta property="og:title" content="Big News" />
            <meta property="article:published_time"
                  content="2026-04-01T10:00:00+00:00" />
            <title>Fallback title</title>
        </head><body>
            <time datetime="2020-01-01T00:00:00Z">old</time>
        </body></html>
        """
        title, pub_date = agent._extract_article_metadata(html)
        assert title == "Big News"
        assert pub_date is not None
        assert pub_date.year == 2026
        assert pub_date.month == 4

    def test_falls_back_to_time_tag(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = """
        <html><head><title>Plain title</title></head>
        <body><time datetime="2026-03-15T12:30:00Z">date</time></body>
        </html>
        """
        title, pub_date = agent._extract_article_metadata(html)
        assert title == "Plain title"
        assert pub_date is not None
        assert pub_date.year == 2026
        assert pub_date.month == 3

    def test_falls_back_to_jsonld(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = """
        <html><head>
            <title>JSON-LD title</title>
            <script type="application/ld+json">
            {"@type": "NewsArticle",
             "datePublished": "2026-02-20T08:00:00+00:00"}
            </script>
        </head><body><h1>Headline</h1></body></html>
        """
        title, pub_date = agent._extract_article_metadata(html)
        assert title == "JSON-LD title"
        assert pub_date is not None
        assert pub_date.year == 2026
        assert pub_date.month == 2

    def test_returns_none_date_when_missing(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = "<html><head><title>No date</title></head><body></body></html>"
        title, pub_date = agent._extract_article_metadata(html)
        assert title == "No date"
        assert pub_date is None

    def test_skips_malformed_jsonld(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = """
        <html><head><title>OK</title>
            <script type="application/ld+json">{not valid json</script>
        </head><body></body></html>
        """
        title, pub_date = agent._extract_article_metadata(html)
        assert title == "OK"
        assert pub_date is None

    def test_falls_back_to_h1(self, html_config):
        agent = NewsScraperAgent(html_config)
        html = "<html><body><h1>Just an H1</h1></body></html>"
        title, _ = agent._extract_article_metadata(html)
        assert title == "Just an H1"


class TestScrapeHtmlListing:
    def _listing_html(self, paths: list[str]) -> str:
        anchors = "".join(
            f'<a href="{p}">link</a>' for p in paths
        )
        return (
            "<html><body><main>"
            + anchors
            + "Some listing-page filler text to satisfy the minimum HTML "
            "validation length requirement of 100 characters.</main></body></html>"
        )

    def _article_html(self, title: str, body: str, date: str | None) -> str:
        date_meta = (
            f'<meta property="article:published_time" content="{date}" />'
            if date
            else ""
        )
        return (
            "<html><head>"
            f'<meta property="og:title" content="{title}" />'
            f"{date_meta}"
            f"<title>{title}</title>"
            "</head><body><article>"
            f"<p>{body}</p>"
            "</article></body></html>"
        )

    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_end_to_end_two_articles(self, _sleep, html_config):
        agent = NewsScraperAgent(html_config)
        listing_url = html_config.news_sources[0].rss_url
        listing = self._listing_html(["/a/1", "/a/2"])
        article_body = (
            "This is the body of an article. " * 20
        )
        article_a = self._article_html(
            "Article One", article_body,
            (datetime.now(TZ) - timedelta(days=1)).isoformat(),
        )
        article_b = self._article_html(
            "Article Two", article_body,
            (datetime.now(TZ) - timedelta(days=2)).isoformat(),
        )

        responses = {
            listing_url: _make_html_response(listing),
            "https://news.example.com/a/1": _make_html_response(article_a),
            "https://news.example.com/a/2": _make_html_response(article_b),
        }
        mock_get = Mock(side_effect=lambda url, **_: responses[url])
        with patch.object(agent.session, "get", mock_get):
            source = cast(
                NewsSource,
                {
                    "name": "Test HTML Source",
                    "rss_url": listing_url,
                    "type": "html",
                },
            )
            articles = agent._scrape_html_listing(source)

        assert len(articles) == 2
        titles = {a.title for a in articles}
        assert titles == {"Article One", "Article Two"}
        for article in articles:
            assert article.source == "Test HTML Source"
            assert article.url.startswith("https://news.example.com/a/")
            assert article.content
            assert article.published_date.year == 2026

    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_skips_articles_past_cutoff(self, _sleep, html_config):
        agent = NewsScraperAgent(html_config)
        listing_url = html_config.news_sources[0].rss_url
        old_date = (
            datetime.now(TZ) - timedelta(days=30)
        ).isoformat()
        listing = self._listing_html(["/a/old"])
        article = self._article_html(
            "Old Article", "body " * 30, old_date,
        )

        responses = {
            listing_url: _make_html_response(listing),
            "https://news.example.com/a/old": _make_html_response(article),
        }
        mock_get = Mock(side_effect=lambda url, **_: responses[url])
        with patch.object(agent.session, "get", mock_get):
            source = cast(
                NewsSource,
                {
                    "name": "Test HTML Source",
                    "rss_url": listing_url,
                    "type": "html",
                },
            )
            articles = agent._scrape_html_listing(source)
        assert articles == []

    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_invalid_listing_html_returns_empty(self, _sleep, html_config):
        agent = NewsScraperAgent(html_config)
        listing_url = html_config.news_sources[0].rss_url
        bad_response = _make_html_response("<small>nope</small>")

        mock_get = Mock(return_value=bad_response)
        with patch.object(agent.session, "get", mock_get):
            source = cast(
                NewsSource,
                {
                    "name": "Test HTML Source",
                    "rss_url": listing_url,
                    "type": "html",
                },
            )
            articles = agent._scrape_html_listing(source)

        assert articles == []
        # Only the listing fetch — no follow-up article fetches.
        assert mock_get.call_count == 1

    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_url_check_skips_already_in_db(self, _sleep, html_config):
        url_check = Mock(return_value=True)  # Pretend everything exists.
        agent = NewsScraperAgent(html_config, url_check=url_check)
        listing_url = html_config.news_sources[0].rss_url
        listing = self._listing_html(["/a/1", "/a/2"])

        mock_get = Mock(return_value=_make_html_response(listing))
        with patch.object(agent.session, "get", mock_get):
            source = cast(
                NewsSource,
                {
                    "name": "Test HTML Source",
                    "rss_url": listing_url,
                    "type": "html",
                },
            )
            articles = agent._scrape_html_listing(source)

        assert articles == []
        assert url_check.call_count == 2
        # Only the listing was fetched; article pages were skipped.
        assert mock_get.call_count == 1

    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_exclude_url_check_skips_links(self, _sleep, html_config):
        exclude_check = Mock(side_effect=lambda url: "1" in url)
        agent = NewsScraperAgent(
            html_config, exclude_url_check=exclude_check,
        )
        listing_url = html_config.news_sources[0].rss_url
        listing = self._listing_html(["/a/1", "/a/2"])
        article_b = self._article_html(
            "Article Two", "body " * 30,
            (datetime.now(TZ) - timedelta(days=1)).isoformat(),
        )

        responses = {
            listing_url: _make_html_response(listing),
            "https://news.example.com/a/2": _make_html_response(article_b),
        }
        mock_get = Mock(side_effect=lambda url, **_: responses[url])
        with patch.object(agent.session, "get", mock_get):
            source = cast(
                NewsSource,
                {
                    "name": "Test HTML Source",
                    "rss_url": listing_url,
                    "type": "html",
                },
            )
            articles = agent._scrape_html_listing(source)

        assert len(articles) == 1
        assert articles[0].url == "https://news.example.com/a/2"

    @patch("newsbot.agents.scraper_agent.time.sleep")
    def test_topic_filter_excludes_non_matches(self, _sleep):
        from utilities.models import (
            ConfigModel,
            NewsSourceModel,
            ReportConfigModel,
            TopicModel,
        )
        config = ConfigModel(
            name="test_html_topic",
            news_sources=[
                NewsSourceModel(
                    name="Test HTML Source",
                    rss_url="https://news.example.com/section",
                    type="html",
                ),
            ],
            topics=[
                TopicModel(
                    name="Sports",
                    keywords=["football", "soccer", "basketball"],
                    description="Sports coverage",
                    similarity_threshold=0.99,
                ),
            ],
            country="US",
            language="en",
            report=ReportConfigModel(lookback_days=7),
        )
        agent = NewsScraperAgent(config)
        listing_url = config.news_sources[0].rss_url
        listing = (
            "<html><body><main>"
            '<a href="/sports/match">Match</a>'
            '<a href="/finance/market">Market</a>'
            "Filler text to push past the minimum HTML validation length "
            "requirement of 100 characters for sure now ok."
            "</main></body></html>"
        )
        article_sports = (
            "<html><head><title>Football match recap</title>"
            '<meta property="og:title" content="Football match recap" />'
            '<meta property="article:published_time" content="'
            + (datetime.now(TZ) - timedelta(days=1)).isoformat()
            + '" />'
            "</head><body><article><p>"
            + ("football " * 30)
            + "</p></article></body></html>"
        )
        article_finance = (
            "<html><head><title>Stock market update</title>"
            '<meta property="og:title" content="Stock market update" />'
            '<meta property="article:published_time" content="'
            + (datetime.now(TZ) - timedelta(days=1)).isoformat()
            + '" />'
            "</head><body><article><p>"
            + ("equities " * 30)
            + "</p></article></body></html>"
        )

        responses = {
            listing_url: _make_html_response(listing),
            "https://news.example.com/sports/match":
                _make_html_response(article_sports),
            "https://news.example.com/finance/market":
                _make_html_response(article_finance),
        }
        mock_get = Mock(side_effect=lambda url, **_: responses[url])
        # Avoid the embedding pass - only keyword matches matter here
        # because the threshold is 0.99 (effectively unreachable for
        # the unrelated finance article).
        with patch.object(agent.session, "get", mock_get):
            source = cast(
                NewsSource,
                {
                    "name": "Test HTML Source",
                    "rss_url": listing_url,
                    "type": "html",
                },
            )
            articles = agent._scrape_html_listing(source)

        assert len(articles) == 1
        assert articles[0].url == "https://news.example.com/sports/match"


class TestScrapeSourceDispatch:
    @patch("newsbot.agents.scraper_agent.NewsScraperAgent._scrape_html_listing")
    def test_dispatches_html_source(self, mock_html, html_config):
        mock_html.return_value = []
        agent = NewsScraperAgent(html_config)
        source = cast(
            NewsSource,
            {
                "name": "Test HTML Source",
                "rss_url": html_config.news_sources[0].rss_url,
                "type": "html",
            },
        )
        agent._scrape_source(source)
        mock_html.assert_called_once_with(source)
