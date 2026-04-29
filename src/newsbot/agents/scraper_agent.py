"""
News Scraper Agent.

Responsible for collecting news articles from various sources
"""

import json
import logging
import socket
import time
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast
from urllib.error import URLError
from urllib.parse import urljoin, urlparse, urlunparse

import feedparser
import numpy as np
import requests
import trafilatura
from bs4 import BeautifulSoup
from feedparser.util import FeedParserDict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

if TYPE_CHECKING:
    from tenacity._utils import LoggerProtocol

from newsbot.constants import EMBEDDING_BATCH_SIZE, TZ
from newsbot.model_cache import get_sentence_transformer
from newsbot.models import Article, NewsSource
from newsbot.utils import clean_text
from utilities.models import ConfigModel, TopicModel

logger = logging.getLogger(__name__)

# Minimum content length to consider extraction successful
MIN_CONTENT_LENGTH = 100

# Minimum response length to consider valid HTML
MIN_RESPONSE_LENGTH = 100

# HTTP status code for rate limiting
HTTP_TOO_MANY_REQUESTS = 429

# Cap on candidate article links discovered per HTML listing page
MAX_LINKS_PER_LISTING = 50

# URL schemes to skip during link discovery
_SKIP_LINK_SCHEMES = ("mailto:", "tel:", "javascript:")


class RateLimitError(Exception):
    """Raised when server returns HTTP 429 Too Many Requests."""


class NewsScraperAgent:
    """Agent responsible for scraping news from various sources."""

    REQUEST_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(
        self,
        config: ConfigModel,
        url_check: Callable[[str], bool] | None = None,
        exclude_url_check: Callable[[str], bool] | None = None,
    ) -> None:
        """
        Initialize the News Scraper Agent.

        Args:
            config: Configuration dictionary
            url_check: Optional callback to decide whether to skip
                fetching full content. Should return True to skip (e.g.
                when the URL already exists with content), False to
                fetch (e.g. when missing or exists with no content).
            exclude_url_check: Optional callback to check if a URL
                should be excluded (e.g. exists in other configs).
                Should return True if the article should be excluded.
                Checked before topic filter and content extraction.

        """
        self.config = config
        self.config_name = config.name
        # Convert NewsSourceModel to NewsSource TypedDict for
        # compatibility
        self.sources: list[NewsSource] = [
            {
                "name": source.name,
                "rss_url": source.rss_url,
                "type": source.type,
            }
            for source in config.news_sources
        ]
        self.country = config.country
        self.language = config.language
        self.lookback_days = config.report.lookback_days
        self.session = requests.Session()
        self._setup_session_headers()
        # Rate limiting configuration
        self.request_delay = 3.0  # seconds between requests
        self.request_timeout = 20  # seconds for request timeout
        self.max_retries = 3  # max retries for rate-limited requests
        self.url_check = url_check
        self.exclude_url_check = exclude_url_check

        # Topic filtering configuration
        self.topics: list[TopicModel] = config.topics
        self._embedding_model = None  # Lazy-loaded
        self._topic_embeddings: np.ndarray | None = None
        if self.topics:
            logger.info(
                f"Topic filtering enabled with {len(self.topics)} topics",
            )

    def _setup_session_headers(self) -> None:
        """Set up realistic browser headers for the session."""
        self.session.headers.update(
            {
                "User-Agent": self.REQUEST_AGENT,
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
            },
        )

    def _get_embedding_model(self) -> SentenceTransformer:
        """
        Lazy-load the embedding model for topic filtering.

        Returns:
            SentenceTransformer model instance

        """
        if self._embedding_model is None:
            model_name = self.config.story_clustering.embedding_model
            self._embedding_model = get_sentence_transformer(model_name)
            logger.info("Loaded embedding model for topic filtering")
        return self._embedding_model

    def _get_topic_embeddings(self) -> np.ndarray:
        """Get or generate embeddings for all topic descriptions."""
        if self._topic_embeddings is None:
            model = self._get_embedding_model()
            # Use topic description for semantic matching
            topic_texts = [
                topic.description or topic.name
                for topic in self.topics
            ]
            self._topic_embeddings = model.encode(
                topic_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
            )
            logger.debug(f"Generated embeddings for {len(self.topics)} topics")
        return self._topic_embeddings

    def _matches_topics(self, text: str) -> tuple[bool, str | None]:
        """
        Check if text matches any configured topic.

        Uses a two-pass approach:
        1. Fast keyword matching (O(n) string search)
        2. Semantic similarity if no keyword match

        Args:
            text: Article title + description to check

        Returns:
            Tuple of (matches, matched_topic_name or None)

        """
        if not self.topics:
            # No topics configured = keep all articles
            return True, None

        text_lower = text.lower()

        # Pass 1: Fast keyword matching
        for topic in self.topics:
            for keyword in topic.keywords:
                if keyword in text_lower:
                    logger.debug(
                        f"Article matched topic '{topic.name}' via keyword "
                        f"'{keyword}'",
                    )
                    return True, topic.name

        # Pass 2: Semantic similarity
        model = self._get_embedding_model()
        topic_embeddings = self._get_topic_embeddings()

        article_embedding = model.encode(
            [text],
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
        )
        similarities = np.asarray(
            cosine_similarity(
                article_embedding,
                topic_embeddings,
            )[0],
        )
        thresholds = np.array(
            [t.similarity_threshold for t in self.topics],
            dtype=similarities.dtype,
        )
        matches = similarities >= thresholds
        match_indices = np.flatnonzero(matches)

        if match_indices.size > 0:
            i = int(match_indices[0])
            topic = self.topics[i]
            logger.debug(
                f"Article matched topic '{topic.name}' via semantic "
                f"similarity ({similarities[i]:.3f} >= "
                f"{topic.similarity_threshold})",
            )
            return True, topic.name

        return False, None

    def scrape_all_sources(self) -> list[Article]:
        """
        Scrape news from all configured sources.

        Returns:
            list of Article objects

        """
        logger.info(f"Starting news scraping for {len(self.sources)} sources")
        all_articles = []

        for source in self.sources:
            try:
                articles = self._scrape_source(source)
                all_articles.extend(articles)
                logger.info(
                    f"Scraped {len(articles)} articles from {source['name']}",
                )
            except (
                requests.exceptions.RequestException,
                OSError,
                URLError,
            ):
                logger.exception(f"Error scraping {source['name']}")

        logger.info(f"Total articles scraped: {len(all_articles)}")
        return all_articles

    def _scrape_source(self, source: NewsSource) -> list[Article]:
        """
        Scrape a single news source.

        Args:
            source: Source configuration dictionary

        Returns:
            list of Article objects

        """
        source_type = source.get("type", "rss")

        if source_type == "rss":
            return self._scrape_rss_feed(source)
        if source_type == "html":
            return self._scrape_html_listing(source)
        logger.warning(f"Unknown source type: {source_type}")
        return []

    def _try_trafilatura_fetch_url(self, url: str) -> str | None:
        """
        Try extracting content using trafilatura.fetch_url.

        Args:
            url: Article URL

        Returns:
            Extracted text or None if failed

        """
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    include_images=False,
                    include_links=False,
                )
                if text and len(text.strip()) > MIN_CONTENT_LENGTH:
                    return clean_text(text, self.config_name)
        except Exception as e:
            logger.debug(f"trafilatura.fetch_url failed for {url}: {e}")
        return None

    def _is_valid_html(self, html: str) -> bool:
        """
        Check if response contains valid HTML structure.

        Args:
            html: HTML content to validate

        Returns:
            True if HTML appears valid, False otherwise

        """
        if not html or len(html.strip()) < MIN_RESPONSE_LENGTH:
            return False

        html_lower = html.lower()

        # Check for common error page indicators
        error_indicators = [
            "rate limit",
            "too many requests",
            "access denied",
            "403 forbidden",
            "429 too many",
        ]
        if any(indicator in html_lower for indicator in error_indicators):
            return False

        # Check for basic HTML structure (should have opening tags)
        return (
            "<html" in html_lower
            or "<body" in html_lower
            or any(
                tag in html_lower
                for tag in ["<div", "<article", "<main", "<p>"]
            )
        )

    def _try_trafilatura_extract(self, html: str) -> str | None:
        """
        Try extracting content using trafilatura.extract on HTML.

        Suppresses trafilatura's internal error logging.

        Args:
            html: HTML content

        Returns:
            Extracted text or None if failed

        """
        if not self._is_valid_html(html):
            return None

        try:
            # Suppress trafilatura's error logging temporarily
            trafilatura_logger = logging.getLogger("trafilatura")
            old_level = trafilatura_logger.level
            trafilatura_logger.setLevel(logging.CRITICAL)

            try:
                text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=False,
                    include_images=False,
                    include_links=False,
                )
            finally:
                trafilatura_logger.setLevel(old_level)

            if text and len(text.strip()) > MIN_CONTENT_LENGTH:
                return clean_text(text, self.config_name)
        except Exception as e:
            # Trafilatura errors are logged by the library itself
            logger.debug(f"trafilatura.extract failed: {e}")
        return None

    def _try_beautifulsoup_extract(self, html: str) -> str | None:
        """
        Try extracting content using BeautifulSoup fallback.

        Args:
            html: HTML content

        Returns:
            Extracted text or None if failed

        """
        try:
            soup = BeautifulSoup(html, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()

            # Try to find article content in common semantic HTML5 tags
            tag_names = ["article", "main", "[role='main']", ".article-body"]
            for tag_name in tag_names:
                if tag_name.startswith((".", "[")):
                    article_content = soup.select_one(tag_name)
                else:
                    article_content = soup.find(tag_name)

                if article_content:
                    text = article_content.get_text(separator=" ", strip=True)
                    if len(text.strip()) > MIN_CONTENT_LENGTH:
                        return clean_text(text, self.config_name)

            # Fallback: get all paragraph text
            paragraphs = soup.find_all("p")
            if paragraphs:
                text = " ".join(p.get_text(strip=True) for p in paragraphs)
                if len(text.strip()) > MIN_CONTENT_LENGTH:
                    return clean_text(text, self.config_name)
        except Exception as e:
            logger.debug(f"BeautifulSoup extraction failed: {e}")
        return None

    def _extract_content_from_html(self, html: str) -> str | None:
        """
        Try extracting content from HTML using multiple strategies.

        Args:
            html: HTML content

        Returns:
            Extracted text or None if failed

        """
        # Try trafilatura extraction
        content = self._try_trafilatura_extract(html)
        if content:
            return content

        # BeautifulSoup fallback
        content = self._try_beautifulsoup_extract(html)
        if content:
            return content

        return None

    def _fetch_with_retry(self, url: str) -> requests.Response | None:
        """
        Fetch URL with retry logic for rate limiting and errors.

        Uses tenacity for automatic retries with exponential backoff
        on timeouts, connection errors, and rate limiting (429).

        Args:
            url: Article URL

        Returns:
            Response object or None if failed

        """

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=16),
            retry=retry_if_exception_type(
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    RateLimitError,
                ),
            ),
            before_sleep=before_sleep_log(
                cast("LoggerProtocol", logger), logging.WARNING,
            ),
            reraise=True,
        )
        def fetch() -> requests.Response:
            response = self.session.get(
                url,
                timeout=self.request_timeout,
                allow_redirects=True,
            )
            if response.status_code == HTTP_TOO_MANY_REQUESTS:
                msg = f"Rate limited for {url}"
                raise RateLimitError(msg)
            response.raise_for_status()
            return response

        try:
            return fetch()
        except (RetryError, RateLimitError):
            logger.warning(f"Max retries reached for {url}")
            return None
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ):
            logger.warning(f"Request failed for {url}, max retries reached")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning(f"HTTP error for {url}: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None

    def _fetch_full_content(self, url: str) -> str | None:
        """
        Fetch full article content from URL using multiple strategies.

        Tries multiple extraction methods in order:
        1. trafilatura.extract() on fetched HTML
        2. BeautifulSoup4 fallback for structured content

        Includes retry logic with exponential backoff.

        Args:
            url: Article URL

        Returns:
            Full text content or None if failed

        """
        # Add delay between requests to avoid rate limiting
        time.sleep(self.request_delay)

        # Fetch with session and extract (with retries)
        response = self._fetch_with_retry(url)
        if not response:
            return None

        # Validate HTML before attempting extraction
        if not self._is_valid_html(response.text):
            logger.debug(f"Invalid HTML detected for {url}, skipping")
            return None

        # Try extraction strategies
        return self._extract_content_from_html(response.text)

    def _parse_entry_date(self, entry: FeedParserDict) -> datetime | None:
        """Parse publication date from RSS entry."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            return datetime.fromtimestamp(
                time.mktime(entry.published_parsed),
                tz=TZ,
            )
        if hasattr(entry, "updated_parsed") and entry.updated_parsed:
            return datetime.fromtimestamp(
                time.mktime(entry.updated_parsed),
                tz=TZ,
            )
        return None

    def _extract_content(self, entry: FeedParserDict) -> tuple[str, str]:
        """Extract description and content from RSS entry."""
        description = ""
        if hasattr(entry, "summary"):
            description = clean_text(
                entry.summary,
                self.config_name,
            )
        elif hasattr(entry, "description"):
            description = clean_text(
                entry.description,
                self.config_name,
            )

        content = description
        if hasattr(entry, "content") and entry.content:
            content = clean_text(
                entry.content[0].value,
                self.config_name,
            )
        return description, content

    def _should_exclude_by_url(self, link: str) -> bool:
        """Return True if article excluded (e.g. in other configs)."""
        if not link or not self.exclude_url_check:
            return False
        return self.exclude_url_check(link)

    def _passes_topic_filter(
        self, title: str, description: str, link: str,
    ) -> bool:
        """Return True if article matches topic filter, else False."""
        filter_text = f"{title}. {description}"
        matches, matched_topic = self._matches_topics(filter_text)
        if not matches:
            logger.debug(
                f"Article filtered out (no topic match): "
                f"{title[:60]} - {link}",
            )
            return False
        logger.debug(
            f"Article matched topic '{matched_topic}': "
            f"{title[:40]} - {link}",
        )
        return True

    def _process_rss_entry(
        self,
        entry: FeedParserDict,
        source_name: str,
        cutoff_date: datetime,
    ) -> Article | None:
        """Process a single RSS entry."""
        try:
            pub_date = self._parse_entry_date(entry)

            # Skip old articles
            if pub_date and pub_date < cutoff_date:
                return None

            title = clean_text(
                entry.get("title", ""),
                self.config_name,
            )
            link = entry.get("link", "")

            # Exclude articles whose URL exists in other configs (check
            # before topic filter and content extraction)
            if self._should_exclude_by_url(link):
                logger.debug(
                    "Article excluded (URL exists in exclude configs): %s",
                    link,
                )
                return None

            description, content = self._extract_content(entry)

            # Topic filtering: check before fetching full content
            if self.topics and not self._passes_topic_filter(
                title, description, link,
            ):
                return None

            # Fetch full content from URL only if not already in
            # database
            if link:
                # Check if URL already exists in database before
                # fetching
                if self.url_check and self.url_check(link):
                    logger.debug(
                        f"Article already in database, skipping fetch: {link}",
                    )
                    # Use description/content from RSS feed if available
                    if not content:
                        content = description
                else:
                    # URL not in database, fetch full content
                    full_content = self._fetch_full_content(link)
                    if full_content:
                        content = full_content
                    elif not content:
                        # If we couldn't fetch full content and no
                        # description, skip this article
                        logger.debug(
                            "No content available for article: "
                            f"{title} - {link}",
                        )
                        return None

            return self._build_article(
                title=title,
                content=content or description,
                source_name=source_name,
                url=link,
                pub_date=pub_date,
            )
        except Exception as e:
            logger.debug(
                f"Error parsing entry from {source_name}: {e!s}",
            )
            return None

    def _build_article(
        self,
        title: str,
        content: str,
        source_name: str,
        url: str,
        pub_date: datetime | None,
    ) -> Article:
        """Build an Article, defaulting pub_date to now when missing."""
        return Article(
            title=title,
            content=content,
            source=source_name,
            url=url,
            published_date=pub_date or datetime.now(TZ),
            scraped_date=datetime.now(TZ),
        )

    def _scrape_rss_feed(self, source: NewsSource) -> list[Article]:
        """
        Scrape articles from an RSS feed.

        Args:
            source: Source configuration dictionary

        Returns:
            list of Article objects

        """
        articles = []
        cutoff_date = datetime.now(TZ) - timedelta(days=self.lookback_days)

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=16),
            retry=retry_if_exception_type((TimeoutError, URLError, OSError)),
            before_sleep=before_sleep_log(
                cast("LoggerProtocol", logger), logging.WARNING,
            ),
            reraise=False,
        )
        def _fetch_feed() -> FeedParserDict:
            socket.setdefaulttimeout(30)
            try:
                return feedparser.parse(
                    source["rss_url"],
                    agent=self.REQUEST_AGENT,
                )
            finally:
                socket.setdefaulttimeout(None)

        try:
            feed = _fetch_feed()
        except RetryError:
            logger.warning(
                "Max retries reached for RSS feed %s",
                source["rss_url"],
            )
            return articles
        except (
            requests.exceptions.RequestException,
            OSError,
            URLError,
        ):
            logger.exception(
                "Error parsing RSS feed %s",
                source["rss_url"],
            )
            return articles

        for entry in feed.entries:
            article = self._process_rss_entry(
                entry,
                source["name"],
                cutoff_date,
            )
            if article:
                articles.append(article)

            # Add delay between articles from same source
            time.sleep(1.5)

        return articles

    def _discover_article_links(
        self, html: str, base_url: str,
    ) -> list[str]:
        """
        Extract candidate article links from a listing page.

        Filters to same-domain links, drops mailto/tel/javascript and
        pure-fragment hrefs, strips fragments, dedupes preserving order,
        and caps the result at MAX_LINKS_PER_LISTING.
        """
        soup = BeautifulSoup(html, "html.parser")
        base_netloc = urlparse(base_url).netloc
        normalized_base = self._normalize_url(base_url)

        seen: dict[str, None] = {}
        for anchor in soup.find_all("a", href=True):
            href_attr = anchor.get("href") or ""
            if isinstance(href_attr, list):
                href_attr = href_attr[0] if href_attr else ""
            href = href_attr.strip()
            if not href or href.startswith("#"):
                continue
            if href.lower().startswith(_SKIP_LINK_SCHEMES):
                continue

            absolute = urljoin(base_url, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != base_netloc:
                continue

            normalized = self._normalize_url(absolute)
            if normalized == normalized_base:
                continue
            if normalized in seen:
                continue
            seen[normalized] = None
            if len(seen) >= MAX_LINKS_PER_LISTING:
                break

        return list(seen)

    @staticmethod
    def _normalize_url(url: str) -> str:
        """Strip fragment from a URL for dedup comparisons."""
        parsed = urlparse(url)
        return urlunparse(parsed._replace(fragment=""))

    def _extract_article_metadata(
        self, html: str,
    ) -> tuple[str, datetime | None]:
        """
        Extract title and published_date from an article page's HTML.

        Title sources (in order): og:title meta, <title>, first <h1>.
        Date sources (in order): article:published_time meta,
        <time datetime>, JSON-LD datePublished. Returns (title, None)
        when no parseable date is present; caller substitutes now().
        """
        soup = BeautifulSoup(html, "html.parser")
        title = self._extract_title(soup)
        pub_date = self._extract_pub_date(soup)
        return title, pub_date

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        og = soup.find("meta", attrs={"property": "og:title"})
        if og and og.get("content"):
            return str(og["content"]).strip()
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return ""

    def _extract_pub_date(self, soup: BeautifulSoup) -> datetime | None:
        meta = soup.find(
            "meta", attrs={"property": "article:published_time"},
        )
        if meta and meta.get("content"):
            parsed = self._parse_iso_datetime(str(meta["content"]))
            if parsed:
                return parsed

        time_tag = soup.find("time", attrs={"datetime": True})
        if time_tag:
            parsed = self._parse_iso_datetime(str(time_tag["datetime"]))
            if parsed:
                return parsed

        for script in soup.find_all(
            "script", attrs={"type": "application/ld+json"},
        ):
            raw = script.string
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except (ValueError, TypeError):
                continue
            for candidate in self._iter_jsonld_dates(data):
                parsed = self._parse_iso_datetime(candidate)
                if parsed:
                    return parsed
        return None

    @staticmethod
    def _iter_jsonld_dates(data: object) -> list[str]:
        """Yield datePublished values found in a JSON-LD payload."""
        results: list[str] = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "datePublished" and isinstance(value, str):
                    results.append(value)
                results.extend(NewsScraperAgent._iter_jsonld_dates(value))
        elif isinstance(data, list):
            for item in data:
                results.extend(NewsScraperAgent._iter_jsonld_dates(item))
        return results

    @staticmethod
    def _parse_iso_datetime(value: str) -> datetime | None:
        """Parse ISO 8601, normalising 'Z' and naive cases."""
        candidate = value.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=TZ)
        return parsed

    def _scrape_html_listing(self, source: NewsSource) -> list[Article]:
        """
        Scrape articles by discovering links on an HTML listing page.

        Fetches the listing URL stored in source["rss_url"], extracts
        candidate article links via _discover_article_links, then for
        each link fetches the article page and reuses the existing
        trafilatura/BeautifulSoup extraction pipeline.
        """
        listing_url = source["rss_url"]
        source_name = source["name"]
        cutoff_date = datetime.now(TZ) - timedelta(days=self.lookback_days)

        time.sleep(self.request_delay)
        listing_response = self._fetch_with_retry(listing_url)
        if not listing_response:
            return []
        if not self._is_valid_html(listing_response.text):
            logger.debug(
                "Invalid HTML for listing %s, skipping", listing_url,
            )
            return []

        links = self._discover_article_links(
            listing_response.text, listing_url,
        )
        logger.info(
            "Discovered %d candidate links on %s",
            len(links), listing_url,
        )

        articles: list[Article] = []
        for link in links:
            article = self._process_html_link(
                link, source_name, cutoff_date,
            )
            if article:
                articles.append(article)

        return articles

    def _process_html_link(
        self,
        link: str,
        source_name: str,
        cutoff_date: datetime,
    ) -> Article | None:
        """Fetch an article URL and build an Article, or return None."""
        if self._should_skip_html_link(link):
            return None

        time.sleep(self.request_delay)
        response = self._fetch_with_retry(link)
        if not response or not self._is_valid_html(response.text):
            return None

        title, pub_date = self._extract_article_metadata(response.text)
        if pub_date and pub_date < cutoff_date:
            return None
        if self.topics and not self._passes_topic_filter(
            title, "", link,
        ):
            return None

        content = self._extract_content_from_html(response.text)
        if not content:
            logger.debug("No extractable content for %s", link)
            return None

        return self._build_article(
            title=title or link,
            content=content,
            source_name=source_name,
            url=link,
            pub_date=pub_date,
        )

    def _should_skip_html_link(self, link: str) -> bool:
        """Return True if the link should be skipped before fetching."""
        if self._should_exclude_by_url(link):
            logger.debug(
                "Article excluded (URL exists in exclude configs): %s", link,
            )
            return True
        if self.url_check and self.url_check(link):
            logger.debug(
                "Article already in database, skipping fetch: %s", link,
            )
            return True
        return False
