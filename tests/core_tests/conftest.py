"""Test fixtures and configuration for newsbot tests"""

import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from newsbot.models import Article, SentimentResult
from utilities import models as config_models

logger = logging.getLogger(__name__)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(
        self,
        generate_response: str = "mocked response",
        chat_response: str = "Mocked Story Title",
        chat_json_response: dict | None = None,
    ):
        """Initialize mock provider with configurable responses."""
        self.generate_response = generate_response
        self.chat_response = chat_response
        self.chat_json_response = chat_json_response or {
            "is_valid": True,
            "violation_type": None,
        }
        self.generate_calls: list[tuple[str, dict]] = []
        self.chat_calls: list[tuple[list, dict]] = []
        self.chat_json_calls: list[tuple[list, dict, dict]] = []

    def generate(self, prompt: str, options: dict[str, Any]) -> str:
        """Mock generate method."""
        self.generate_calls.append((prompt, options))
        return self.generate_response

    def chat(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> str:
        """Mock chat method."""
        self.chat_calls.append((messages, options))
        return self.chat_response

    def chat_json(
        self,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Mock chat_json method."""
        self.chat_json_calls.append((messages, options, schema))
        return self.chat_json_response


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    return MockLLMProvider()


@pytest.fixture
def sample_config() -> config_models.ConfigModel:
    """Sample configuration for testing"""
    return config_models.ConfigModel(
        news_sources=[
            config_models.NewsSourceModel(
                name="Test News",
                rss_url="https://test.com/feed",
                type="rss",
            ),
        ],
        name="TestConfig",
        country="US",
        language="en",
        report=config_models.ReportConfigModel(
            lookback_days=7,
            format="html",
            include_summaries=True,
        ),
        sentiment=config_models.SentimentConfigModel(
            method="vader",
            comparison_threshold=0.3,
        ),
        llm=config_models.LLMConfigModel(
            provider="ollama",
            model="llama2",
            base_url="http://localhost:11434",
            temperature=0.7,
            max_tokens=2000,
        ),
        story_clustering=config_models.StoryClusteringConfigModel(
            min_sources=2,
            similarity_threshold=0.3,
            embedding_model="Xenova/all-MiniLM-L6-v2",
            algorithm="greedy",
        ),
        scheduler=config_models.SchedulerConfigModel(
            weekly_analysis=config_models.WeeklyAnalysisConfigModel(
                lookback_days=7,
            ),
        ),
    )


@pytest.fixture
def sample_article():
    """Create a sample article for testing"""
    return Article(
        title="Breaking News: Technology Advances",
        content="This is a test article about technology advances in the field of AI and machine learning.",
        source="Test News",
        url="https://test.com/article1",
        published_date=datetime.now() - timedelta(days=1),
        scraped_date=datetime.now(),
    )


@pytest.fixture
def sample_articles():
    """Create multiple sample articles for testing"""
    return [
        Article(
            title="AI Breakthrough Announced",
            content="Major AI breakthrough announced by research team. The new technology promises to revolutionize the industry.",
            source="Tech News",
            url="https://technews.com/article1",
            published_date=datetime.now() - timedelta(days=1),
            scraped_date=datetime.now(),
        ),
        Article(
            title="Artificial Intelligence Makes Strides",
            content="Recent developments in artificial intelligence show promising results in various applications.",
            source="Science Daily",
            url="https://sciencedaily.com/article1",
            published_date=datetime.now() - timedelta(days=1),
            scraped_date=datetime.now(),
        ),
        Article(
            title="Climate Change Summit Begins",
            content="World leaders gather to discuss climate change policies and initiatives.",
            source="World News",
            url="https://worldnews.com/article1",
            published_date=datetime.now() - timedelta(days=2),
            scraped_date=datetime.now(),
        ),
        Article(
            title="Climate Conference Opens",
            content="International climate conference kicks off with representatives from over 100 countries.",
            source="Global Times",
            url="https://globaltimes.com/article1",
            published_date=datetime.now() - timedelta(days=2),
            scraped_date=datetime.now(),
        ),
    ]


@pytest.fixture
def sample_sentiment_result():
    """Create a sample sentiment result for testing"""
    return SentimentResult(
        article_url="https://test.com/article1",
        source="Test News",
        polarity=0.5,
        subjectivity=0.6,
        compound=0.5,
        label="positive",
    )


@pytest.fixture
def mock_rss_feed():
    """Mock RSS feed data"""
    import time

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
            now = datetime.now()
            one_day_ago = now - timedelta(days=1)
            two_days_ago = now - timedelta(days=2)
            # time.struct_time expects 9 elements
            self.entries = [
                MockEntry(
                    "Test Article 1",
                    "https://test.com/article1",
                    "This is a test article summary",
                    time.struct_time(one_day_ago.timetuple()),
                ),
                MockEntry(
                    "Test Article 2",
                    "https://test.com/article2",
                    "Another test article summary",
                    time.struct_time(two_days_ago.timetuple()),
                ),
            ]

    return MockFeed()


@pytest.fixture(autouse=True)
def mock_environment_variables(monkeypatch):
    """Mock required environment variables for all tests."""
    # Set GEMINI_API_KEY to a mock value to prevent validation errors
    monkeypatch.setenv("GEMINI_API_KEY", "mock-gemini-api-key-for-testing")
    # Disable email sending so tests never trigger real SMTP
    monkeypatch.setenv("EMAIL_ENABLED", "false")


@pytest.fixture(autouse=True)
def mock_smtp(monkeypatch):
    """Mock SMTP so no test can open a real email connection."""
    smtp_context = MagicMock()
    smtp_context.__enter__ = MagicMock(return_value=smtp_context)
    smtp_context.__exit__ = MagicMock(return_value=False)
    mock_smtp_class = MagicMock(return_value=smtp_context)
    mock_smtp_ssl_class = MagicMock(return_value=smtp_context)
    monkeypatch.setattr("smtplib.SMTP", mock_smtp_class)
    monkeypatch.setattr("smtplib.SMTP_SSL", mock_smtp_ssl_class)


@pytest.fixture(autouse=True)
def mock_ollama_calls(monkeypatch, request):
    """Mock all ollama API calls for all tests by default.

    This fixture mocks ollama.Client, ollama.generate, ollama.pull, and ollama.list
    to prevent actual API calls during testing. Tests can opt out by using the
    'use_real_ollama' marker.
    """
    # Check if test is marked to use real ollama
    use_real_ollama = (
        request.node.get_closest_marker("use_real_ollama") is not None
    )

    # Check environment variables
    is_github_action = os.getenv("GITHUB_ACTIONS") == "true"
    mock_ollama_env = os.getenv("MOCK_OLLAMA")
    enable_ollama = os.getenv("ENABLE_OLLAMA", "false").lower() == "true"

    # Determine if we should mock ollama
    should_mock = True
    if use_real_ollama:
        # Only allow real ollama if:
        # 1. Not in CI (or explicitly enabled with MOCK_OLLAMA=false)
        # 2. ENABLE_OLLAMA is set to true
        if not is_github_action and enable_ollama:
            should_mock = False
            logger.info(
                "Using REAL ollama (test marked with @pytest.mark.use_real_ollama)"
            )
        elif is_github_action and mock_ollama_env == "false" and enable_ollama:
            should_mock = False
            logger.info(
                "Using REAL ollama in CI (MOCK_OLLAMA=false, ENABLE_OLLAMA=true)"
            )
        else:
            logger.warning(
                "Test marked with @pytest.mark.use_real_ollama but conditions not met. "
                "Mocking ollama. Set ENABLE_OLLAMA=true and ensure not in CI (or MOCK_OLLAMA=false)."
            )

    if should_mock:
        # Mock ollama.Client
        mock_client = MagicMock()
        mock_client.list.return_value = {
            "models": [{"model": "llama2", "name": "llama2"}]
        }
        mock_client.chat.return_value = {
            "message": {"content": "Mocked chat response"}
        }

        def mock_client_init(host=None):
            return mock_client

        # Mock ollama module functions
        mock_generate = MagicMock(
            return_value={"response": "Mocked generate response"}
        )
        mock_pull = MagicMock()
        mock_list = MagicMock(
            return_value={"models": [{"model": "llama2", "name": "llama2"}]}
        )

        # Patch ollama module
        try:
            import ollama

            monkeypatch.setattr(ollama, "Client", mock_client_init)
            monkeypatch.setattr(ollama, "generate", mock_generate)
            monkeypatch.setattr(ollama, "pull", mock_pull)
            monkeypatch.setattr(ollama, "list", mock_list)
        except ImportError:
            # If ollama is not installed, create a mock module
            mock_ollama_module = MagicMock()
            mock_ollama_module.Client = mock_client_init
            mock_ollama_module.generate = mock_generate
            mock_ollama_module.pull = mock_pull
            mock_ollama_module.list = mock_list
            monkeypatch.setitem(sys.modules, "ollama", mock_ollama_module)
    else:
        logger.info("Skipping ollama mocking - using real ollama")


@pytest.fixture(autouse=True)
def mock_llm_provider_in_ci(monkeypatch, request):
    """Mock LLM provider in GitHub Actions CI environment."""
    is_github_action = os.getenv("GITHUB_ACTIONS") == "true"
    mock_ollama = os.getenv("MOCK_OLLAMA")

    if is_github_action and mock_ollama != "false":
        logger.info(
            f"Running in GitHub Actions with LLM mocking enabled "
            f"(GITHUB_ACTIONS={os.getenv('GITHUB_ACTIONS')}, MOCK_OLLAMA={mock_ollama})"
        )
        # Skip mocking for tests that explicitly test LLM provider functionality
        if "test_utils_and_database" not in request.node.nodeid:
            # Create a mock provider
            mock_provider = MockLLMProvider()

            # Mock get_llm_provider to return our mock
            def mock_get_llm_provider(config):
                return mock_provider

            monkeypatch.setattr(
                "newsbot.llm_provider.get_llm_provider",
                mock_get_llm_provider,
            )

            # Also patch in the agent modules
            monkeypatch.setattr(
                "newsbot.agents.summarization_agent.get_llm_provider",
                mock_get_llm_provider,
            )
            monkeypatch.setattr(
                "newsbot.agents.judge_agent.get_llm_provider",
                mock_get_llm_provider,
            )
            monkeypatch.setattr(
                "newsbot.agents.story_clustering_agent.get_llm_provider",
                mock_get_llm_provider,
            )

            # Mock judge agent validate_and_fix to avoid LLM calls,
            # but skip this for tests that explicitly test judge agent functionality
            if "test_judge_agent" not in request.node.nodeid:
                mock_judge_validate = MagicMock(
                    side_effect=lambda output,
                    prompt_context,
                    allow_empty=False: output
                )
                monkeypatch.setattr(
                    "newsbot.agents.judge_agent.JudgeAgent.validate_and_fix",
                    mock_judge_validate,
                )
    elif is_github_action and mock_ollama == "false":
        logger.info(
            f"Running in GitHub Actions with REAL LLM (self-hosted runner) "
            f"(GITHUB_ACTIONS={os.getenv('GITHUB_ACTIONS')}, MOCK_OLLAMA={mock_ollama})"
        )
    elif not is_github_action:
        logger.info(
            "Running locally with REAL LLM (not in GitHub Actions environment)"
        )
