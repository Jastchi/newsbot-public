"""Tests for topic filtering in News Scraper Agent."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from newsbot.agents.scraper_agent import NewsScraperAgent
from utilities import models as config_models


class TestTopicFiltering:
    """Test cases for topic filtering functionality."""

    @pytest.fixture
    def config_with_topics(self) -> config_models.ConfigModel:
        """Create a config with topic filtering enabled."""
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
            topics=[
                config_models.TopicModel(
                    name="Artificial Intelligence",
                    keywords=["ai", "machine learning", "gpt", "llm"],
                    description="Articles about artificial intelligence and machine learning",
                    similarity_threshold=0.6,
                ),
                config_models.TopicModel(
                    name="Climate",
                    keywords=["climate change", "global warming", "carbon"],
                    description="Articles about climate change and environmental issues",
                    similarity_threshold=0.5,
                ),
            ],
            report=config_models.ReportConfigModel(lookback_days=7),
            story_clustering=config_models.StoryClusteringConfigModel(
                embedding_model="Xenova/all-MiniLM-L6-v2",
            ),
        )

    @pytest.fixture
    def config_without_topics(self) -> config_models.ConfigModel:
        """Create a config without topic filtering."""
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
            topics=[],  # No topics
            report=config_models.ReportConfigModel(lookback_days=7),
            story_clustering=config_models.StoryClusteringConfigModel(
                embedding_model="Xenova/all-MiniLM-L6-v2",
            ),
        )

    def test_init_with_topics(self, config_with_topics):
        """Test agent initialization with topics enabled."""
        agent = NewsScraperAgent(config_with_topics)

        assert len(agent.topics) == 2
        assert agent.topics[0].name == "Artificial Intelligence"
        assert agent.topics[1].name == "Climate"
        assert agent._embedding_model is None  # Lazy-loaded
        assert agent._topic_embeddings is None

    def test_init_without_topics(self, config_without_topics):
        """Test agent initialization without topics."""
        agent = NewsScraperAgent(config_without_topics)

        assert len(agent.topics) == 0
        assert agent._embedding_model is None
        assert agent._topic_embeddings is None

    def test_matches_topics_no_topics_configured(self, config_without_topics):
        """Test that all articles pass when no topics configured."""
        agent = NewsScraperAgent(config_without_topics)

        matches, topic_name = agent._matches_topics(
            "Random article about local bake sale"
        )

        assert matches is True
        assert topic_name is None

    def test_matches_topics_keyword_match(self, config_with_topics):
        """Test keyword matching works correctly."""
        agent = NewsScraperAgent(config_with_topics)

        # Should match "ai" keyword
        matches, topic_name = agent._matches_topics(
            "New breakthrough in AI technology announced today"
        )

        assert matches is True
        assert topic_name == "Artificial Intelligence"

    def test_matches_topics_keyword_match_case_insensitive(
        self, config_with_topics
    ):
        """Test keyword matching is case-insensitive."""
        agent = NewsScraperAgent(config_with_topics)

        # Should match "Machine Learning" -> "machine learning" keyword
        matches, topic_name = agent._matches_topics(
            "Machine Learning advances in healthcare"
        )

        assert matches is True
        assert topic_name == "Artificial Intelligence"

    def test_matches_topics_keyword_match_second_topic(
        self, config_with_topics
    ):
        """Test matching against second topic."""
        agent = NewsScraperAgent(config_with_topics)

        # Should match "climate change" keyword from Climate topic
        matches, topic_name = agent._matches_topics(
            "World leaders discuss climate change at summit"
        )

        assert matches is True
        assert topic_name == "Climate"

    @patch("newsbot.agents.scraper_agent.get_sentence_transformer")
    @patch("newsbot.agents.scraper_agent.cosine_similarity")
    def test_matches_topics_semantic_match(
        self, mock_cosine, mock_get_model, config_with_topics
    ):
        """Test semantic matching when no keyword match."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        # Similarity above threshold (0.6) for first topic
        mock_cosine.return_value = np.array([[0.7, 0.3]])

        agent = NewsScraperAgent(config_with_topics)

        # Text that doesn't contain keywords but is semantically related
        matches, topic_name = agent._matches_topics(
            "Neural networks revolutionize data processing"
        )

        assert matches is True
        assert topic_name == "Artificial Intelligence"
        mock_model.encode.assert_called()

    @patch("newsbot.agents.scraper_agent.get_sentence_transformer")
    @patch("newsbot.agents.scraper_agent.cosine_similarity")
    def test_matches_topics_no_match(
        self, mock_cosine, mock_get_model, config_with_topics
    ):
        """Test no match when article is unrelated."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        # Similarity below both thresholds
        mock_cosine.return_value = np.array([[0.2, 0.3]])

        agent = NewsScraperAgent(config_with_topics)

        # Unrelated text
        matches, topic_name = agent._matches_topics(
            "Local bakery wins best croissant award"
        )

        assert matches is False
        assert topic_name is None

    def test_get_embedding_model_lazy_loading(self, config_with_topics):
        """Test that embedding model is lazy-loaded."""
        with patch(
            "newsbot.agents.scraper_agent.get_sentence_transformer"
        ) as mock_get:
            mock_model = MagicMock()
            mock_get.return_value = mock_model

            agent = NewsScraperAgent(config_with_topics)

            # Model should not be loaded yet
            mock_get.assert_not_called()
            assert agent._embedding_model is None

            # Access the model
            result = agent._get_embedding_model()

            # Now it should be loaded
            mock_get.assert_called_once()
            assert result == mock_model
            assert agent._embedding_model == mock_model

            # Second access should not reload
            result2 = agent._get_embedding_model()
            assert mock_get.call_count == 1
            assert result2 == mock_model

    def test_get_topic_embeddings_caching(self, config_with_topics):
        """Test that topic embeddings are cached."""
        with patch(
            "newsbot.agents.scraper_agent.get_sentence_transformer"
        ) as mock_get:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])
            mock_get.return_value = mock_model

            agent = NewsScraperAgent(config_with_topics)

            # First call generates embeddings
            embeddings1 = agent._get_topic_embeddings()

            assert mock_model.encode.call_count == 1
            assert embeddings1 is not None

            # Second call uses cache
            embeddings2 = agent._get_topic_embeddings()

            assert mock_model.encode.call_count == 1  # Not called again
            assert np.array_equal(embeddings1, embeddings2)

    @patch("newsbot.agents.scraper_agent.feedparser.parse")
    @patch.object(NewsScraperAgent, "_fetch_full_content")
    def test_process_rss_entry_filters_articles(
        self, mock_fetch, mock_parse, config_with_topics
    ):
        """Test that articles are filtered during RSS processing."""
        import time
        from datetime import datetime, timedelta

        from typing import cast

        from newsbot.models import NewsSource

        mock_fetch.return_value = "Full article content about AI"

        recent_date = datetime.now() - timedelta(days=1)

        class MockEntry:
            def __init__(self, title, summary):
                self.title = title
                self.link = "http://test.com/article"
                self.summary = summary
                self.published_parsed = time.struct_time(
                    recent_date.timetuple()
                )

            def get(self, key, default=None):
                return getattr(self, key, default)

        # Create an agent with keyword matching
        agent = NewsScraperAgent(config_with_topics)

        source: NewsSource = cast(
            NewsSource,
            {
                "name": "Test News",
                "rss_url": "http://test.com/feed",
                "type": "rss",
            },
        )

        # Test with matching article
        mock_parse.return_value = MagicMock(
            entries=[MockEntry("AI breakthrough", "New AI model released")]
        )
        articles = agent._scrape_rss_feed(source)
        assert len(articles) == 1

        # Test with non-matching article
        mock_parse.return_value = MagicMock(
            entries=[
                MockEntry("Local bake sale", "Community event this weekend")
            ]
        )

        # We need to mock the semantic matching to return no match
        with patch.object(agent, "_matches_topics", return_value=(False, None)):
            articles = agent._scrape_rss_feed(source)
            assert len(articles) == 0


class TestTopicModel:
    """Test cases for TopicModel."""

    def test_topic_model_defaults(self):
        """Test TopicModel default values."""
        topic = config_models.TopicModel(name="Test Topic")

        assert topic.name == "Test Topic"
        assert topic.keywords == []
        assert topic.description == ""
        assert topic.similarity_threshold == 0.6

    def test_topic_model_full_init(self):
        """Test TopicModel with all fields."""
        topic = config_models.TopicModel(
            name="AI",
            keywords=["artificial intelligence", "ml"],
            description="AI and ML articles",
            similarity_threshold=0.7,
        )

        assert topic.name == "AI"
        assert topic.keywords == ["artificial intelligence", "ml"]
        assert topic.description == "AI and ML articles"
        assert topic.similarity_threshold == 0.7

    def test_config_model_with_topics(self):
        """Test ConfigModel includes topics field."""
        config = config_models.ConfigModel(
            name="Test",
            topics=[
                config_models.TopicModel(
                    name="Topic1",
                    keywords=["kw1"],
                ),
                config_models.TopicModel(
                    name="Topic2",
                    keywords=["kw2"],
                ),
            ],
        )

        assert len(config.topics) == 2
        assert config.topics[0].name == "Topic1"
        assert config.topics[1].name == "Topic2"
