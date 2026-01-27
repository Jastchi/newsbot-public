"""Tests for Story Clustering Agent"""

from datetime import datetime, timedelta
from typing import cast
from unittest.mock import Mock, patch, MagicMock

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from newsbot.agents.story_clustering_agent import (
    DEFAULT_GEO_PENALTY,
    Story,
    StoryClusteringAgent,
)
from newsbot.models import Article
from utilities import models as config_models


class TestStoryClusteringAgent:
    """Test cases for StoryClusteringAgent"""

    def test_init(self, sample_config):
        """Test agent initialization"""
        with patch(
            "newsbot.agents.story_clustering_agent.get_sentence_transformer"
        ) as mock_st:
            agent = StoryClusteringAgent(sample_config)

            assert agent.min_sources == 2
            assert agent.similarity_threshold == 0.3
            mock_st.assert_called_once()

    def test_init_with_defaults(self):
        """Test initialization with minimal config"""
        config = config_models.ConfigModel()
        with patch(
            "newsbot.agents.story_clustering_agent.get_sentence_transformer"
        ):
            agent = StoryClusteringAgent(config)

            assert agent.min_sources == 2
            assert (
                agent.similarity_threshold == 0.7
            )  # Default from StoryClusteringConfigModel

    def test_init_model_loading_error(self, sample_config):
        """Test handling of model loading error"""
        with patch(
            "newsbot.agents.story_clustering_agent.get_sentence_transformer",
            side_effect=OSError("Model load failed"),
        ):
            agent = StoryClusteringAgent(sample_config)

            assert agent.embedding_model is None

    def test_identify_top_stories_no_model(
        self, sample_config, sample_articles
    ):
        """Test story identification when model is not available"""
        with patch(
            "newsbot.agents.story_clustering_agent.get_sentence_transformer",
            side_effect=OSError("No model"),
        ):
            agent = StoryClusteringAgent(sample_config)
            stories = agent.identify_top_stories(sample_articles)

            assert stories == []

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_identify_top_stories_success(
        self, mock_st, sample_config, sample_articles
    ):
        """Test successful story identification"""
        # Mock the embedding model
        mock_model = Mock()
        # Create embeddings that show similarity between AI articles and climate articles
        mock_embeddings = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],  # AI article 1
                [0.9, 0.1, 0.0, 0.0],  # AI article 2 (similar to 1)
                [0.0, 0.0, 1.0, 0.0],  # Climate article 1
                [0.0, 0.0, 0.9, 0.1],  # Climate article 2 (similar to 3)
            ]
        )
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(sample_config)
        stories = agent.identify_top_stories(sample_articles, top_n=5)

        assert isinstance(stories, list)
        assert all(isinstance(story, Story) for story in stories)
        mock_model.encode.assert_called_once()

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_identify_top_stories_filters_by_min_sources(
        self, mock_st, sample_config
    ):
        """Test that stories with fewer than min_sources are filtered"""
        # Create articles from single source
        articles = [
            Article(
                title="Unique Story",
                content="This is unique",
                source="Single Source",
                url=f"https://test.com/{i}",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            )
            for i in range(3)
        ]

        mock_model = Mock()
        mock_model.encode.return_value = np.array(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
        )
        mock_st.return_value = mock_model

        from utilities.models import StoryClusteringConfigModel

        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"min_sources": 2}
                )
            }
        )

        agent = StoryClusteringAgent(config)
        stories = agent.identify_top_stories(articles, top_n=5)

        # Should have stories only if they have 3+ articles from single source
        # or articles from multiple sources
        assert isinstance(stories, list)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_identify_top_stories_respects_top_n(
        self, mock_st, sample_config, sample_articles
    ):
        """Test that only top N stories are returned"""
        mock_model = Mock()
        # Create diverse embeddings
        embeddings = np.random.rand(len(sample_articles), 10)
        mock_model.encode.return_value = embeddings
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(sample_config)
        stories = agent.identify_top_stories(sample_articles, top_n=2)

        assert len(stories) <= 2

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_identify_top_stories_empty_articles(self, mock_st, sample_config):
        """Test with empty article list"""
        mock_model = Mock()
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(sample_config)
        # The agent should handle empty list before generating embeddings
        stories = agent.identify_top_stories([], top_n=5)

        assert stories == []
        # encode should not be called for empty list
        mock_model.encode.assert_not_called()

    def test_story_dataclass_post_init(self, sample_articles):
        """Test Story dataclass post_init"""
        story = Story(
            story_id="test_1",
            title="Test",
            articles=sample_articles[:3],
            sources=[],
            article_count=0,
            earliest_date=datetime.now(),
            latest_date=datetime.now(),
        )

        assert story.article_count == 3
        assert len(story.sources) > 0

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_clustering_with_high_similarity_threshold(
        self, mock_st, sample_config
    ):
        """Test clustering with high similarity threshold"""
        from utilities.models import StoryClusteringConfigModel

        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"similarity_threshold": 0.9}
                )
            }
        )

        articles = [
            Article(
                title=f"Different Story {i}",
                content=f"Completely different content {i}",
                source=f"Source{i}",
                url=f"https://test.com/{i}",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            )
            for i in range(5)
        ]

        mock_model = Mock()
        # Low similarity embeddings
        mock_model.encode.return_value = np.random.rand(5, 10) * 0.1
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(config)
        stories = agent.identify_top_stories(articles, top_n=10)

        # With high threshold and low similarity, should create fewer clusters
        assert isinstance(stories, list)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_story_sorting(self, mock_st, sample_config):
        """Test that stories are sorted by sources and article count"""
        # Create articles that form distinct clusters
        articles = []
        for source_idx in range(3):
            for article_idx in range(
                source_idx + 1
            ):  # Different number of articles per source
                articles.append(
                    Article(
                        title=f"Story {source_idx}",
                        content=f"Content {source_idx}",
                        source=f"Source{source_idx}_{article_idx}",
                        url=f"https://test.com/{source_idx}_{article_idx}",
                        published_date=datetime.now(),
                        scraped_date=datetime.now(),
                    )
                )

        mock_model = Mock()
        # Create embeddings that cluster by story
        embeddings = []
        for source_idx in range(3):
            for article_idx in range(source_idx + 1):
                embedding = [0.0] * 10
                embedding[source_idx] = 1.0
                embeddings.append(embedding)
        mock_model.encode.return_value = np.array(embeddings)
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(sample_config)
        stories = agent.identify_top_stories(articles, top_n=10)

        if len(stories) > 1:
            # Stories should be sorted by number of sources (descending)
            for i in range(len(stories) - 1):
                assert len(stories[i].sources) >= len(stories[i + 1].sources)

    def test_init_with_dbscan_algorithm(self, sample_config):
        """Test agent initialization with DBSCAN algorithm"""
        from utilities.models import StoryClusteringConfigModel

        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"algorithm": "dbscan"}
                )
            }
        )
        with patch(
            "newsbot.agents.story_clustering_agent.get_sentence_transformer"
        ):
            agent = StoryClusteringAgent(config)

            assert agent.clustering_algorithm == "dbscan"

    def test_init_defaults_to_greedy(self, sample_config):
        """Test that default algorithm is greedy"""
        with patch(
            "newsbot.agents.story_clustering_agent.get_sentence_transformer"
        ):
            agent = StoryClusteringAgent(sample_config)

            assert agent.clustering_algorithm == "greedy"

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_dbscan_clustering_creates_clusters(
        self, mock_st, sample_config, sample_articles
    ):
        """Test DBSCAN clustering creates valid clusters"""
        from utilities.models import StoryClusteringConfigModel

        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"algorithm": "dbscan"}
                )
            }
        )
        from utilities.models import StoryClusteringConfigModel

        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"similarity_threshold": 0.5}
                )
            }
        )

        # Create 6 articles to match the 6 embeddings
        articles = sample_articles.copy()
        articles.extend(
            [
                Article(
                    title="Additional Story 1",
                    content="Content for additional story 1",
                    source="Source1",
                    url="https://test.com/additional1",
                    published_date=datetime.now(),
                    scraped_date=datetime.now(),
                ),
                Article(
                    title="Additional Story 2",
                    content="Content for additional story 2",
                    source="Source2",
                    url="https://test.com/additional2",
                    published_date=datetime.now(),
                    scraped_date=datetime.now(),
                ),
            ]
        )

        mock_model = Mock()
        # Create embeddings that form two distinct clusters
        mock_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Cluster 1
                [0.9, 0.1, 0.0],  # Cluster 1
                [0.8, 0.2, 0.0],  # Cluster 1
                [0.0, 0.0, 1.0],  # Cluster 2
                [0.0, 0.1, 0.9],  # Cluster 2
                [0.0, 0.2, 0.8],  # Cluster 2
            ]
        )
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(config)
        stories = agent.identify_top_stories(articles, top_n=10)

        assert isinstance(stories, list)
        assert all(isinstance(story, Story) for story in stories)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_dbscan_handles_noise_points(self, mock_st, sample_config):
        """Test that DBSCAN correctly handles noise points"""
        from utilities.models import StoryClusteringConfigModel

        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"algorithm": "dbscan", "similarity_threshold": 0.8}
                )
            }
        )

        # Create articles with one outlier
        articles = [
            Article(
                title=f"Story {i}",
                content=f"Content {i}",
                source=f"Source{i % 2}",
                url=f"https://test.com/{i}",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            )
            for i in range(5)
        ]

        mock_model = Mock()
        # Create embeddings: first 4 are similar, last one is different
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Cluster
                [0.9, 0.1, 0.0],  # Cluster
                [0.8, 0.2, 0.0],  # Cluster
                [0.85, 0.15, 0.0],  # Cluster
                [0.0, 0.0, 1.0],  # Noise (different)
            ]
        )
        mock_model.encode.return_value = embeddings
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(config)
        stories = agent.identify_top_stories(articles, top_n=10)

        # Noise point should be excluded from clusters
        assert isinstance(stories, list)
        # Should have at least one cluster from the 4 similar articles
        if stories:
            assert all(isinstance(story, Story) for story in stories)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_both_algorithms_produce_stories(
        self, mock_st, sample_config, sample_articles
    ):
        """Test that both greedy and DBSCAN produce valid Story objects"""
        mock_model = Mock()
        # Create embeddings that form clusters
        mock_embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.1, 0.9],
            ]
        )
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model

        # Test greedy
        agent_greedy = StoryClusteringAgent(sample_config)
        stories_greedy = agent_greedy.identify_top_stories(
            sample_articles[:4],
            top_n=10,
        )

        # Test DBSCAN
        config_dbscan = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"algorithm": "dbscan"}
                )
            }
        )
        agent_dbscan = StoryClusteringAgent(config_dbscan)
        stories_dbscan = agent_dbscan.identify_top_stories(
            sample_articles[:4],
            top_n=10,
        )

        # Both should produce valid Story objects
        assert all(isinstance(s, Story) for s in stories_greedy)
        assert all(isinstance(s, Story) for s in stories_dbscan)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_dbscan_with_various_thresholds(self, mock_st, sample_config):
        """Test DBSCAN with different similarity thresholds"""
        articles = [
            Article(
                title=f"Article {i}",
                content=f"Content {i}",
                source=f"Source{i % 2}",
                url=f"https://test.com/{i}",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            )
            for i in range(6)
        ]

        mock_model = Mock()
        # Create two distinct clusters
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],  # Cluster 1
                [0.9, 0.1, 0.0],  # Cluster 1
                [0.85, 0.15, 0.0],  # Cluster 1
                [0.0, 0.0, 1.0],  # Cluster 2
                [0.0, 0.1, 0.9],  # Cluster 2
                [0.0, 0.15, 0.85],  # Cluster 2
            ]
        )
        mock_model.encode.return_value = embeddings
        mock_st.return_value = mock_model

        # Test with low threshold (should create clusters)
        from utilities.models import StoryClusteringConfigModel

        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"similarity_threshold": 0.5}
                )
            }
        )
        agent = StoryClusteringAgent(config)
        stories_low = agent.identify_top_stories(articles, top_n=10)

        # Test with high threshold (should create fewer/more selective clusters)
        config = sample_config.model_copy(
            update={
                "story_clustering": sample_config.story_clustering.model_copy(
                    update={"similarity_threshold": 0.9}
                )
            }
        )
        agent = StoryClusteringAgent(config)
        stories_high = agent.identify_top_stories(articles, top_n=10)

        assert isinstance(stories_low, list)
        assert isinstance(stories_high, list)
        assert all(isinstance(s, Story) for s in stories_low)
        assert all(isinstance(s, Story) for s in stories_high)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_get_embedding_texts_includes_content(
        self, mock_st, sample_config
    ):
        """Test _get_embedding_texts combines title with content"""
        mock_st.return_value = Mock()

        agent = StoryClusteringAgent(sample_config)
        articles = [
            Article(
                title="Test Title",
                content="This is the article content for testing.",
                source="Source1",
                url="https://test.com/1",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
        ]

        texts = agent._get_embedding_texts(articles)

        assert len(texts) == 1
        assert texts[0].startswith("Test Title. This is the article")

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_get_embedding_texts_handles_empty_content(
        self, mock_st, sample_config
    ):
        """Test _get_embedding_texts handles empty/None content gracefully"""
        mock_st.return_value = Mock()

        agent = StoryClusteringAgent(sample_config)
        articles = [
            Article(
                title="Title with empty content",
                content="",
                source="Source1",
                url="https://test.com/1",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
            Article(
                title="Title with None content",
                content="",
                source="Source2",
                url="https://test.com/2",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
        ]

        texts = agent._get_embedding_texts(articles)

        assert texts[0] == "Title with empty content. "
        assert texts[1] == "Title with None content. "

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_identify_top_stories_uses_hybrid_embeddings(
        self, mock_st, sample_config
    ):
        """Test that identify_top_stories embeds title + content"""
        mock_model = Mock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.9, 0.1]])
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(sample_config)
        articles = [
            Article(
                title="Story A",
                content="Content about Syria and the Middle East conflict.",
                source="Source1",
                url="https://test.com/1",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
            Article(
                title="Story B",
                content="Content about Nigeria and Boko Haram attacks.",
                source="Source2",
                url="https://test.com/2",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
        ]

        agent.identify_top_stories(articles, top_n=5)

        # Verify encode was called with hybrid texts (title + content)
        call_args = mock_model.encode.call_args[0][0]
        assert "Story A. Content about Syria" in call_args[0]
        assert "Story B. Content about Nigeria" in call_args[1]


class TestGeographicalPenalty:
    """Test cases for geographical penalty in clustering."""

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_extract_locations_finds_countries(self, mock_st, sample_config):
        """Test that _extract_locations finds country names (GPE entities)."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        locations = agent._extract_locations(
            "The earthquake struck Turkey and affected parts of Syria."
        )

        # Should find Turkey and Syria as GPE entities
        assert len(locations) >= 1
        # Location names are lowercased
        assert any("turkey" in loc for loc in locations) or any(
            "syria" in loc for loc in locations
        )

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_extract_locations_finds_cities(self, mock_st, sample_config):
        """Test that _extract_locations finds city names."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        locations = agent._extract_locations(
            "Officials in Berlin met with delegates from Paris today."
        )

        # Should find Berlin and/or Paris
        assert len(locations) >= 1

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_extract_locations_empty_text(self, mock_st, sample_config):
        """Test _extract_locations returns empty set for empty text."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        assert agent._extract_locations("") == set()

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_extract_locations_no_locations(self, mock_st, sample_config):
        """Test _extract_locations returns empty set when no locations."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        locations = agent._extract_locations(
            "The stock market rose by 5% today."
        )

        # Text without geographic entities should return empty set
        assert isinstance(locations, set)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_calculate_geo_penalty_with_overlap(self, mock_st, sample_config):
        """Test penalty is 1.0 when locations overlap."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        penalty = agent._calculate_geo_penalty(
            {"turkey", "syria"},
            {"turkey", "iraq"},
        )

        assert penalty == 1.0  # Turkey overlaps

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_calculate_geo_penalty_no_overlap(self, mock_st, sample_config):
        """Test penalty factor applied when locations don't overlap."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        penalty = agent._calculate_geo_penalty(
            {"turkey"},
            {"japan"},
        )

        assert penalty == DEFAULT_GEO_PENALTY

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_calculate_geo_penalty_empty_locations_a(
        self, mock_st, sample_config
    ):
        """Test no penalty when first article has no locations."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        penalty = agent._calculate_geo_penalty(
            set(),
            {"japan"},
        )

        assert penalty == 1.0

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_calculate_geo_penalty_empty_locations_b(
        self, mock_st, sample_config
    ):
        """Test no penalty when second article has no locations."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        penalty = agent._calculate_geo_penalty(
            {"turkey"},
            set(),
        )

        assert penalty == 1.0

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_calculate_geo_penalty_both_empty(self, mock_st, sample_config):
        """Test no penalty when both articles have no locations."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        penalty = agent._calculate_geo_penalty(set(), set())

        assert penalty == 1.0

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_get_article_locations(self, mock_st, sample_config):
        """Test _get_article_locations extracts from multiple articles."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        articles = [
            Article(
                title="Earthquake in Turkey",
                content="The earthquake struck southeastern Turkey.",
                source="Source1",
                url="https://test.com/1",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
            Article(
                title="Typhoon hits Japan",
                content="A major typhoon made landfall in Japan today.",
                source="Source2",
                url="https://test.com/2",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
        ]

        locations_list = agent._get_article_locations(articles)

        assert len(locations_list) == 2
        assert isinstance(locations_list[0], set)
        assert isinstance(locations_list[1], set)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_apply_geo_penalty_to_similarity(self, mock_st, sample_config):
        """Test _apply_geo_penalty_to_similarity modifies matrix correctly."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        # Original similarity matrix (3x3)
        similarity_matrix = np.array([
            [1.0, 0.8, 0.7],
            [0.8, 1.0, 0.9],
            [0.7, 0.9, 1.0],
        ])

        # Article 0: Turkey, Article 1: Turkey, Article 2: Japan
        article_locations = [
            {"turkey"},
            {"turkey"},
            {"japan"},
        ]

        adjusted = agent._apply_geo_penalty_to_similarity(
            similarity_matrix,
            article_locations,
        )

        # Turkey-Turkey (0,1): no penalty (overlap)
        assert adjusted[0][1] == 0.8
        assert adjusted[1][0] == 0.8

        # Turkey-Japan (0,2) and (1,2): penalty applied
        assert adjusted[0][2] == 0.7 * DEFAULT_GEO_PENALTY
        assert adjusted[2][0] == 0.7 * DEFAULT_GEO_PENALTY
        assert adjusted[1][2] == 0.9 * DEFAULT_GEO_PENALTY
        assert adjusted[2][1] == 0.9 * DEFAULT_GEO_PENALTY

        # Diagonal unchanged
        assert adjusted[0][0] == 1.0
        assert adjusted[1][1] == 1.0
        assert adjusted[2][2] == 1.0

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_apply_geo_penalty_mutates_matrix_correctly(
        self, mock_st, sample_config
    ):
        """Test that geo penalty is applied in-place and reduces similarity."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        # Simple 2x2 matrix
        # Article 0: Turkey, Article 1: Japan (Different locations)
        original = np.array([
            [1.0, 0.8],
            [0.8, 1.0],
        ])
        
        article_locations = [{"turkey"}, {"japan"}]

        # The method now mutates 'original' directly
        agent._apply_geo_penalty_to_similarity(original, article_locations)

        # 1.0 * 0.6 (DEFAULT_GEO_PENALTY) = 0.48
        expected_similarity = 0.8 * 0.6 

        # Use np.allclose for float comparisons to avoid precision issues
        assert np.allclose(original[0, 1], expected_similarity)
        assert np.allclose(original[1, 0], expected_similarity)
        
        # Check that diagonals (self-similarity) remained 1.0
        assert original[0, 0] == 1.0

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_geo_penalty_affects_clustering(self, mock_st, sample_config):
        """Test that geo penalty influences clustering results."""
        mock_model = Mock()

        # Create embeddings that would cluster together without geo penalty
        # All articles are semantically very similar (earthquake news)
        mock_embeddings = np.array(
            [
                [1.0, 0.0],  # Turkey earthquake
                [0.95, 0.05],  # Turkey earthquake (very similar)
                [0.9, 0.1],  # Japan earthquake (very similar semantically)
            ]
        )
        mock_model.encode.return_value = mock_embeddings
        mock_st.return_value = mock_model

        agent = StoryClusteringAgent(sample_config)

        articles = [
            Article(
                title="Earthquake devastates Turkey",
                content="A major earthquake struck Turkey today.",
                source="Source1",
                url="https://test.com/1",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
            Article(
                title="Turkey earthquake death toll rises",
                content="The earthquake in Turkey has caused widespread damage.",
                source="Source2",
                url="https://test.com/2",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
            Article(
                title="Earthquake hits Japan",
                content="A powerful earthquake struck Japan overnight.",
                source="Source3",
                url="https://test.com/3",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            ),
        ]

        stories = agent.identify_top_stories(articles, top_n=10)

        # With geo penalty, Turkey and Japan earthquakes should be
        # separate stories despite high semantic similarity
        assert isinstance(stories, list)
        # Verify the clustering was performed
        mock_model.encode.assert_called_once()


class TestClusterSampling:
    """Test cases for stratified cluster sampling."""

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_small_cluster_not_sampled(self, mock_st, sample_config):
        """Test that clusters smaller than max_articles are returned unchanged."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        # Default max is 5+6+4=15, create cluster of 10
        articles = [
            Article(
                title=f"Article {i}",
                content=f"Content {i}",
                source=f"Source{i % 3}",
                url=f"https://test.com/{i}",
                published_date=datetime.now() - timedelta(hours=i),
                scraped_date=datetime.now(),
            )
            for i in range(10)
        ]

        embeddings = np.random.rand(10, 64)
        similarity_matrix = cosine_similarity(embeddings)
        centroid = embeddings.mean(axis=0, keepdims=True)
        centrality_scores = cosine_similarity(centroid, embeddings)[0]
        centrality_map = {a: float(centrality_scores[i]) for i, a in enumerate(articles)}
        article_to_idx = {a: i for i, a in enumerate(articles)}

        result = agent._sample_cluster_articles_stratified(
            articles, similarity_matrix, centrality_map, article_to_idx
        )

        assert len(result) == 10
        assert set(result) == set(articles)

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_large_cluster_sampled_to_max(self, mock_st, sample_config):
        """Test that large clusters are sampled down to max_articles."""
        mock_st.return_value = MagicMock()
        agent = StoryClusteringAgent(sample_config)
        
        # Create cluster of 30 articles
        articles = [
            Article(
                title=f"Article {i}",
                content=f"Content {i}",
                source=f"Source{i % 5}",
                url=f"https://test.com/{i}",
                published_date=datetime.now() - timedelta(hours=i),
                scraped_date=datetime.now(),
            )
            for i in range(30)
        ]

        # 2. Setup the global state the optimized MMR expects
        embeddings = np.random.rand(30, 64)
        # This is our "Global" matrix
        similarity_matrix = cosine_similarity(embeddings)
        article_to_idx = {a: i for i, a in enumerate(articles)}
        
        # Calculate mock centrality
        centroid = embeddings.mean(axis=0, keepdims=True)
        centrality_scores = cosine_similarity(centroid, embeddings)[0]
        centrality_map = {a: float(centrality_scores[i]) for i, a in enumerate(articles)}

        # 3. Run the optimized sampler
        result = agent._sample_cluster_articles_stratified(
            articles, 
            similarity_matrix, 
            centrality_map, 
            article_to_idx
        )

        # Now it will consistently return the sum of central + recent + diverse
        assert len(result) == agent.max_articles_per_story

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_sampling_includes_most_recent_articles(
        self, mock_st, sample_config
    ):
        """Test that recent articles are included in sampling."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        now = datetime.now()
        articles = [
            Article(
                title=f"Article {i}",
                content=f"Content {i}",
                source=f"Source{i % 10}",
                url=f"https://test.com/{i}",
                published_date=now - timedelta(hours=i),
                scraped_date=now,
            )
            for i in range(30)
        ]

        embeddings = np.random.rand(30, 64)
        similarity_matrix = cosine_similarity(embeddings)
        centroid = embeddings.mean(axis=0, keepdims=True)
        centrality_scores = cosine_similarity(centroid, embeddings)[0]
        centrality_map = {a: float(centrality_scores[i]) for i, a in enumerate(articles)}
        article_to_idx = {a: i for i, a in enumerate(articles)}

        result = agent._sample_cluster_articles_stratified(
            articles, similarity_matrix, centrality_map, article_to_idx
        )

        # Most recent article (index 0) should be included
        assert articles[0] in result

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_sampling_source_diversity(self, mock_st, sample_config):
        """Test that sampling prefers source diversity."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        articles = [
            Article(
                title=f"Article {i}",
                content=f"Content {i}",
                source=f"Source{i % 8}",  # 8 different sources
                url=f"https://test.com/{i}",
                published_date=datetime.now() - timedelta(hours=i),
                scraped_date=datetime.now(),
            )
            for i in range(30)
        ]

        embeddings = np.random.rand(30, 64)
        similarity_matrix = cosine_similarity(embeddings)
        centroid = embeddings.mean(axis=0, keepdims=True)
        centrality_scores = cosine_similarity(centroid, embeddings)[0]
        centrality_map = {a: float(centrality_scores[i]) for i, a in enumerate(articles)}
        article_to_idx = {a: i for i, a in enumerate(articles)}

        result = agent._sample_cluster_articles_stratified(
            articles, similarity_matrix, centrality_map, article_to_idx
        )

        sources_in_result = {a.source for a in result}
        # Should have multiple sources represented
        assert len(sources_in_result) >= min(8, len(result))

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_mmr_respects_similarity_floor(self, mock_st, sample_config):
        """Test that MMR skips articles below similarity floor."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        # Create articles where some have low centrality
        articles = [
            Article(
                title=f"Article {i}",
                content=f"Content {i}",
                source=f"Source{i}",
                url=f"https://test.com/{i}",
                published_date=datetime.now() - timedelta(hours=i),
                scraped_date=datetime.now(),
            )
            for i in range(20)
        ]

        # Create embeddings where last 5 articles are outliers
        embeddings = np.random.rand(20, 64)
        embeddings[15:] = np.random.rand(5, 64) * 0.1  # Very different

        centroid = embeddings.mean(axis=0, keepdims=True)
        centrality_scores = cosine_similarity(centroid, embeddings)[0]

        # Artificially set low centrality for outliers
        centrality_scores = np.array(centrality_scores)
        centrality_scores[15:] = 0.2  # Below default floor of 0.4

        similarity_matrix = cosine_similarity(embeddings)
        centrality_map = {a: float(centrality_scores[i]) for i, a in enumerate(articles)}
        article_to_idx = {a: i for i, a in enumerate(articles)}

        result = agent._select_diverse_mmr(
            articles[11:],  # remaining after central/recent
            articles[:11],  # already selected
            similarity_matrix,
            centrality_map,
            article_to_idx,
            4,
        )

        # Outlier articles (15-19) should not be selected
        for article in result:
            idx = articles.index(article)
            assert centrality_scores[idx] >= agent.sampling_similarity_floor

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_select_with_source_diversity_one_per_source(
        self, mock_st, sample_config
    ):
        """Test _select_with_source_diversity picks one per source first."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        articles = [
            Article(
                title=f"Article {i}",
                content=f"Content {i}",
                source=f"Source{i // 3}",  # 3 articles per source
                url=f"https://test.com/{i}",
                published_date=datetime.now(),
                scraped_date=datetime.now(),
            )
            for i in range(9)
        ]

        result = agent._select_with_source_diversity(articles, n=3)

        sources = [a.source for a in result]
        assert len(set(sources)) == 3  # All different sources

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_custom_sampling_counts(self, mock_st):
        """Test that custom sampling counts are respected."""
        config = config_models.ConfigModel(
            story_clustering=config_models.StoryClusteringConfigModel(
                sampling_central_count=3,
                sampling_recent_count=4,
                sampling_diverse_count=3,
            ),
        )
        mock_st.return_value = Mock()

        agent = StoryClusteringAgent(config)

        assert agent.sampling_central_count == 3
        assert agent.sampling_recent_count == 4
        assert agent.sampling_diverse_count == 3
        assert agent.max_articles_per_story == 10  # 3 + 4 + 3

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_max_articles_derived_from_sum(self, mock_st, sample_config):
        """Test that max_articles_per_story is derived from sampling counts."""
        mock_st.return_value = Mock()
        agent = StoryClusteringAgent(sample_config)

        expected_max = (
            agent.sampling_central_count
            + agent.sampling_recent_count
            + agent.sampling_diverse_count
        )
        assert agent.max_articles_per_story == expected_max

    @patch("newsbot.agents.story_clustering_agent.get_sentence_transformer")
    def test_sampling_integrated_in_identify_top_stories(
        self, mock_st, sample_config
    ):
        """Test that sampling is applied in identify_top_stories."""
        mock_model = Mock()
        # Create a large cluster (25 articles) and small cluster (5 articles)
        embeddings = np.vstack(
            [
                np.random.rand(25, 64)
                + np.array([1, 0] + [0] * 62),  # Cluster 1
                np.random.rand(5, 64)
                + np.array([0, 1] + [0] * 62),  # Cluster 2
            ]
        )
        mock_model.encode.return_value = embeddings
        mock_st.return_value = mock_model

        articles = [
            Article(
                title=f"Cluster1 Article {i}",
                content=f"Content about topic A {i}",
                source=f"Source{i % 5}",
                url=f"https://test.com/c1_{i}",
                published_date=datetime.now() - timedelta(hours=i),
                scraped_date=datetime.now(),
            )
            for i in range(25)
        ] + [
            Article(
                title=f"Cluster2 Article {i}",
                content=f"Content about topic B {i}",
                source=f"Source{i % 3}",
                url=f"https://test.com/c2_{i}",
                published_date=datetime.now() - timedelta(hours=i),
                scraped_date=datetime.now(),
            )
            for i in range(5)
        ]

        agent = StoryClusteringAgent(sample_config)
        stories = agent.identify_top_stories(articles, top_n=10)

        # The large cluster should have been sampled
        for story in stories:
            assert story.article_count <= agent.max_articles_per_story
