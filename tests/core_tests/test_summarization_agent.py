"""Tests for Summarization Agent"""

from datetime import datetime
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from newsbot.agents.summarization_agent import SummarizationAgent
from newsbot.models import Article
from utilities import models as config_models


def create_mock_provider(
    generate_response: str = "Test summary",
    chat_response: str = "Chat response",
    chat_json_response: dict | None = None,
) -> MagicMock:
    """Create a mock LLM provider with configurable responses."""
    mock = MagicMock()
    mock.generate.return_value = generate_response
    mock.chat.return_value = chat_response
    mock.chat_json.return_value = chat_json_response or {
        "is_valid": True,
        "violation_type": None,
    }
    return mock


class TestSummarizationAgent:
    """Test cases for SummarizationAgent"""

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_init(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
    ):
        """Test agent initialization"""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)

        assert agent.config == sample_config
        assert agent.temperature == 0.7
        assert agent.max_tokens == 2000

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_init_with_defaults(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
    ):
        """Test initialization with minimal config"""
        config = config_models.ConfigModel()
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(config)

        assert agent.temperature == 0.0  # Default from LLMConfigModel
        assert agent.max_tokens == 16384  # Default from LLMConfigModel

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_create_summarization_prompt(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_article,
    ):
        """Test prompt creation for summarization"""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        prompt = agent._create_summarization_prompt(sample_article)

        assert isinstance(prompt, str)
        assert sample_article.title in prompt
        assert sample_article.content in prompt

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_with_custom_temperature(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_article,
    ):
        """Test summarization with custom temperature setting"""
        config = config_models.ConfigModel(
            llm=config_models.LLMConfigModel(
                model="llama2",
                temperature=0.3,
                max_tokens=1000,
            ),
        )
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(config)
        agent.summarize_article(sample_article)

        # Verify temperature was passed correctly
        call_args = mock_provider.generate.call_args
        assert call_args[0][1]["temperature"] == 0.3

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_empty_content(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
    ):
        """Test handling of article with empty content"""
        article = Article(
            title="Empty Article",
            content="",
            source="Test",
            url="https://test.com/empty",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )

        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        summary = agent.summarize_article(article)

        assert summary == "Content not available."


class TestTwoPassSummarization:
    """Test cases for two-pass (three-pass) summarization"""

    @pytest.fixture
    def sample_story(self, sample_articles):
        """Create a sample story for testing"""
        from newsbot.agents.story_clustering_agent import Story

        return Story(
            story_id="story_1",
            title="AI Breakthrough Announced",
            articles=sample_articles[:2],  # Use first 2 articles
            sources=["Tech News", "Science Daily"],
            article_count=2,
            earliest_date=sample_articles[0].published_date,
            latest_date=sample_articles[1].published_date,
        )

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_init_with_two_pass_config(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
    ):
        """Test initialization with two-pass configuration"""
        config = config_models.ConfigModel(
            summarization=config_models.SummarizationConfigModel(
                two_pass_enabled=True,
                max_articles_batch=2,
                article_order="chronological",
            ),
        )
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(config)

        assert agent.two_pass_enabled is True
        assert agent.max_articles_batch == 2
        assert agent.article_order == "chronological"

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_init_with_default_two_pass_config(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
    ):
        """Test initialization with default two-pass settings"""
        config = config_models.ConfigModel()
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(config)

        assert agent.two_pass_enabled is True
        assert agent.max_articles_batch == 1
        assert agent.article_order == "chronological"

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_story_two_pass_enabled(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test story summarization with two-pass enabled"""
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = [
            "- Article 1 summary point",
            "- Article 1 and 2 summary points",
            "AI research achieves breakthrough in machine learning.",
            "- Additional point from article 1",
            "- Additional point from article 2",
            "- Consolidated source summary",
        ]
        mock_provider.chat_json.return_value = {
            "is_valid": True,
            "violation_type": None,
        }
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        summaries = agent.summarize_story(sample_story)

        assert len(summaries) == 2
        assert sample_story.story_summary is not None
        assert len(sample_story.story_summary) > 0
        # Verify all articles have summaries
        for article in sample_story.articles:
            assert article.summary is not None
        # Verify source-level summaries are stored
        assert sample_story.source_additional_points is not None
        assert len(sample_story.source_additional_points) > 0

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_story_two_pass_disabled(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test story summarization with two-pass disabled"""
        from utilities.models import SummarizationConfigModel
        sample_config = sample_config.model_copy(
            update={
                "summarization": SummarizationConfigModel(two_pass_enabled=False)
            }
        )
        mock_provider = create_mock_provider(generate_response="Summary")
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        summaries = agent.summarize_story(sample_story)

        assert len(summaries) == 2
        # Should use batch summarization instead

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_build_story_summary(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test building cumulative story summary"""
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = [
            "- First article point",
            "- First and second article points",
        ]
        mock_provider.chat_json.return_value = {
            "is_valid": True,
            "violation_type": None,
        }
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        summary = agent._build_story_summary(sample_story)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert mock_provider.generate.call_count == 2

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_refine_story_summary(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test story summary refinement"""
        raw_summary = "- Point 1\n- Point 2\n- Point 3"
        refined = "A concise narrative summary of points 1, 2, and 3."
        mock_provider = create_mock_provider(generate_response=refined)
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._refine_story_summary(raw_summary, sample_story)

        assert result == refined
        mock_provider.generate.assert_called_once()

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_refine_story_summary_error_fallback(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test refinement fallback on error"""
        raw_summary = "- Point 1\n- Point 2"
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = Exception("API error")
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._refine_story_summary(raw_summary, sample_story)

        # Should return original summary on error
        assert result == raw_summary

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_extract_additional_points(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_article,
    ):
        """Test extracting additional points from article"""
        story_summary = "AI breakthrough announced by research team."
        additional = "- New funding secured for the project"
        mock_provider = create_mock_provider(generate_response=additional)
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._extract_additional_points(sample_article, story_summary)

        assert result == additional
        mock_provider.generate.assert_called_once()

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_extract_additional_points_empty_content(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
    ):
        """Test extracting points from article with no content"""
        article = Article(
            title="Empty",
            content="",
            source="Test",
            url="https://test.com/empty",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )
        story_summary = "Summary"
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._extract_additional_points(article, story_summary)

        assert result == "Content not available."

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_extract_additional_points_error_fallback(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_article,
    ):
        """Test additional points extraction fallback on error"""
        story_summary = "Summary"
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = Exception("API error")
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._extract_additional_points(sample_article, story_summary)

        assert isinstance(result, str)
        assert len(result) > 0

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_sort_articles_chronological(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_articles,
    ):
        """Test chronological article sorting"""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        agent.article_order = "chronological"
        sorted_articles = agent._sort_articles(sample_articles)

        # Verify chronological order (oldest first)
        for i in range(len(sorted_articles) - 1):
            assert (
                sorted_articles[i].published_date
                <= sorted_articles[i + 1].published_date
            )

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_sort_articles_by_source(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_articles,
    ):
        """Test article sorting by source"""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        agent.article_order = "source"
        sorted_articles = agent._sort_articles(sample_articles)

        # Verify source order
        for i in range(len(sorted_articles) - 1):
            assert sorted_articles[i].source <= sorted_articles[i + 1].source

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_sort_articles_unknown_order(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_articles,
    ):
        """Test article sorting with unknown order preserves original"""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        agent.article_order = "unknown"
        sorted_articles = agent._sort_articles(sample_articles)

        # Should return original order
        assert sorted_articles == sample_articles

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_create_story_summary_prompt_no_existing(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_article,
    ):
        """Test story summary prompt creation without existing summary"""
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        prompt = agent._create_story_summary_prompt(sample_article, "")

        assert isinstance(prompt, str)
        assert sample_article.title in prompt
        assert sample_article.content in prompt
        assert "Current Story Summary:" not in prompt

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_create_story_summary_prompt_with_existing(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_article,
    ):
        """Test story summary prompt creation with existing summary"""
        existing = "Previous summary points"
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        prompt = agent._create_story_summary_prompt(sample_article, existing)

        assert isinstance(prompt, str)
        assert sample_article.title in prompt
        assert sample_article.content in prompt
        assert "Current Story Summary:" in prompt
        assert existing in prompt

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_build_story_summary_with_batching(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test story summary building with batch size > 1"""
        from utilities.models import SummarizationConfigModel
        sample_config = sample_config.model_copy(
            update={
                "summarization": sample_config.summarization.model_copy(
                    update={"max_articles_batch": 2}
                )
            }
        )
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = [
            "- Batch 1 summary",
            "- Batch 1 and 2 summary",
        ]
        mock_provider.chat_json.return_value = {
            "is_valid": True,
            "violation_type": None,
        }
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        summary = agent._build_story_summary(sample_story)

        assert isinstance(summary, str)
        assert len(summary) > 0

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_build_story_summary_handles_empty_articles(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test story summary building handles articles without content"""
        sample_story.articles[0].content = ""
        mock_provider = create_mock_provider(generate_response="- Summary")
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        summary = agent._build_story_summary(sample_story)

        # Should skip empty articles but still process others
        assert isinstance(summary, str)

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_source_additional_points(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_articles,
    ):
        """Test summarizing additional points from multiple articles of a source"""
        source = "Tech News"
        additional_points_list = [
            "- New funding secured",
            "- Research team expanded",
            "- Publication date announced",
        ]
        story_summary = "AI breakthrough announced by research team."
        consolidated = "- Consolidated insights: funding and team expansion"
        mock_provider = create_mock_provider(generate_response=consolidated)
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._summarize_source_additional_points(
            source, additional_points_list, story_summary, sample_articles
        )

        assert result == consolidated
        mock_provider.generate.assert_called_once()

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_source_additional_points_empty_list(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_articles,
    ):
        """Test summarizing with empty additional points list"""
        source = "Tech News"
        additional_points_list: list[str] = []
        story_summary = "Summary"
        mock_provider = create_mock_provider()
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._summarize_source_additional_points(
            source, additional_points_list, story_summary, sample_articles
        )

        assert result == ""
        mock_provider.generate.assert_not_called()

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_source_additional_points_filters_invalid(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_articles,
    ):
        """Test that invalid points are filtered out"""
        source = "Tech News"
        additional_points_list = [
            "- Valid point 1",
            "",
            "Content not available.",
            "- Valid point 2",
        ]
        story_summary = "Summary"
        consolidated = "- Consolidated valid points"
        mock_provider = create_mock_provider(generate_response=consolidated)
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._summarize_source_additional_points(
            source, additional_points_list, story_summary, sample_articles
        )

        assert result == consolidated
        # Should only process valid points
        mock_provider.generate.assert_called_once()

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_summarize_source_additional_points_error_fallback(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_articles,
    ):
        """Test fallback behavior on error"""
        source = "Tech News"
        additional_points_list = ["- Point 1", "- Point 2"]
        story_summary = "Summary"
        mock_provider = MagicMock()
        mock_provider.generate.side_effect = Exception("API error")
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        result = agent._summarize_source_additional_points(
            source, additional_points_list, story_summary, sample_articles
        )

        # Should return first valid point as fallback
        assert result == "- Point 1"

    @patch("newsbot.agents.name_validation_agent.get_llm_provider")
    @patch("newsbot.agents.judge_agent.get_llm_provider")
    @patch("newsbot.agents.summarization_agent.get_llm_provider")
    def test_aggregate_additional_points_by_source(
        self,
        mock_get_provider,
        mock_judge_provider,
        mock_name_validation_provider,
        sample_config,
        sample_story,
    ):
        """Test aggregating additional points by source (fourth pass)"""
        # Set up articles with summaries (from pass 3)
        sample_story.articles[0].summary = "- Additional point from Tech News article 1"
        sample_story.articles[1].summary = "- Additional point from Science Daily article 1"
        story_summary = "AI breakthrough announced by research team."

        mock_provider = create_mock_provider(
            generate_response="- Consolidated Tech News insights"
        )
        mock_get_provider.return_value = mock_provider
        mock_judge_provider.return_value = mock_provider
        mock_name_validation_provider.return_value = mock_provider

        agent = SummarizationAgent(sample_config)
        agent._aggregate_additional_points_by_source(sample_story, story_summary)

        # Verify source_additional_points was set
        assert sample_story.source_additional_points is not None
        assert len(sample_story.source_additional_points) > 0
        # Verify all sources have summaries
        for source in sample_story.sources:
            assert source in sample_story.source_additional_points
