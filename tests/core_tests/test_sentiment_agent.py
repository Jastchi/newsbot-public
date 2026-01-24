"""Tests for Sentiment Analysis Agent"""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, cast

import pytest

from newsbot.agents.sentiment_agent import SentimentAnalysisAgent
from newsbot.models import Article, SentimentResult

if TYPE_CHECKING:
    from pysentimiento.analyzer import AnalyzerForSequenceClassification
else:
    AnalyzerForSequenceClassification = Any


class TestSentimentAnalysisAgent:
    """Test cases for SentimentAnalysisAgent"""

    def test_init_vader(self, sample_config):
        """Test agent initialization with VADER"""
        agent = SentimentAnalysisAgent(sample_config)

        assert agent.config == sample_config
        assert agent.method == "vader"
        assert agent.comparison_threshold == 0.3
        assert hasattr(agent, "vader")

    def test_init_textblob(self, sample_config):
        """Test agent initialization with TextBlob"""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={
                "sentiment": sample_config.sentiment.model_copy(
                    update={"method": "textblob"}
                )
            }
        )

        agent = SentimentAnalysisAgent(config)
        assert agent.method == "textblob"

    def test_init_hybrid(self, sample_config):
        """Test agent initialization with hybrid method"""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={
                "sentiment": sample_config.sentiment.model_copy(
                    update={"method": "hybrid"}
                )
            }
        )

        agent = SentimentAnalysisAgent(config)
        assert agent.method == "hybrid"
        assert hasattr(agent, "vader")

    def test_analyze_article_vader(self, sample_config, sample_article):
        """Test sentiment analysis with VADER"""
        agent = SentimentAnalysisAgent(sample_config)
        result = agent.analyze_article(sample_article)

        assert isinstance(result, SentimentResult)
        assert result.article_url == sample_article.url
        assert result.source == sample_article.source
        assert -1 <= result.polarity <= 1
        assert 0 <= result.subjectivity <= 1
        assert -1 <= result.compound <= 1
        assert result.label in ["positive", "negative", "neutral"]
        assert sample_article.sentiment == result

    def test_analyze_article_with_positive_sentiment(self, sample_config):
        """Test analysis of positive article"""
        article = Article(
            title="Amazing Success Story",
            content="This is wonderful news! Everyone is extremely happy and excited about the fantastic results.",
            source="Test News",
            url="https://test.com/positive",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )

        agent = SentimentAnalysisAgent(sample_config)
        result = agent.analyze_article(article)

        assert result.label == "positive"
        assert result.polarity > 0

    def test_analyze_article_with_negative_sentiment(self, sample_config):
        """Test analysis of negative article"""
        article = Article(
            title="Terrible Disaster Strikes",
            content="This is horrible news. The tragic situation has caused terrible suffering and devastating losses.",
            source="Test News",
            url="https://test.com/negative",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )

        agent = SentimentAnalysisAgent(sample_config)
        result = agent.analyze_article(article)

        assert result.label == "negative"
        assert result.polarity < 0

    def test_analyze_article_error_handling(
        self, sample_config, sample_article
    ):
        """Test error handling during sentiment analysis"""
        agent = SentimentAnalysisAgent(sample_config)

        # Create article with empty content
        bad_article = Article(
            title="",
            content="",
            source="Test",
            url="https://test.com",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )

        result = agent.analyze_article(bad_article)

        assert isinstance(result, SentimentResult)
        assert result.label == "neutral"
        assert result.polarity == 0.0

    def test_init_pysentimiento(self, sample_config):
        """Test agent initialization with pysentimiento"""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={"sentiment": SentimentConfigModel(method="pysentimiento")}
        )

        agent = SentimentAnalysisAgent(config)

        assert agent.method == "pysentimiento"
        assert agent._pysentimiento_analyzer is not None

    def test_analyze_article_pysentimiento(self, sample_config):
        """Test sentiment analysis with pysentimiento"""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={"sentiment": SentimentConfigModel(method="pysentimiento")}
        )

        article = Article(
            title="Amazing Success Story",
            content="This is wonderful news! Everyone is extremely happy.",
            source="Test News",
            url="https://test.com/positive",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )

        agent = SentimentAnalysisAgent(config)
        result = agent.analyze_article(article)

        assert isinstance(result, SentimentResult)
        assert result.article_url == article.url
        assert -1 <= result.polarity <= 1
        assert result.label in ["positive", "negative", "neutral"]

    def test_pysentimiento_positive_sentiment(self, sample_config):
        """Test pysentimiento detects positive sentiment"""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={"sentiment": SentimentConfigModel(method="pysentimiento")}
        )

        article = Article(
            title="Great News",
            content="This is fantastic! Amazing results, everyone is thrilled.",
            source="Test News",
            url="https://test.com/positive",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )

        agent = SentimentAnalysisAgent(config)
        result = agent.analyze_article(article)

        assert result.label == "positive"
        assert result.polarity > 0

    def test_pysentimiento_negative_sentiment(self, sample_config):
        """Test pysentimiento detects negative sentiment"""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={"sentiment": SentimentConfigModel(method="pysentimiento")}
        )

        article = Article(
            title="Terrible Disaster",
            content="This is awful. Horrible tragedy with devastating losses.",
            source="Test News",
            url="https://test.com/negative",
            published_date=datetime.now(),
            scraped_date=datetime.now(),
        )

        agent = SentimentAnalysisAgent(config)
        result = agent.analyze_article(article)

        assert result.label == "negative"
        assert result.polarity < 0

    def test_pysentimiento_non_english_fallback(self, sample_config):
        """Test pysentimiento falls back to TextBlob for non-English configs"""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={
                "language": "de",  # German
                "sentiment": SentimentConfigModel(method="pysentimiento")
            }
        )

        agent = SentimentAnalysisAgent(config)

        # Should fall back to textblob
        assert agent.method == "textblob"
        assert agent._pysentimiento_analyzer is None

    def test_pysentimiento_analysis_mapping(self, sample_config, monkeypatch):
        """Ensure pysentimiento results map to our sentiment fields."""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={"sentiment": SentimentConfigModel(method="pysentimiento")}
        )

        agent = SentimentAnalysisAgent(config)
        fake_result = type(
            "Res",
            (),
            {"output": "POS", "probas": {"POS": 0.7, "NEG": 0.2, "NEU": 0.1}},
        )

        mock_analyzer = type(
            "Analyzer",
            (),
            {"predict": lambda *_: fake_result},
        )()
        agent._pysentimiento_analyzer = cast(
            AnalyzerForSequenceClassification,
            mock_analyzer,
        )

        output = agent._analyze_pysentimiento("text")

        assert output["label"] == "positive"
        assert output["compound"] == pytest.approx(0.5)

    def test_hybrid_combines_vader_and_textblob(
        self, sample_config, monkeypatch
    ):
        """Hybrid method averages VADER and TextBlob scores."""
        from utilities.models import SentimentConfigModel
        config = sample_config.model_copy(
            update={
                "sentiment": sample_config.sentiment.model_copy(
                    update={"method": "hybrid"}
                )
            }
        )

        agent = SentimentAnalysisAgent(config)
        agent._analyze_vader: Callable = lambda *_: {"compound": 0.6}
        agent._analyze_textblob: Callable = lambda *_: {
            "polarity": -0.19,
            "subjectivity": 0.1,
        }

        result = agent._analyze_hybrid("sample text")

        # Average: (0.6 + (-0.19)) / 2 = 0.205, which is clearly > 0.2 threshold
        assert result["compound"] == pytest.approx(0.205)
        assert result["label"] == "positive"

    def test_identify_differences_sorts_results(self, sample_config):
        """Differences list is sorted by magnitude."""
        agent = SentimentAnalysisAgent(sample_config)
        comparison = {
            "A": {"avg_sentiment": 0.9},
            "B": {"avg_sentiment": 0.1},
            "C": {"avg_sentiment": -0.2},
        }

        diffs = agent._identify_differences(comparison)

        assert diffs[0]["source1"] == "A"
        assert diffs[0]["source2"] == "C"
