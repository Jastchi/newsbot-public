"""
Sentiment Analysis Agent.

Analyzes sentiment of articles and compares across sources.

Supported methods: vader, textblob, pysentimiento, hybrid.
pysentimiento requires English language; falls back to TextBlob for
other languages.
"""

import logging
from typing import TYPE_CHECKING

import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from newsbot.constants import POLARITY_THRESHOLD, SENTIMENT_THRESHOLD
from newsbot.models import (
    Article,
    SentimentAnalysisDict,
    SentimentDifference,
    SentimentResult,
)
from utilities.models import ConfigModel

if TYPE_CHECKING:
    from pysentimiento.analyzer import AnalyzerForSequenceClassification
else:
    AnalyzerForSequenceClassification = None

logger = logging.getLogger(__name__)


class SentimentAnalysisAgent:
    """Agent responsible for sentiment analysis."""

    def __init__(self, config: ConfigModel) -> None:
        """
        Initialize the Sentiment Analysis Agent.

        Args:
            config: Configuration dictionary

        """
        self.config = config
        self.sentiment_config = config.sentiment
        self.method = self.sentiment_config.method
        self.comparison_threshold = self.sentiment_config.comparison_threshold
        self.language = config.language

        # Check pysentimiento language support
        self._pysentimiento_analyzer: (
            AnalyzerForSequenceClassification | None
        ) = None
        if self.method == "pysentimiento" and self.language != "en":
            logger.warning(
                f"pysentimiento only supports English; config language is "
                f"'{self.language}'. Falling back to TextBlob.",
            )
            self.method = "textblob"

        # Initialize sentiment analyzers
        if self.method in ["vader", "hybrid"]:
            self.vader = SentimentIntensityAnalyzer()
            logger.info("Initialized VADER sentiment analyzer")

        if self.method == "pysentimiento":
            self._init_pysentimiento()

    def _init_pysentimiento(self) -> None:
        """Initialize pysentimiento analyzer for sentiment analysis."""
        from pysentimiento import create_analyzer

        self._pysentimiento_analyzer = create_analyzer(
            task="sentiment",
            lang="en",
        )
        logger.info("Initialized pysentimiento sentiment analyzer (English)")

    def analyze_article(self, article: Article) -> SentimentResult:
        """
        Analyze sentiment of a single article.

        Args:
            article: Article object

        Returns:
            SentimentResult object

        """
        try:
            # Analyze title and content/summary
            text_to_analyze = (
                f"{article.title}. {article.summary or article.content}"
            )

            if self.method == "vader":
                sentiment = self._analyze_vader(text_to_analyze)
            elif self.method == "textblob":
                sentiment = self._analyze_textblob(text_to_analyze)
            elif self.method == "pysentimiento":
                sentiment = self._analyze_pysentimiento(text_to_analyze)
            else:  # hybrid
                sentiment = self._analyze_hybrid(text_to_analyze)

            result = SentimentResult(
                article_url=article.url,
                source=article.source,
                polarity=sentiment["polarity"],
                subjectivity=sentiment.get("subjectivity", 0.5),
                compound=sentiment.get("compound", sentiment["polarity"]),
                label=sentiment["label"],
            )

            article.sentiment = result

        except (ValueError, RuntimeError, AttributeError):
            logger.exception(
                f"Error analyzing sentiment for '{article.title}'",
            )
            return SentimentResult(
                article_url=article.url,
                source=article.source,
                polarity=0.0,
                subjectivity=0.5,
                compound=0.0,
                label="neutral",
            )
        else:
            return result

    def _analyze_vader(self, text: str) -> SentimentAnalysisDict:
        """
        Analyze sentiment using VADER.

        Args:
            text: Text to analyze

        Returns:
            Sentiment scores dictionary

        """
        scores = self.vader.polarity_scores(text)

        # Determine label
        compound = scores["compound"]
        if compound >= SENTIMENT_THRESHOLD:
            label = "positive"
        elif compound <= -SENTIMENT_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {
            "polarity": compound,
            "compound": compound,
            "label": label,
            "positive": scores["pos"],
            "negative": scores["neg"],
            "neutral": scores["neu"],
        }

    def _analyze_textblob(self, text: str) -> SentimentAnalysisDict:
        """
        Analyze sentiment using TextBlob.

        Args:
            text: Text to analyze

        Returns:
            Sentiment scores dictionary

        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Determine label
        if polarity > POLARITY_THRESHOLD:
            label = "positive"
        elif polarity < -POLARITY_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "compound": polarity,
            "label": label,
        }

    def _analyze_pysentimiento(self, text: str) -> SentimentAnalysisDict:
        """
        Analyze sentiment using pysentimiento (transformer-based).

        Args:
            text: Text to analyze

        Returns:
            Sentiment scores dictionary

        """
        if self._pysentimiento_analyzer is None:
            msg = "pysentimiento analyzer not initialized"
            raise RuntimeError(msg)

        result = self._pysentimiento_analyzer.predict(text)

        # Map pysentimiento output to our format - pysentimiento
        # returns: output.output (label), output.probas (dict)
        label_map = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
        label = label_map.get(result.output, "neutral")

        # Calculate polarity from probabilities
        # Range: -1 (negative) to +1 (positive)
        probas = result.probas
        polarity = probas.get("POS", 0) - probas.get("NEG", 0)

        return {
            "polarity": polarity,
            "compound": polarity,
            "subjectivity": 0.5,  # pysentimiento doesn't provide subjectivity
            "label": label,
            "probas": probas,  # Include raw probabilities for debugging
        }

    def _analyze_hybrid(self, text: str) -> SentimentAnalysisDict:
        """
        Analyze sentiment using both VADER and TextBlob.

        This is the hybrid method that combines results from
        both analyzers.

        Args:
            text: Text to analyze

        Returns:
            Combined sentiment scores

        """
        vader_result = self._analyze_vader(text)
        textblob_result = self._analyze_textblob(text)

        # Average the polarities
        avg_polarity = (
            vader_result["compound"] + textblob_result["polarity"]
        ) / 2

        # Determine label based on average
        if avg_polarity >= SENTIMENT_THRESHOLD:
            label = "positive"
        elif avg_polarity <= -SENTIMENT_THRESHOLD:
            label = "negative"
        else:
            label = "neutral"

        return {
            "polarity": avg_polarity,
            "compound": avg_polarity,
            "subjectivity": textblob_result["subjectivity"],
            "label": label,
        }

    def _identify_differences(
        self,
        comparison: dict[str, dict[str, float]],
    ) -> list[SentimentDifference]:
        """
        Identify significant sentiment differences between sources.

        Args:
            comparison: Source comparison data

        Returns:
            list of significant differences

        """
        sources = list(comparison.keys())
        avgs = np.array(
            [comparison[s]["avg_sentiment"] for s in sources],
        )
        diffs = np.abs(avgs[:, np.newaxis] - avgs[np.newaxis, :])

        iu = np.triu_indices(len(sources), k=1)
        tri_diffs = diffs[iu]
        tri_i, tri_j = iu[0], iu[1]

        mask = tri_diffs >= self.comparison_threshold
        filtered_diffs = tri_diffs[mask]
        filtered_i = tri_i[mask]
        filtered_j = tri_j[mask]

        order = np.argsort(-filtered_diffs)
        differences: list[SentimentDifference] = []
        for idx in order:
            i, j = int(filtered_i[idx]), int(filtered_j[idx])
            differences.append(
                {
                    "source1": sources[i],
                    "source2": sources[j],
                    "difference": float(filtered_diffs[idx]),
                    "source1_avg": float(avgs[i]),
                    "source2_avg": float(avgs[j]),
                },
            )

        return differences
