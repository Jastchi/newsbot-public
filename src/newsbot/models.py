"""
Data Models.

Defines data classes and database models
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field

from newsbot.constants import TZ


class ViolationType(str, Enum):
    """
    Categories of prompt violations detected by the judge.

    meta_commentary: LLM added explanatory text like "This is..." or
        "The generated..."
    wrong_format: Output does not match expected format (e.g., bullet
        points when none expected)
    off_topic: Output deviates from the requested content
    """

    META_COMMENTARY = "meta_commentary"
    WRONG_FORMAT = "wrong_format"
    OFF_TOPIC = "off_topic"


class JudgeVerdict(BaseModel):
    """
    Structured output from the LLM-as-a-Judge agent.

    Attributes:
        is_valid: Whether the output correctly follows the prompt
            instructions.
        violation_type: Category of violation if is_valid is False:
            meta_commentary, wrong_format, or off_topic.
        is_no_content: Whether the output is a meta-response indicating
            no content (like "no additional points", "nothing to add")
            rather than actual content.

    """

    is_valid: bool = Field(
        description=(
            "True if the output correctly follows the prompt, False otherwise"
        ),
    )
    violation_type: ViolationType | None = Field(
        default=None,
        description=(
            "Category of violation: meta_commentary, wrong_format, or "
            "off_topic"
        ),
    )
    is_no_content: bool = Field(
        default=False,
        description=(
            "True if the output is a meta-response indicating no content "
            "(like 'no additional points', 'nothing interesting to add', "
            "'nothing to add', 'no interesting points', 'no new information') "
            "rather than actual content. These should typically also be "
            "flagged as META_COMMENTARY violations."
        ),
    )


class NameValidationResult(BaseModel):
    """
    Result of name validation against source content.

    Validates that named entities in LLM output appear in sources.

    Attributes:
        is_valid: True if all entities are verified in sources.
        unverified_entities: Entity names not found in sources.
        verified_entities: Entity names confirmed in sources.

    """

    is_valid: bool = Field(
        description="True if all named entities are verified in sources",
    )
    unverified_entities: list[str] = Field(
        default_factory=list,
        description="Entity names not found in any source article",
    )
    verified_entities: list[str] = Field(
        default_factory=list,
        description="Entity names confirmed to exist in source articles",
    )


@dataclass
class SentimentResult:
    """Data class for sentiment analysis results."""

    article_url: str
    source: str
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    compound: float  # -1 to 1 (VADER compound score)
    label: str  # 'positive', 'negative', or 'neutral'


@dataclass
class Article:
    """Data class representing a news article."""

    title: str
    content: str
    source: str
    url: str
    published_date: datetime
    scraped_date: datetime
    summary: str | None = None
    sentiment: SentimentResult | None = None

    def __hash__(self) -> int:
        """Hash based on article URL."""
        return hash(self.url)

    def __eq__(self, other: object) -> bool:
        """Equality based on article URL."""
        if isinstance(other, Article):
            return self.url == other.url
        return False


@dataclass
class Story:
    """Represents a story covered by multiple sources."""

    story_id: str
    title: str  # Representative title
    articles: list[Article]
    sources: list[str]
    article_count: int
    earliest_date: datetime
    latest_date: datetime
    story_summary: str | None = None
    source_additional_points: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Post-initialization to set derived fields."""
        self.article_count = len(self.articles)
        self.sources = list({article.source for article in self.articles})


class SummaryItem(TypedDict):
    """
    Type definition for article summary item.

    Attributes:
        article: The Article object.
        summary: The generated summary text.

    """

    article: Article
    summary: str | None


class SentimentSummary(TypedDict):
    """
    Type definition for sentiment summary per source.

    Attributes:
        avg_sentiment: Average sentiment compound score.
        label: Sentiment label (positive, negative, or neutral).
        article_count: Number of articles analyzed.
        sentiments: List of SentimentResult objects for articles.

    """

    avg_sentiment: float
    label: str
    article_count: int
    sentiments: list[SentimentResult]


class SentimentAnalysisDict(TypedDict, total=False):
    """
    Type definition for sentiment analysis results from various methods.

    Required fields present in all methods: polarity, compound, label.
    Optional fields vary by method: subjectivity, positive, negative,
    neutral (VADER), probas (pysentimiento).

    Attributes:
        polarity: Polarity score (-1 to 1).
        compound: Compound sentiment score (-1 to 1).
        label: Sentiment label (positive, negative, or neutral).
        subjectivity: Subjectivity score (0 to 1).
        positive: Positive sentiment score (VADER).
        negative: Negative sentiment score (VADER).
        neutral: Neutral sentiment score (VADER).
        probas: Probability scores (pysentimiento).

    """

    polarity: float
    compound: float
    label: str
    subjectivity: float
    positive: float
    negative: float
    neutral: float
    probas: dict[str, float]


class SentimentDifference(TypedDict):
    """
    Type definition for sentiment difference between sources.

    Attributes:
        source1: First source name.
        source2: Second source name.
        difference: Absolute difference in sentiment.
        source1_avg: Average sentiment for source1.
        source2_avg: Average sentiment for source2.

    """

    source1: str
    source2: str
    difference: float
    source1_avg: float
    source2_avg: float


class StoryAnalysis(TypedDict):
    """
    Type definition for story analysis data.

    Attributes:
        story: The Story object containing clustered articles.
        source_summaries: Dict mapping sources to article/summary pairs.
        source_sentiments: Dict mapping sources to sentiment summaries.

    """

    story: Story
    source_summaries: dict[str, list[SummaryItem]]
    source_sentiments: dict[str, SentimentSummary]


class NewsSource(TypedDict, total=False):
    """
    Type definition for a news source configuration.

    Attributes:
        name: Source name.
        rss_url: RSS feed URL.
        type: Source type (e.g., 'rss').

    """

    name: str
    rss_url: str
    type: str


class LLMConfig(TypedDict, total=False):
    """
    Type definition for LLM configuration.

    Attributes:
        provider: LLM provider name.
        model: Model identifier.
        base_url: Base URL for the LLM service.
        temperature: Sampling temperature (0-1).
        max_tokens: Maximum tokens to generate.
        judge_model: Model identifier for the judge LLM
        judge_max_retries: Maximum retry attempts when judge detects
            violations.
        judge_enabled: Whether to enable LLM-as-a-judge validation.
        name_validation_enabled: Whether to validate named entities
            against source content.
        name_validation_max_retries: Maximum rewrite attempts when
            unverified entities are found.
        spacy_model: spaCy model for named entity recognition.

    """

    provider: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    judge_model: str
    judge_max_retries: int
    judge_enabled: bool
    name_validation_enabled: bool
    name_validation_max_retries: int
    spacy_model: str


class SummarizationConfig(TypedDict, total=False):
    """
    Type definition for summarization configuration.

    Attributes:
        two_pass_enabled: Whether to use two-pass summarization.
        max_articles_batch: Maximum articles per batch.
        article_order: Order for processing articles.

    """

    two_pass_enabled: bool
    max_articles_batch: int
    article_order: str


class SentimentConfig(TypedDict, total=False):
    """
    Type definition for sentiment analysis configuration.

    Attributes:
        method: Sentiment analysis method.
        comparison_threshold: Threshold for comparing sentiments.

    """

    method: str
    comparison_threshold: float


class StoryClusteringConfig(TypedDict, total=False):
    """
    Type definition for story clustering configuration.

    Attributes:
        top_stories_count: Number of top stories to identify.
        min_sources: Minimum sources for a story.
        similarity_threshold: Threshold for similarity.
        embedding_model: Model for embeddings.
        algorithm: Clustering algorithm to use ('greedy' or 'dbscan').
        dbscan_min_samples: Minimum articles per cluster for DBSCAN.
        sampling_central_count: Number of central articles to select.
        sampling_recent_count: Number of recent articles to select.
        sampling_diverse_count: Number of diverse articles via MMR.
        sampling_similarity_floor: Minimum centroid similarity for MMR.

    """

    top_stories_count: int
    min_sources: int
    similarity_threshold: float
    embedding_model: str
    algorithm: str
    dbscan_min_samples: int
    sampling_central_count: int
    sampling_recent_count: int
    sampling_diverse_count: int
    sampling_similarity_floor: float


class ReportConfig(TypedDict, total=False):
    """
    Type definition for report configuration.

    Attributes:
        format: Report format (html, markdown, text).
        include_summaries: Whether to include article summaries.
        lookback_days: Number of days to look back for articles.

    """

    format: str
    include_summaries: bool
    lookback_days: int


class SchedulerConfig(TypedDict, total=False):
    """
    Type definition for scheduler configuration.

    Attributes:
        weekly_analysis: Configuration for weekly analysis scheduling.

    """

    weekly_analysis: dict[str, int]


class LoggingConfig(TypedDict, total=False):
    """
    Type definition for logging configuration.

    Attributes:
        level: Logging level.
        format: Log format string.

    """

    level: str
    format: str


class DatabaseConfig(TypedDict, total=False):
    """
    Type definition for database configuration.

    Attributes:
        type: Database type.
        path: Database file path.

    """

    type: str
    path: str


class Config(TypedDict, total=False):
    """
    Type definition for the main configuration dictionary.

    Attributes:
        name: Configuration name (country/region).
        country: Country code.
        language: Language code.
        news_sources: List of news sources.
        llm: LLM configuration.
        summarization: Summarization configuration.
        sentiment: Sentiment analysis configuration.
        story_clustering: Story clustering configuration.
        report: Report generation configuration.
        scheduler: Scheduler configuration.
        logging: Logging configuration.
        database: Database configuration.

    """

    name: str
    country: str
    language: str
    news_sources: list[NewsSource]
    llm: LLMConfig
    summarization: SummarizationConfig
    sentiment: SentimentConfig
    story_clustering: StoryClusteringConfig
    report: ReportConfig
    scheduler: SchedulerConfig
    logging: LoggingConfig
    database: DatabaseConfig


class AnalysisData(TypedDict, total=False):
    """
    Type definition for post-analysis hook data.

    Attributes:
        success: Whether analysis succeeded.
        articles_count: Number of articles analyzed.
        stories_count: Number of stories identified.
        duration: Analysis duration in seconds.
        timestamp: When the analysis was run.
        format: Report format (html/txt/md).
        config_name: Display name of the configuration used.
        config_key: Key of the configuration used.
        from_date: Start date for the analysis.
        to_date: End date for the analysis.
        email_receivers_override: Override email receivers list.

    """

    success: bool
    articles_count: int
    stories_count: int
    duration: float
    timestamp: str
    format: str
    config_name: str
    config_key: str
    from_date: datetime
    to_date: datetime
    email_receivers_override: list[str] | None


class PipelineStatusAgents(TypedDict):
    """
    Type definition for pipeline status agents.

    Attributes:
        scraper: Status of scraper agent.
        summarizer: Status of summarizer agent.
        sentiment_analyzer: Status of sentiment analyzer agent.
        report_generator: Status of report generator agent.

    """

    scraper: str
    summarizer: str
    sentiment_analyzer: str
    report_generator: str


class PipelineStatusConfig(TypedDict, total=False):
    """
    Type definition for pipeline status config.

    Attributes:
        country: Country code from config.
        sources: Number of news sources.
        llm_model: LLM model name.

    """

    country: str
    sources: int
    llm_model: str


class PipelineStatus(TypedDict):
    """
    Type definition for pipeline status response.

    Attributes:
        agents: Status of all agents.
        config: Configuration information.

    """

    agents: PipelineStatusAgents
    config: PipelineStatusConfig


@dataclass
class Results:
    """Results of pipeline execution."""

    success: bool = False
    start_time: datetime = field(default_factory=lambda: datetime.now(TZ))
    duration: float = 0.0
    articles_count: int = 0
    stories_count: int = 0
    saved_to_db: int = 0
    top_stories: list[Story] = field(default_factory=list)
    story_analyses: list[StoryAnalysis] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    report_path: str = ""
    end_time: datetime | None = None
