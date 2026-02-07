"""Utility models for the newsbot application."""

from pydantic import BaseModel, ConfigDict, Field


class NewsSourceModel(BaseModel):
    """Pydantic model for a news source configuration."""

    name: str
    rss_url: str
    type: str = "rss"


class TopicModel(BaseModel):
    """Pydantic model for a topic configuration."""

    name: str
    keywords: list[str] = Field(default_factory=list)
    description: str = ""
    similarity_threshold: float = 0.6


class LLMConfigModel(BaseModel):
    """Pydantic model for LLM configuration."""

    provider: str = "ollama"
    model: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 16384
    judge_enabled: bool = False
    judge_model: str = ""
    judge_max_retries: int = 2
    name_validation_enabled: bool = False
    name_validation_max_retries: int = 2
    spacy_model: str = "en_core_web_sm"


class SummarizationConfigModel(BaseModel):
    """Pydantic model for summarization configuration."""

    two_pass_enabled: bool = True
    max_articles_batch: int = 1
    article_order: str = "chronological"


class SentimentConfigModel(BaseModel):
    """Pydantic model for sentiment analysis configuration."""

    method: str = "pysentimiento"
    comparison_threshold: float = 0.3


class StoryClusteringConfigModel(BaseModel):
    """Pydantic model for story clustering configuration."""

    top_stories_count: int = 5
    min_sources: int = 2
    similarity_threshold: float = 0.7
    embedding_model: str = "Xenova/all-mpnet-base-v2"
    algorithm: str = "dbscan"
    dbscan_min_samples: int = 2
    sampling_central_count: int = 4
    sampling_recent_count: int = 6
    sampling_diverse_count: int = 4
    sampling_similarity_floor: float = 0.4


class ReportConfigModel(BaseModel):
    """Pydantic model for report configuration."""

    format: str = "html"
    include_summaries: bool = True
    include_sentiment_charts: bool = True
    max_articles_per_source: int = 10
    lookback_days: int = 7


class DailyScrapeConfigModel(BaseModel):
    """Pydantic model for daily scrape scheduler configuration."""

    enabled: bool = True
    hour: int = 2
    minute: int = 0


class WeeklyAnalysisConfigModel(BaseModel):
    """Pydantic model for weekly analysis scheduler configuration."""

    enabled: bool = True
    day_of_week: str = "mon"
    hour: int = 9
    minute: int = 0
    lookback_days: int = 7


class SchedulerConfigModel(BaseModel):
    """Pydantic model for scheduler configuration."""

    daily_scrape: DailyScrapeConfigModel = Field(
        default_factory=DailyScrapeConfigModel,
    )
    weekly_analysis: WeeklyAnalysisConfigModel = Field(
        default_factory=WeeklyAnalysisConfigModel,
    )


class LoggingConfigModel(BaseModel):
    """Pydantic model for logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s [%(levelname)s - %(name)s] %(message)s"


class DatabaseConfigModel(BaseModel):
    """Pydantic model for database configuration."""

    url: str = "${DATABASE_URL}"


class ConfigModel(BaseModel):
    """
    Pydantic model for the main configuration.

    Provides full type safety and attribute access for all config
    fields. Also supports dictionary-style access for backward
    compatibility.

    Example:
        config = ConfigModel(name="Technology", country="US")
        assert config.name == "Technology"
        assert config["country"] == "IL"  # Also works
        assert config.llm.provider == "ollama"

    """

    name: str = ""
    country: str = "US"
    language: str = "en"
    is_active: bool = True
    news_sources: list[NewsSourceModel] = Field(default_factory=list)
    topics: list[TopicModel] = Field(default_factory=list)
    llm: LLMConfigModel = Field(default_factory=LLMConfigModel)
    summarization: SummarizationConfigModel = Field(
        default_factory=SummarizationConfigModel,
    )
    sentiment: SentimentConfigModel = Field(
        default_factory=SentimentConfigModel,
    )
    story_clustering: StoryClusteringConfigModel = Field(
        default_factory=StoryClusteringConfigModel,
    )
    report: ReportConfigModel = Field(default_factory=ReportConfigModel)
    scheduler: SchedulerConfigModel = Field(
        default_factory=SchedulerConfigModel,
    )
    logging: LoggingConfigModel = Field(default_factory=LoggingConfigModel)
    database: DatabaseConfigModel = Field(default_factory=DatabaseConfigModel)
    exclude_articles_from_config_keys: list[str] = Field(default_factory=list)

    model_config = ConfigDict(
        populate_by_name=True,
        extra="allow",
    )
