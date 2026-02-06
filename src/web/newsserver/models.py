"""Data models for the newsserver app."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from django.core.exceptions import ValidationError
from django.db import models

from newsbot.constants import DAILY_SCRAPE_HOUR, DAILY_SCRAPE_MINUTE
from utilities import models as config_models

if TYPE_CHECKING:
    from django.db.models.manager import Manager
    from django_stubs_ext.db.models.manager import RelatedManager


class BaseModel(models.Model):
    """Base model for all newsserver models with app_label set."""

    class Meta:
        """Meta options for BaseModel."""

        abstract = True
        app_label = "newsserver"


class NewsSource(BaseModel):
    """
    Model representing a news source (RSS feed).

    A news source can belong to multiple configurations.
    """

    class SourceType(models.TextChoices):
        """Source type choices."""

        RSS = "rss", "RSS"

    name = models.CharField(
        max_length=200,
        help_text="Source name (e.g., 'Example News Source')",
    )
    url = models.URLField(
        max_length=500,
        unique=True,
        help_text="Source URL",
    )
    type = models.CharField(
        max_length=50,
        choices=SourceType.choices,
        default=SourceType.RSS,
        help_text="Source type",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    if TYPE_CHECKING:
        objects: Manager[NewsSource]
        configs: RelatedManager[NewsConfig]

    class Meta:
        """Meta options for NewsSource."""

        verbose_name_plural = "News sources"
        ordering: ClassVar[list[str]] = ["name"]

    def __str__(self) -> str:
        """Represent the news source as a string."""
        return f"{self.name} - {self.url}"


class Topic(BaseModel):
    """
    Model representing a topic for filtering news articles.

    Topics can be linked to news configurations to filter articles
    during scraping. Articles are matched using keyword matching
    and/or semantic similarity.
    """

    name = models.CharField(
        max_length=200,
        unique=True,
        help_text="Topic name (e.g., 'Artificial Intelligence')",
    )
    keywords = models.TextField(
        blank=True,
        help_text=(
            "Comma-separated keywords for fast matching "
            "(e.g., 'AI, machine learning, GPT, LLM')"
        ),
    )
    description = models.TextField(
        blank=True,
        help_text=(
            "Description of the topic for semantic matching "
            "(e.g., 'Articles about artificial intelligence, "
            "machine learning, and neural networks')"
        ),
    )
    similarity_threshold = models.FloatField(
        default=0.6,
        help_text=(
            "Minimum cosine similarity score (0-1) for semantic matching. "
            "Higher values = stricter matching."
        ),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    if TYPE_CHECKING:
        objects: Manager[Topic]
        configs: RelatedManager[NewsConfig]

    class Meta:
        """Meta options for Topic."""

        verbose_name_plural = "Topics"
        ordering: ClassVar[list[str]] = ["name"]

    def __str__(self) -> str:
        """Represent the topic as a string."""
        return self.name

    def get_keywords_list(self) -> list[str]:
        """Return keywords as a list of lowercase strings."""
        if not self.keywords:
            return []
        return [
            kw.strip().lower() for kw in self.keywords.split(",") if kw.strip()
        ]


class NewsConfig(BaseModel):
    """Model representing a news configuration."""

    class LLMProvider(models.TextChoices):
        """LLM provider choices."""

        OLLAMA = "ollama", "Ollama"
        GEMINI = "gemini", "Gemini"

    class ArticleOrder(models.TextChoices):
        """Article order choices."""

        CHRONOLOGICAL = "chronological", "Chronological"

    class ClusteringAlgorithm(models.TextChoices):
        """Story clustering algorithm choices."""

        DBSCAN = "dbscan", "DBSCAN"
        GREEDY = "greedy", "Greedy"

    class DayOfWeek(models.TextChoices):
        """Day of week choices."""

        MONDAY = "mon", "Monday"
        TUESDAY = "tue", "Tuesday"
        WEDNESDAY = "wed", "Wednesday"
        THURSDAY = "thu", "Thursday"
        FRIDAY = "fri", "Friday"
        SATURDAY = "sat", "Saturday"
        SUNDAY = "sun", "Sunday"

    key = models.SlugField(
        max_length=100,
        unique=True,
        help_text=(
            "A unique identifier for the news configuration "
            "(e.g., 'technology', 'world')"
        ),
    )
    display_name = models.CharField(
        max_length=200,
        help_text="Human-readable display name for the news configuration",
    )
    news_sources = models.ManyToManyField(
        NewsSource,
        blank=True,
        related_name="configs",
        help_text="News sources for this configuration",
    )
    topics = models.ManyToManyField(
        Topic,
        blank=True,
        related_name="configs",
        help_text=(
            "Topics to filter articles by. If empty, all articles are kept."
        ),
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Basic configuration fields
    country = models.CharField(
        max_length=10,
        default="US",
        help_text="Country code",
    )
    language = models.CharField(
        max_length=10,
        default="en",
        help_text="Language code",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Enable/disable this configuration",
    )

    # LLM Configuration fields
    llm_provider = models.CharField(
        max_length=50,
        choices=LLMProvider.choices,
        default=LLMProvider.OLLAMA,
        help_text="LLM provider name",
    )
    llm_model = models.CharField(
        max_length=200,
        default="",
        help_text="LLM model identifier",
    )
    llm_base_url = models.CharField(
        max_length=500,
        default="http://localhost:11434",
        blank=True,
        help_text="Base URL for the LLM service (unused if not ollama)",
    )
    llm_temperature = models.FloatField(
        default=0.0,
        help_text="Sampling temperature (0-1)",
    )
    llm_max_tokens = models.IntegerField(
        default=16384,
        help_text="Maximum tokens to generate",
    )
    llm_judge_enabled = models.BooleanField(
        default=False,
        help_text="Whether to enable LLM-as-a-judge validation",
    )
    llm_judge_model = models.CharField(
        max_length=200,
        default="",
        help_text="Model identifier for the judge LLM",
    )
    llm_judge_max_retries = models.IntegerField(
        default=2,
        help_text="Maximum retry attempts when judge detects violations",
    )
    llm_name_validation_enabled = models.BooleanField(
        default=False,
        help_text="Whether to validate named entities against source content",
    )
    llm_name_validation_max_retries = models.IntegerField(
        default=2,
        help_text=(
            "Maximum rewrite attempts when unverified entities are found"
        ),
    )
    llm_spacy_model = models.CharField(
        max_length=200,
        default="en_core_web_sm",
        help_text="spaCy model for named entity recognition",
    )

    # Summarization Configuration fields
    summarization_two_pass_enabled = models.BooleanField(
        default=True,
        help_text="Whether to use two-pass summarization",
    )
    summarization_max_articles_batch = models.IntegerField(
        default=1,
        help_text="Maximum articles per batch",
    )

    summarization_article_order = models.CharField(
        max_length=50,
        choices=ArticleOrder.choices,
        default=ArticleOrder.CHRONOLOGICAL,
        help_text="Order for processing articles",
    )

    # Sentiment Configuration fields
    class SentimentMethod(models.TextChoices):
        """Sentiment analysis method choices."""

        VADER = "vader", "VADER"
        TEXTBLOB = "textblob", "TextBlob"
        PYSENTIMIENTO = "pysentimiento", "PySentimiento"
        HYBRID = "hybrid", "Hybrid (VADER + TextBlob)"

    sentiment_method = models.CharField(
        max_length=50,
        choices=SentimentMethod.choices,
        default=SentimentMethod.PYSENTIMIENTO,
        help_text="Sentiment analysis method",
    )
    sentiment_comparison_threshold = models.FloatField(
        default=0.3,
        help_text="Threshold for comparing sentiments",
    )

    # Story Clustering Configuration fields
    story_clustering_top_stories_count = models.IntegerField(
        default=5,
        help_text="Number of top stories to identify",
    )
    story_clustering_min_sources = models.IntegerField(
        default=2,
        help_text="Minimum sources for a story",
    )
    story_clustering_similarity_threshold = models.FloatField(
        default=0.7,
        help_text="Threshold for similarity",
    )
    story_clustering_embedding_model = models.CharField(
        max_length=200,
        default="Xenova/all-mpnet-base-v2",
        help_text="Model for embeddings",
    )
    story_clustering_algorithm = models.CharField(
        max_length=50,
        choices=ClusteringAlgorithm.choices,
        default=ClusteringAlgorithm.DBSCAN,
        help_text="Clustering algorithm to use",
    )
    story_clustering_dbscan_min_samples = models.IntegerField(
        default=2,
        help_text="Minimum number of neighbors to be a core point",
    )
    story_clustering_sampling_central_count = models.IntegerField(
        default=4,
        help_text="Number of central articles to select",
    )
    story_clustering_sampling_recent_count = models.IntegerField(
        default=6,
        help_text="Number of recent articles to select",
    )
    story_clustering_sampling_diverse_count = models.IntegerField(
        default=4,
        help_text="Number of diverse articles via MMR",
    )
    story_clustering_sampling_similarity_floor = models.FloatField(
        default=0.4,
        help_text="Minimum centroid similarity for MMR",
    )

    # Report Configuration fields
    class ReportFormat(models.TextChoices):
        """Report format choices."""

        HTML = "html", "HTML"
        MARKDOWN = "markdown", "Markdown"
        TEXT = "text", "Text"

    report_format = models.CharField(
        max_length=20,
        choices=ReportFormat.choices,
        default=ReportFormat.HTML,
        help_text="Report format",
    )
    report_include_summaries = models.BooleanField(
        default=True,
        help_text="Whether to include article summaries",
    )
    report_include_sentiment_charts = models.BooleanField(
        default=True,
        help_text="Whether to include sentiment charts",
    )
    report_max_articles_per_source = models.IntegerField(
        default=10,
        help_text="Maximum articles per source in report",
    )
    report_lookback_days = models.IntegerField(
        default=7,
        help_text="Number of days to look back for articles",
    )

    # Scheduler Configuration fields
    scheduler_daily_scrape_enabled = models.BooleanField(
        default=True,
        help_text="Whether daily scrape is enabled",
    )
    scheduler_weekly_analysis_enabled = models.BooleanField(
        default=True,
        help_text="Whether weekly analysis is enabled",
    )
    scheduler_weekly_analysis_day_of_week = models.CharField(
        max_length=10,
        choices=DayOfWeek.choices,
        default=DayOfWeek.MONDAY,
        help_text="Day of week for weekly analysis",
    )
    scheduler_weekly_analysis_hour = models.IntegerField(
        default=9,
        help_text="Hour for weekly analysis (0-23)",
    )
    scheduler_weekly_analysis_minute = models.IntegerField(
        default=0,
        help_text="Minute for weekly analysis (0-59)",
    )
    scheduler_weekly_analysis_lookback_days = models.IntegerField(
        default=7,
        help_text="Number of days to look back for weekly analysis",
    )

    # Exclude articles from other configs (scrape skips their URLs)
    exclude_articles_from_configs = models.ManyToManyField(
        "self",
        symmetrical=False,
        blank=True,
        related_name="configs_excluding_my_articles",
        help_text=(
            "Optional. Exclude articles whose URL exists in any of these "
            "configs' articles."
        ),
    )

    # Logging Configuration fields
    logging_level = models.CharField(
        max_length=20,
        default="INFO",
        help_text="Logging level",
    )
    logging_format = models.CharField(
        max_length=500,
        default="%(asctime)s [%(levelname)s - %(name)s] %(message)s",
        help_text="Log format string",
    )

    # Database Configuration fields
    database_url = models.CharField(
        max_length=500,
        default="${DATABASE_URL}",
        help_text="Database URL (supports environment variable expansion)",
    )

    if TYPE_CHECKING:
        subscribers: Manager[Subscriber]
        articles: Manager[Article]

    def __str__(self) -> str:
        """Represent the news configuration as a string."""
        return f"{self.display_name} - {self.key}"

    def clean(self) -> None:
        """Validate that a config does not exclude itself."""
        super().clean()
        if self.pk and self.exclude_articles_from_configs.filter(
            pk=self.pk,
        ).exists():
            raise ValidationError(
                {
                    "exclude_articles_from_configs": (
                        "A config cannot exclude articles from itself."
                    ),
                },
            )

    def to_config_dict(self) -> config_models.ConfigModel:
        """
        Convert NewsConfig to ConfigModel with full type safety.

        Builds nested structure from flat fields and includes related
        NewsSource objects. Returns a ConfigModel (Pydantic) that
        provides full type safety and attribute access.

        Returns:
            ConfigModel instance with nested config structure

        """
        # Build news_sources list from related NewsSource objects
        news_sources_list = [
            config_models.NewsSourceModel(
                name=source.name,
                rss_url=source.url,
                type=source.type,
            )
            for source in self.news_sources.all()
        ]

        # Build topics list from related Topic objects
        topics_list = [
            config_models.TopicModel(
                name=topic.name,
                keywords=topic.get_keywords_list(),
                description=topic.description,
                similarity_threshold=topic.similarity_threshold,
            )
            for topic in self.topics.all()
        ]

        # Build and return ConfigModel
        return config_models.ConfigModel(
            name=self.display_name,  # Use display_name as name
            country=self.country,
            language=self.language,
            is_active=self.is_active,
            news_sources=news_sources_list,
            topics=topics_list,
            llm=config_models.LLMConfigModel(
                provider=self.llm_provider,
                model=self.llm_model,
                base_url=self.llm_base_url,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                judge_enabled=self.llm_judge_enabled,
                judge_model=self.llm_judge_model,
                judge_max_retries=self.llm_judge_max_retries,
                name_validation_enabled=self.llm_name_validation_enabled,
                name_validation_max_retries=self.llm_name_validation_max_retries,
                spacy_model=self.llm_spacy_model,
            ),
            summarization=config_models.SummarizationConfigModel(
                two_pass_enabled=self.summarization_two_pass_enabled,
                max_articles_batch=self.summarization_max_articles_batch,
                article_order=self.summarization_article_order,
            ),
            sentiment=config_models.SentimentConfigModel(
                method=self.sentiment_method,
                comparison_threshold=self.sentiment_comparison_threshold,
            ),
            story_clustering=config_models.StoryClusteringConfigModel(
                top_stories_count=self.story_clustering_top_stories_count,
                min_sources=self.story_clustering_min_sources,
                similarity_threshold=self.story_clustering_similarity_threshold,
                embedding_model=self.story_clustering_embedding_model,
                algorithm=self.story_clustering_algorithm,
                dbscan_min_samples=self.story_clustering_dbscan_min_samples,
                sampling_central_count=self.story_clustering_sampling_central_count,
                sampling_recent_count=self.story_clustering_sampling_recent_count,
                sampling_diverse_count=self.story_clustering_sampling_diverse_count,
                sampling_similarity_floor=self.story_clustering_sampling_similarity_floor,
            ),
            report=config_models.ReportConfigModel(
                format=self.report_format,
                include_summaries=self.report_include_summaries,
                include_sentiment_charts=self.report_include_sentiment_charts,
                max_articles_per_source=self.report_max_articles_per_source,
                lookback_days=self.report_lookback_days,
            ),
            scheduler=config_models.SchedulerConfigModel(
                daily_scrape=config_models.DailyScrapeConfigModel(
                    enabled=self.scheduler_daily_scrape_enabled,
                    hour=DAILY_SCRAPE_HOUR, # for deprecated command
                    minute=DAILY_SCRAPE_MINUTE, # newsbot schedule
                ),
                weekly_analysis=config_models.WeeklyAnalysisConfigModel(
                    enabled=self.scheduler_weekly_analysis_enabled,
                    day_of_week=self.scheduler_weekly_analysis_day_of_week,
                    hour=self.scheduler_weekly_analysis_hour,
                    minute=self.scheduler_weekly_analysis_minute,
                    lookback_days=self.scheduler_weekly_analysis_lookback_days,
                ),
            ),
            logging=config_models.LoggingConfigModel(
                level=self.logging_level,
                format=self.logging_format,
            ),
            database=config_models.DatabaseConfigModel(
                url=self.database_url,
            ),
            exclude_articles_from_config_keys=list(
                self.exclude_articles_from_configs.values_list(
                    "key", flat=True,
                ),
            ),
        )


class Subscriber(BaseModel):
    """Model representing a subscriber to news configurations."""

    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField(unique=True)
    is_active = models.BooleanField(default=True)
    configs = models.ManyToManyField(
        NewsConfig,
        blank=True,
        related_name="subscribers",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self) -> str:
        """Represent the subscriber as a string."""
        return f"{self.first_name} {self.last_name} <{self.email}>"

    def subscribed_config_keys(self) -> list[str]:
        """
        Get subscribed configuration keys.

        Returns a list of keys for the news configurations the
        subscriber is subscribed to.
        """
        return list(self.configs.values_list("key", flat=True))


class ScrapeSummary(BaseModel):
    """Model for storing summary of daily scrape runs."""

    config = models.ForeignKey(
        NewsConfig,
        on_delete=models.CASCADE,
        related_name="scrape_summaries",
        null=True,
        blank=True,
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=False)
    duration = models.FloatField(help_text="Duration in seconds")

    articles_scraped = models.IntegerField(
        default=0,
        help_text="Number of articles scraped",
    )
    articles_saved = models.IntegerField(
        default=0,
        help_text="Number of articles saved to DB",
    )

    error_count = models.IntegerField(default=0)
    errors = models.TextField(blank=True, help_text="List of errors")

    if TYPE_CHECKING:
        objects: Manager[ScrapeSummary]

    class Meta:
        """Meta options for ScrapeSummary."""

        verbose_name_plural = "Scrape summaries"
        ordering: ClassVar[list[str]] = ["-timestamp"]

    def __str__(self) -> str:
        """Represent the scrape summary as a string."""
        return f"Scrape - {self.config} - {self.timestamp}"


class AnalysisSummary(BaseModel):
    """Model for storing summary of weekly analysis runs."""

    config = models.ForeignKey(
        NewsConfig,
        on_delete=models.CASCADE,
        related_name="analysis_summaries",
        null=True,
        blank=True,
    )
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=False)
    duration = models.FloatField(help_text="Duration in seconds")

    articles_analyzed = models.IntegerField(
        default=0,
        help_text="Number of articles analyzed",
    )
    stories_identified = models.IntegerField(
        default=0,
        help_text="Number of stories identified",
    )
    top_stories = models.TextField(
        blank=True,
        help_text="JSON or text summary of top stories",
    )

    error_count = models.IntegerField(default=0)
    errors = models.TextField(blank=True, help_text="List of errors")

    if TYPE_CHECKING:
        objects: Manager[AnalysisSummary]

    class Meta:
        """Meta options for AnalysisSummary."""

        verbose_name_plural = "Analysis summaries"
        ordering: ClassVar[list[str]] = ["-timestamp"]

    def __str__(self) -> str:
        """Represent the analysis summary as a string."""
        return f"Analysis - {self.config} - {self.timestamp}"


class Article(BaseModel):
    """Model representing a news article."""

    config = models.ForeignKey(
        NewsConfig,
        on_delete=models.CASCADE,
        related_name="articles",
        null=True,
        blank=True,
        help_text="News configuration this article belongs to",
    )
    # Kept temporarily for migration
    # will be removed after data migration
    config_file = models.CharField(
        max_length=100,
        db_index=True,
        blank=True,
        help_text="DEPRECATED: Configuration file name (use config FK)",
    )
    title = models.CharField(
        max_length=500,
        help_text="Article title",
    )
    content = models.TextField(
        blank=True,
        help_text="Full article content",
    )
    summary = models.TextField(
        blank=True,
        help_text="Generated article summary",
    )
    source = models.CharField(
        max_length=100,
        help_text="News source name",
    )
    url = models.CharField(
        max_length=1000,
        db_index=True,
        help_text="Article URL",
    )
    published_date = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Original publication date",
    )
    scraped_date = models.DateTimeField(
        auto_now_add=True,
        db_index=True,
        help_text="Date when article was scraped",
    )
    sentiment_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Sentiment analysis score (compound score)",
    )
    sentiment_label = models.CharField(
        max_length=20,
        blank=True,
        help_text="Sentiment label (positive, negative, neutral)",
    )

    if TYPE_CHECKING:
        objects: Manager[Article]
        config_id: int | None

    class Meta:
        """Meta options for Article."""

        verbose_name_plural = "Articles"
        ordering: ClassVar[list[str]] = ["-scraped_date"]
        indexes: ClassVar[list[models.Index]] = [
            models.Index(fields=["config", "scraped_date"]),
        ]
        constraints: ClassVar[list[models.UniqueConstraint]] = [
            models.UniqueConstraint(
                fields=["url", "config"],
                name="unique_url_per_config_fk",
            ),
        ]

    def __str__(self) -> str:
        """Represent the article as a string."""
        title_preview_length = 50
        title_preview = (
            self.title[:title_preview_length] + "..."
            if len(self.title) > title_preview_length
            else self.title
        )
        config_key = (
            self.config.key if self.config else self.config_file or "unknown"
        )
        return (
            f"Article(config='{config_key}', "
            f"title='{title_preview}', source='{self.source}')"
        )
