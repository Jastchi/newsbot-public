"""Admin configuration for the newsserver app."""

from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from django.contrib import admin, messages
from django.db import models
from django.forms import ModelMultipleChoiceField
from django.http import HttpRequest

from .models import (
    Article,
    NewsConfig,
    NewsSource,
    Subscriber,
    SubscriberRequest,
    Topic,
)

if TYPE_CHECKING:
    from datetime import datetime




class DisplayMethod(Protocol):
    """Protocol for Django admin display methods."""

    short_description: str


@admin.register(NewsSource)
class NewsSourceAdmin(admin.ModelAdmin):
    """Admin for NewsSource model."""

    list_display = ("name", "url", "type", "config_count", "created_at")
    list_filter = ("type", "created_at")
    search_fields = ("name", "url")
    readonly_fields = ("created_at", "updated_at")

    def config_count(self, obj: NewsSource) -> str:
        """Display count of configs using this source."""
        count = obj.configs.count()
        return f"{count} config{'s' if count != 1 else ''}"

    cast("DisplayMethod", config_count).short_description = "Configs"


class NewsSourceInline(admin.TabularInline):
    """Inline admin for NewsSource in NewsConfig."""

    model = NewsConfig.news_sources.through
    extra = 1
    verbose_name = "News Source"
    verbose_name_plural = "News Sources"

@admin.register(Topic)
class TopicAdmin(admin.ModelAdmin):
    """Admin for Topic model."""

    list_display = ("name", "keywords", "description", "similarity_threshold")
    search_fields = ("name", "keywords", "description")
    readonly_fields = ("created_at", "updated_at")

    def config_count(self, obj: Topic) -> str:
        """Display count of configs using this source."""
        count = obj.configs.count()
        return f"{count} config{'s' if count != 1 else ''}"

    cast("DisplayMethod", config_count).short_description = "Configs"

class TopicInline(admin.TabularInline):
    """Inline admin for NewsSource in NewsConfig."""

    model = NewsConfig.topics.through
    extra = 1
    verbose_name = "Topic"
    verbose_name_plural = "Topics"

@admin.register(NewsConfig)
class NewsConfigAdmin(admin.ModelAdmin):
    """Admin for NewsConfig model."""

    filter_horizontal = ("exclude_articles_from_configs",)

    list_display = (
        "display_name",
        "key",
        "is_active",
        "published_for_subscription",
        "country",
        "language",
        "subscriber_count",
        "article_count",
        "created_at_date",
    )
    list_filter = (
        "is_active",
        "published_for_subscription",
        "country",
        "language",
        "created_at",
    )
    search_fields = ("key", "display_name")
    readonly_fields = ("created_at", "updated_at")
    inlines: ClassVar[list] = [NewsSourceInline, TopicInline]

    fieldsets = (
        (None, {
            "fields": (
                "key",
                "display_name",
                "is_active",
                "published_for_subscription",
            ),
        }),
        ("Basic Settings", {
            "fields": ("country", "language"),
        }),
        ("LLM Configuration", {
            "classes": ("collapse",),
            "fields": (
                "llm_provider", "llm_model", "llm_base_url",
                "llm_temperature", "llm_max_tokens",
                "llm_judge_enabled", "llm_judge_model",
                "llm_judge_max_retries",
                "llm_name_validation_enabled",
                "llm_name_validation_max_retries",
                "llm_spacy_model",
            ),
        }),
        ("Summarization", {
            "classes": ("collapse",),
            "fields": (
                "summarization_two_pass_enabled",
                "summarization_max_articles_batch",
                "summarization_article_order",
            ),
        }),
        ("Sentiment Analysis", {
            "classes": ("collapse",),
            "fields": ("sentiment_method", "sentiment_comparison_threshold"),
        }),
        ("Story Clustering", {
            "classes": ("collapse",),
            "fields": (
                "story_clustering_top_stories_count",
                "story_clustering_min_sources",
                "story_clustering_similarity_threshold",
                "story_clustering_embedding_model",
                "story_clustering_algorithm",
                "story_clustering_dbscan_min_samples",
                "story_clustering_sampling_central_count",
                "story_clustering_sampling_recent_count",
                "story_clustering_sampling_diverse_count",
                "story_clustering_sampling_similarity_floor",
            ),
        }),
        ("Report Settings", {
            "classes": ("collapse",),
            "fields": (
                "report_format",
                "report_include_summaries",
                "report_include_sentiment_charts",
                "report_max_articles_per_source",
                "report_lookback_days",
            ),
        }),
        ("Scheduler", {
            "classes": ("collapse",),
            "fields": (
                "scheduler_daily_scrape_enabled",
                "scheduler_weekly_analysis_enabled",
                "scheduler_weekly_analysis_day_of_week",
                "scheduler_weekly_analysis_hour",
                "scheduler_weekly_analysis_minute",
                "scheduler_weekly_analysis_lookback_days",
            ),
        }),
        ("Filtering", {
            "classes": ("collapse",),
            "fields": ("exclude_articles_from_configs",),
        }),
        ("Logging", {
            "classes": ("collapse",),
            "fields": ("logging_level", "logging_format"),
        }),
        ("Database", {
            "classes": ("collapse",),
            "fields": ("database_url",),
        }),
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
        }),
    )

    def formfield_for_manytomany(
        self,
        db_field: models.ManyToManyField,
        request: HttpRequest,
        **kwargs: object,
    ) -> ModelMultipleChoiceField | None:
        """Exclude self from exclude_articles_from_configs choices."""
        if db_field.name == "exclude_articles_from_configs":
            match = request.resolver_match
            if match and (obj_id := match.kwargs.get("object_id")):
                kwargs["queryset"] = NewsConfig.objects.exclude(pk=obj_id)
        return super().formfield_for_manytomany(db_field, request, **kwargs)

    def created_at_date(self, obj: NewsConfig) -> str:
        """Display created_at date."""
        created_at = cast("datetime", obj.created_at)
        return created_at.strftime("%Y-%m-%d")

    def subscriber_count(self, obj: NewsConfig) -> str:
        """Display count of subscribers for this config."""
        count = obj.subscribers.count()
        return f"{count} subscriber{'s' if count != 1 else ''}"

    def article_count(self, obj: NewsConfig) -> str:
        """Display count of articles for this config."""
        count = obj.articles.count()
        return f"{count} article{'s' if count != 1 else ''}"

    cast(
        "DisplayMethod",
        subscriber_count,
    ).short_description = "Subscribers"
    cast(
        "DisplayMethod",
        article_count,
    ).short_description = "Articles"


@admin.register(Subscriber)
class SubscriberAdmin(admin.ModelAdmin):
    """Admin for Subscriber model."""

    list_display = (
        "full_name",
        "email",
        "is_active",
        "is_staff",
        "is_superuser",
        "config_count",
        "last_login",
    )
    list_editable = ("is_active",)
    list_filter = (
        "is_active",
        "is_staff",
        "is_superuser",
        "configs",
        "created_at",
    )
    search_fields = ("first_name", "last_name", "email")
    filter_horizontal = ("configs",)
    readonly_fields = (
        "created_at",
        "updated_at",
        "last_login",
    )

    def full_name(self, obj: Subscriber) -> str:
        """Display subscriber's full name."""
        return f"{obj.first_name} {obj.last_name}"

    cast("DisplayMethod", full_name).short_description = "Name"

    def config_count(self, obj: Subscriber) -> str:
        """Display count of subscribed configs."""
        count = obj.configs.count()
        return f"{count} config{'s' if count != 1 else ''}"

    cast("DisplayMethod", config_count).short_description = "Configs"


@admin.register(SubscriberRequest)
class SubscriberRequestAdmin(admin.ModelAdmin):
    """Admin for SubscriberRequest (read-only list for processing)."""

    list_display = (
        "email",
        "first_name",
        "last_name",
        "user",
        "created_at",
        "included_in_daily_email_at",
    )
    list_filter = ("created_at", "included_in_daily_email_at")
    search_fields = ("email", "first_name", "last_name")
    readonly_fields = (
        "email",
        "first_name",
        "last_name",
        "user",
        "created_at",
        "included_in_daily_email_at",
    )
    actions: ClassVar[list[str]] = ["accept_requests_create_subscriber"]

    def included(self, obj: SubscriberRequest) -> str:
        """Display whether request was included in daily email."""
        return "Yes" if obj.included_in_daily_email_at else "No"

    cast("DisplayMethod", included).short_description = "Included in digest"

    @admin.action(description="Accept selected (create Subscriber)")
    def accept_requests_create_subscriber(
        self,
        request: HttpRequest,
        queryset: "models.QuerySet[SubscriberRequest]",
    ) -> None:
        """Create a Subscriber for each request, then remove it."""
        created = 0
        skipped = 0
        for req in queryset:
            if Subscriber.objects.filter(email__iexact=req.email).exists():
                skipped += 1
                continue
            first = (req.first_name or "").strip() or "—"
            last = (req.last_name or "").strip() or "—"
            Subscriber.objects.create(
                email=req.email,
                first_name=first,
                last_name=last,
                is_active=True,
            )
            created += 1
            req.delete()
        if created:
            self.message_user(
                request,
                f"Created {created} subscriber(s). "
                "They can now use “My subscriptions”.",
                level=messages.SUCCESS,
            )
        if skipped:
            self.message_user(
                request,
                f"Skipped {skipped} request(s) (email already a subscriber).",
                level=messages.WARNING,
            )
        if not created and not skipped:
            self.message_user(
                request,
                "No requests selected or all already accepted.",
                level=messages.WARNING,
            )


@admin.register(Article)
class ArticleAdmin(admin.ModelAdmin):
    """Admin for Article model."""

    list_display = (
        "title",
        "source",
        "config_display",
        "published_date",
        "scraped_date",
        "sentiment_label",
    )
    list_filter = (
        "config",
        "source",
        "scraped_date",
        "sentiment_label",
    )
    search_fields = (
        "title", "source", "url", "config__key", "config__display_name",
    )
    readonly_fields = (
        "config",
        "config_file",
        "title",
        "content",
        "summary",
        "source",
        "url",
        "published_date",
        "scraped_date",
        "sentiment_score",
        "sentiment_label",
    )
    ordering = ("-scraped_date",)
    raw_id_fields = ("config",)

    def config_display(self, obj: Article) -> str:
        """Display config key or config_file as fallback."""
        if obj.config:
            return obj.config.key
        return obj.config_file or "—"

    cast("DisplayMethod", config_display).short_description = "Config"
