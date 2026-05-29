"""Admin configuration for the newsserver app."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Protocol, cast

from django import forms
from django.contrib import admin, messages
from django.db import models
from django.forms import ModelMultipleChoiceField
from django.http import HttpRequest
from django.urls import reverse
from django.utils.html import format_html

from .models import (
    Article,
    ConfigSuggestion,
    NewsConfig,
    NewsSource,
    Subscriber,
    SubscriberRequest,
    Topic,
)
from .widgets import OptionalColorWidget

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
    readonly_fields = ("source_type",)

    @admin.display(description="Type")
    def source_type(self, obj: models.Model) -> str:
        """Show the type of the linked NewsSource."""
        try:
            source = getattr(obj, "newssource", None)
            return source.get_type_display() if source else "—"
        except Exception:
            return "—"

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

_COLOR_INPUT_STYLE = (
    "width:60px; height:36px; padding:2px; cursor:pointer;"
)

_NEWS_CONFIG_FORM_FIELDS = (
    "key",
    "display_name",
    "country",
    "language",
    "is_active",
    "published_for_subscription",
    "llm_provider",
    "llm_model",
    "llm_base_url",
    "llm_temperature",
    "llm_max_tokens",
    "llm_judge_enabled",
    "llm_judge_model",
    "llm_judge_max_retries",
    "llm_name_validation_enabled",
    "llm_name_validation_max_retries",
    "llm_spacy_model",
    "summarization_two_pass_enabled",
    "summarization_max_articles_batch",
    "summarization_article_order",
    "summarization_explain_for_outsiders",
    "sentiment_method",
    "sentiment_comparison_threshold",
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
    "report_format",
    "report_include_summaries",
    "report_include_sentiment_charts",
    "report_max_articles_per_source",
    "report_lookback_days",
    "scheduler_daily_scrape_enabled",
    "scheduler_weekly_analysis_enabled",
    "scheduler_weekly_analysis_day_of_week",
    "scheduler_weekly_analysis_hour",
    "scheduler_weekly_analysis_minute",
    "scheduler_weekly_analysis_lookback_days",
    "exclude_articles_from_configs",
    "hero_color_primary",
    "hero_color_secondary",
    "hero_color_middle",
    "database_url",
)


class NewsConfigAdminForm(forms.ModelForm):
    """ModelForm for NewsConfig with native color pickers."""

    class Meta:
        """NewsConfig admin form options."""

        model = NewsConfig
        fields = _NEWS_CONFIG_FORM_FIELDS
        widgets: ClassVar[dict] = {
            "hero_color_primary": forms.TextInput(
                attrs={"type": "color", "style": _COLOR_INPUT_STYLE},
            ),
            "hero_color_secondary": forms.TextInput(
                attrs={"type": "color", "style": _COLOR_INPUT_STYLE},
            ),
            "hero_color_middle": OptionalColorWidget(),
        }


@admin.register(NewsConfig)
class NewsConfigAdmin(admin.ModelAdmin):
    """Admin for NewsConfig model."""

    form = NewsConfigAdminForm
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
    readonly_fields = ("created_at", "updated_at", "email_preview_panel")
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
                "summarization_explain_for_outsiders",
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
        ("Email Preview", {
            "fields": (
                "hero_color_primary",
                "hero_color_secondary",
                "hero_color_middle",
                "email_preview_panel",
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
        ("Timestamps", {
            "fields": ("created_at", "updated_at"),
        }),
    )

    class Media:
        """Assets for conditional field visibility in the admin form."""

        js = ("newsserver/admin/newsconfig_admin.js",)

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

    def email_preview_panel(self, obj: NewsConfig) -> str:
        """Render a live iframe preview of the email template."""
        if not obj.pk:
            return "Save the config first to enable preview."
        preview_url = reverse("newsserver:email_preview", args=[obj.key])
        iframe_style = (
            "width:100%; height:700px; border:1px solid #ddd; "
            "border-radius:4px;"
        )
        return format_html(
            '<iframe id="email-preview-iframe" src="{}" style="{}">'
            "</iframe>",
            preview_url,
            iframe_style,
        )

    cast("DisplayMethod", email_preview_panel).short_description = (
        "Live Preview"
    )

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
    actions: Sequence[str] = ("accept_requests_create_subscriber",)

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


@admin.register(ConfigSuggestion)
class ConfigSuggestionAdmin(admin.ModelAdmin):
    """Admin for ConfigSuggestion (read-only list for review)."""

    preview_length = 80

    list_display = (
        "name",
        "email",
        "submitted_by",
        "sources_preview",
        "note_preview",
        "created_at",
    )
    list_filter = ("created_at",)
    search_fields = ("name", "email", "sources", "note")
    readonly_fields = (
        "name",
        "sources",
        "note",
        "email",
        "submitted_by",
        "created_at",
    )
    ordering = ("-created_at",)

    @admin.display(description="Sources")
    def sources_preview(self, obj: ConfigSuggestion) -> str:
        """Show first preview_length chars of sources."""
        length = self.preview_length
        return (
            obj.sources[:length] + "…"
            if len(obj.sources) > length
            else obj.sources
        )

    @admin.display(description="Note")
    def note_preview(self, obj: ConfigSuggestion) -> str:
        """Show first preview_length chars of note."""
        if not obj.note:
            return "—"
        length = self.preview_length
        return (
            obj.note[:length] + "…"
            if len(obj.note) > length
            else obj.note
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
