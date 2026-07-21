"""Views for displaying NewsBot reports."""

import hmac
import json
import logging
import secrets
from collections import deque
from collections.abc import Generator
from datetime import datetime, time, timedelta
from pathlib import Path
from time import sleep
from typing import cast

from django.conf import settings
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import (
    FileResponse,
    Http404,
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.views.generic import TemplateView

from after_analysis.email._tokens import _generate_unsubscribe_token
from newsbot.agents.report_agent import ReportGeneratorAgent
from newsbot.color_utils import derive_color_palette
from newsbot.constants import (
    DAY_NAME_TO_PYTHON_WEEKDAY,
    PYTHON_WEEKDAY_TO_DAY_NAME,
    TZ,
)

from .adapters import SESSION_KEY_SUBSCRIPTION_REQUEST_FROM_SOCIAL
from .auth_helpers import (
    notify_admin_config_suggestion,
    notify_admin_subscriber_request,
)
from .models import (
    AnalysisSummary,
    ConfigSuggestion,
    NewsConfig,
    ScrapeSummary,
    Subscriber,
    SubscriberRequest,
)
from .palettes import published_palettes
from .services.config_service import ConfigService
from .services.log_service import LogService
from .services.report_service import ReportService
from .site_urls import site_origin
from .utils import get_date_range, parse_date_or_default

logger = logging.getLogger(__name__)

_TEXT_PLAIN_UTF8 = "text/plain; charset=utf-8"


def robots_txt(request: HttpRequest) -> HttpResponse:
    """Serve robots.txt, pointing crawlers at the sitemap."""
    lines = [
        "User-agent: *",
        "Allow: /",
        "Content-Signal: search=yes, ai-input=yes, ai-train=no",
        f"Sitemap: {site_origin(request)}/sitemap.xml",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain")


def _contact_url_and_label() -> tuple[str, str]:
    """Return (contact_url, label) for privacy/security requests."""
    email = settings.EMAIL_ADMIN_NOTIFICATION_TO
    if email:
        return f"mailto:{email}", email
    return (
        "https://github.com/Jastchi/newsbot-public/issues",
        "a GitHub issue",
    )


def security_txt(request: HttpRequest) -> HttpResponse:
    """Serve /.well-known/security.txt per RFC 9116."""
    contact, _ = _contact_url_and_label()
    expires = (timezone.now() + timedelta(days=365)).strftime(
        "%Y-%m-%dT%H:%M:%SZ",
    )
    lines = [
        f"Contact: {contact}",
        f"Expires: {expires}",
        "Preferred-Languages: en",
        f"Canonical: {site_origin(request)}/.well-known/security.txt",
    ]
    return HttpResponse("\n".join(lines), content_type=_TEXT_PLAIN_UTF8)


def _app_urls() -> tuple[str, str]:
    """Return the app's (login_url, home_url) for any host."""
    if settings.APP_BASE_URL:
        return (
            f"{settings.APP_BASE_URL}/accounts/login/",
            f"{settings.APP_BASE_URL}/",
        )
    # On a marketing host the active urlconf is marketing_urls (set by
    # MarketingHostMiddleware), which only knows the landing route — so
    # reverse() must target the app's root urlconf explicitly or it
    # raises NoReverseMatch.
    return (
        reverse("account_login", urlconf=settings.ROOT_URLCONF),
        reverse("newsserver:news_schedule", urlconf=settings.ROOT_URLCONF),
    )


def llms_txt(request: HttpRequest) -> HttpResponse:
    """Serve /llms.txt — a structured summary for AI/LLM crawlers."""
    origin = site_origin(request)
    login_url, _ = _app_urls()
    lines = [
        "# NewsBot",
        "",
        "> NewsBot reads articles across many sources every day, clusters "
        "the ones covering the same event, summarizes each one source by "
        "source, scores the tone, and emails a single digest once a week.",
        "",
        "NewsBot is an automated news intelligence tool. It collects "
        "articles from each topic's configured sources daily; once a week "
        "it clusters same-event coverage with sentence-transformer "
        "embeddings, summarizes each story per source with an LLM (checked "
        "by a judge pass for hallucinations), scores sentiment per source, "
        "and emails subscribers a themed weekly digest.",
        "",
        "## Product",
        f"- [Sign in]({login_url}): Create an account and choose topics to "
        "track.",
        f"- [How it works]({origin}/#how): The four-stage pipeline — "
        "collect, cluster, summarize & score, deliver.",
        f"- [Algorithm details]({origin}/#algorithm): Embeddings, "
        "clustering, summarization and sentiment methodology.",
        "",
        "## Source",
        "- [GitHub repository](https://github.com/Jastchi/newsbot-public): "
        "Public source for the pipeline and web app.",
        "",
        "## Optional",
        f"- [llms-full.txt]({origin}/llms-full.txt): Full page content for "
        "deeper analysis.",
    ]
    return HttpResponse("\n".join(lines), content_type=_TEXT_PLAIN_UTF8)


def llms_full_txt(_request: HttpRequest) -> HttpResponse:
    """Serve /llms-full.txt with the landing page content in full."""
    login_url, _ = _app_urls()
    lines = [
        "# NewsBot",
        "",
        "> NewsBot reads articles across many sources every day, clusters "
        "the ones covering the same event, summarizes each one source by "
        "source, scores the tone, and emails a single digest once a week.",
        "",
        f"Sign in: {login_url}",
        "Source code: https://github.com/Jastchi/newsbot-public",
        "",
        "## One event, covered many times, shown once",
        "",
        "Most events are reported by several outlets at once. NewsBot "
        "groups those articles into a single entry per story: what "
        "happened, how each source covered it, and the overall tone.",
        "",
        "## The pipeline",
        "",
        "Collection runs every day. Once a week the rest of the chain "
        "turns seven days of articles into a finished digest.",
        "",
        "1. **Collect** (daily) — Pull fresh articles from each topic's "
        "sources and extract the full text, de-duplicated.",
        "2. **Cluster** (weekly) — Group the week's articles into distinct "
        "stories by meaning, not by keyword.",
        "3. **Summarize & score** (weekly) — Summarize each story per "
        "source and read its tone, with guardrails against AI errors.",
        "4. **Deliver** (weekly) — Render the digest, email every "
        "subscriber, and archive it in the web app.",
        "",
        "## The algorithm, in detail",
        "",
        "### 1. Daily collection & filtering",
        "RSS / Atom feeds and HTML pages. A scraper agent fetches "
        "articles from every source configured for a topic. For an HTML "
        "source it discovers article links on a listing page, then pulls "
        "each full article body with trafilatura (falling back to "
        "BeautifulSoup) rather than trusting feed snippets. Each article "
        "is filtered by keyword and by semantic relevance to the topic, "
        "using a configurable cosine-similarity threshold. Duplicates are "
        "dropped by URL and title similarity before storage.",
        "",
        "### 2. Story clustering",
        "Sentence-transformer embeddings, DBSCAN / HDBSCAN, geographical "
        "penalty. Each week the last seven days of articles are embedded "
        "and grouped into stories. Embeddings run on an ONNX backend; "
        "each article is embedded as a hybrid of its title plus the start "
        "of its body. Clustering is configurable per topic (DBSCAN, "
        "HDBSCAN, or greedy) with a tunable similarity threshold. spaCy "
        "named-entity recognition extracts the locations each article "
        "mentions; when two articles name different, non-overlapping "
        "places their similarity is reduced. A cluster only becomes a "
        "top story if it draws on at least two independent sources.",
        "",
        "### 3. Summarization",
        "Two-pass LLM, per-source, LLM-as-judge, entity validation. Each "
        "story is summarized from the perspective of every source that "
        "covered it, using a cloud model or a local Ollama model. An "
        "intermediate summary is built from the cluster, then condensed "
        "into the final. A separate judge agent reviews each summary for "
        "hallucinations. Named entities in a summary are checked against "
        "the source articles.",
        "",
        "### 4. Sentiment analysis",
        "pysentimiento primary model (transformer-based, multilingual), "
        "VADER and TextBlob as fallbacks. Every article's tone is scored "
        "from -1.0 to +1.0 and labelled negative / neutral / positive, "
        "then averaged per source within each story.",
        "",
        "### 5. Report generation & delivery",
        "Jinja2, inlined CSS, HMAC unsubscribe tokens. The finished "
        "analysis is rendered to HTML, Markdown and plain text, themed "
        "from the topic's palette. Delivery runs through a pluggable "
        "provider layer — SMTP, Resend, or EmailJS. Every report is "
        "archived in the web app.",
        "",
        "## Stack",
        "Django, FastAPI, PostgreSQL, Sentence-Transformers, ONNX, spaCy "
        "NER, DBSCAN / HDBSCAN, Ollama, pysentimiento, APScheduler, "
        "Docker, Raspberry Pi.",
    ]
    return HttpResponse("\n".join(lines), content_type=_TEXT_PLAIN_UTF8)


class LandingView(TemplateView):
    """
    Public marketing landing page.

    Served at ``/`` on marketing hosts (see MarketingHostMiddleware)
    and at ``/welcome/`` on the app host for preview. Samples a
    published NewsConfig palette like the ``site_theme`` context
    processor, derives the full palette server-side for a flash-free
    first paint, and passes every published palette to the page so
    the in-page shuffle re-samples client-side.
    """

    template_name = "newsserver/landing.html"

    def get_context_data(self, **kwargs: object) -> dict[str, object]:
        """Build palette data and the app sign-in URL."""
        ctx = super().get_context_data(**kwargs)

        palettes = published_palettes()
        chosen = secrets.choice(palettes)
        # The server-rendered ``:root`` uses --m in CSS gradients, so
        # it needs a real colour; fall back to the primary when a
        # config has no middle. (The client's derivePalette matches.)
        middle = chosen["m"] or chosen["p"]
        derived = derive_color_palette(chosen["p"], chosen["s"], middle)
        ctx["palettes"] = palettes
        ctx["init"] = {
            "p": chosen["p"],
            "s": chosen["s"],
            "m": middle,
            "headline": derived["hero_color_headline"],
            "muted": derived["hero_color_muted"],
            "tint": derived["hero_color_tint"],
            "ruleAcc": derived["hero_color_border"],
            "accent": derived["hero_color_accent"],
            "link": derived["hero_color_link_dark"],
            "shadow": derived["hero_shadow"],
            "storyShadow": derived["hero_story_shadow"],
            "btnLabel": derived["hero_color_btn_label"],
        }

        ctx["app_login_url"], ctx["app_home_url"] = _app_urls()
        # "privacy/" is registered under this same relative path on
        # both the marketing and app urlconfs, so a plain
        # script-name-prefixed path works either way.
        ctx["privacy_url"] = f"{settings.FORCE_SCRIPT_NAME}/privacy/"
        return ctx


class PrivacyPolicyView(TemplateView):
    """Public privacy policy page (marketing and app hosts)."""

    template_name = "newsserver/privacy.html"

    def get_context_data(self, **kwargs: object) -> dict[str, object]:
        """Add the contact address used for data requests."""
        ctx = super().get_context_data(**kwargs)
        ctx["contact_url"], ctx["contact_label"] = _contact_url_and_label()
        return ctx


class ConfigOverviewView(LoginRequiredMixin, TemplateView):
    """Overview page showing all available configs."""

    template_name = "newsserver/config_overview.html"

    def get_context_data(self, **kwargs) -> dict:
        """Get all configs with their latest reports."""
        context = super().get_context_data(**kwargs)
        configs = ConfigService.get_active_configs_with_reports()
        # Convert dataclasses to dicts for template compatibility
        context["configs"] = [
            {
                "name": config.name,
                "key": config.key,
                "report_count": config.report_count,
                "latest_report": config.latest_report,
                "last_modified": config.last_modified,
                "storage": config.storage,
            }
            for config in configs
        ]
        return context


class ConfigReportView(LoginRequiredMixin, TemplateView):
    """
    Page showing a specific config report.

    Showing reports for a specific config with dropdown to select
    different reports
    """

    template_name = "newsserver/config_report.html"

    def get(
        self,
        request: HttpRequest,
        *args,
        **kwargs,
    ) -> HttpResponse:
        """Override get to handle download requests."""
        config_key = kwargs.get("config_name")
        if not isinstance(config_key, str):
            raise Http404("Invalid config name")
        selected_report = request.GET.get("report", None)
        download = request.GET.get("download", None)
        raw = request.GET.get("raw", None)

        if download and selected_report:
            content = ReportService.download_report(
                config_key,
                selected_report,
            )
            if content:
                response = HttpResponse(
                    content,
                    content_type="text/html",
                )
                response["Content-Disposition"] = (
                    f'attachment; filename="{selected_report}"'
                )
                return response
            raise Http404("Report file not found")

        if raw and selected_report:
            content = ReportService.get_report_content(
                config_key,
                selected_report,
            )
            if content:
                response = HttpResponse(
                    content,
                    content_type="text/html; charset=utf-8",
                )
                response["X-Frame-Options"] = "SAMEORIGIN"
                return response
            raise Http404("Report file not found")

        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs) -> dict:
        """Get context data for the report view."""
        context = super().get_context_data(**kwargs)

        config_key = kwargs.get("config_name")
        if not isinstance(config_key, str):
            context["error"] = "Invalid config name"
            return context
        selected_report = self.request.GET.get("report", None)

        # Get config from database to get display name
        news_config = ConfigService.get_config_by_key(config_key)
        if not news_config:
            context["error"] = f"Config '{config_key}' not found"
            return context

        config_display_name = news_config.display_name

        # Get reports for this config
        reports = ReportService.get_reports_for_config(config_key)

        if not reports:
            context["config_name"] = config_display_name
            context["error"] = (
                f"No reports found for config '{config_display_name}'"
            )
            return context

        # Convert ReportInfo dataclasses to dicts for template
        reports_list = [
            {
                "filename": report.filename,
                "modified": report.modified,
                "size": report.size,
            }
            for report in reports
        ]

        # Determine which report to show
        if selected_report:
            # Validate that the selected report exists
            report_filenames = [r.filename for r in reports]
            if selected_report not in report_filenames:
                selected_report = reports[0].filename  # Use latest
        else:
            selected_report = reports[0].filename  # Latest report

        # Get report content
        report_content = ReportService.get_report_content(
            config_key,
            selected_report,
        )

        if not report_content:
            context.update(
                {
                    "config_name": config_display_name,
                    "reports": reports_list,
                    "current_report": selected_report,
                    "error": "Failed to load report content",
                },
            )
            return context

        # Get storage type from first report (all reports for a config
        # use the same storage)
        storage = reports[0].storage

        context.update(
            {
                "config_name": config_display_name,
                "reports": reports_list,
                "current_report": selected_report,
                "report_content": report_content,
                "storage": storage,
            },
        )

        return context


class RunListView(LoginRequiredMixin, TemplateView):
    """View for displaying run summaries."""

    template_name = "newsserver/runs_list.html"

    def get_context_data(self, **kwargs) -> dict:
        """Get run summaries for the selected date."""
        context = super().get_context_data(**kwargs)

        # Get date from query param or default to today
        date_str = self.request.GET.get("date")
        selected_date = parse_date_or_default(date_str)

        # Calculate next/prev dates
        prev_date = selected_date - timedelta(days=1)
        next_date = selected_date + timedelta(days=1)

        # Don't allow next date if it's in the future
        if next_date > timezone.now().date():
            next_date = None

        # Fetch runs for the selected date
        # We filter by range to handle timezone differences correctly
        start_of_day, end_of_day = get_date_range(selected_date)

        scrape_runs = (
            ScrapeSummary.objects.filter(
                timestamp__range=(start_of_day, end_of_day),
            )
            .select_related("config")
            .order_by("-timestamp")
        )

        analysis_runs = (
            AnalysisSummary.objects.filter(
                timestamp__range=(start_of_day, end_of_day),
            )
            .select_related("config")
            .order_by("-timestamp")
        )

        context.update(
            {
                "selected_date": selected_date,
                "prev_date": prev_date,
                "next_date": next_date,
                "scrape_runs": scrape_runs,
                "analysis_runs": analysis_runs,
            },
        )

        return context


class LogsView(LoginRequiredMixin, TemplateView):
    """View for displaying log files."""

    template_name = "newsserver/logs_list.html"

    def get(
        self,
        request: HttpRequest,
        *args,
        **kwargs,
    ) -> HttpResponse:
        """Override get to handle download requests."""
        selected_log = request.GET.get("log", None)
        download = request.GET.get("download", None)

        if download and selected_log:
            log_path = LogService.validate_log_path(selected_log)
            if log_path:
                response = FileResponse(log_path.open("rb"))
                response["Content-Disposition"] = (
                    f'attachment; filename="{selected_log}"'
                )
                return cast("HttpResponse", response)
            raise Http404("Log file not found")

        return super().get(request, *args, **kwargs)

    def get_context_data(self, **kwargs) -> dict:
        """Get context data for the logs view."""
        context = super().get_context_data(**kwargs)

        selected_log = self.request.GET.get("log", None)

        # Get all log files
        logs_list = LogService.get_log_files()

        if not logs_list:
            context.update(
                {
                    "logs": [],
                    "config_tabs": [],
                    "error": "No log files found",
                },
            )
            return context

        # Convert LogFileInfo dataclasses to dicts for template
        logs_dict_list = [
            {
                "filename": log.filename,
                "config_name": log.config_name,
                "modified": log.modified,
                "size": log.size,
                "is_active": log.is_active,
            }
            for log in logs_list
        ]

        # Get config tabs
        config_tabs = LogService.get_config_tabs()

        # Determine which log to show
        if selected_log:
            # Validate the selected log exists
            log_filenames = [log.filename for log in logs_list]
            if selected_log not in log_filenames:
                selected_log = logs_list[0].filename  # Use latest
        else:
            selected_log = logs_list[0].filename  # Latest log

        # Get active tab for the selected log
        active_tab = LogService.get_active_tab_for_log(selected_log)

        # Get log content
        log_content = LogService.get_log_content(selected_log)

        # Check if current log is active
        current_log_info = next(
            (log for log in logs_list if log.filename == selected_log),
            None,
        )
        is_current_log_active = (
            current_log_info.is_active if current_log_info else False
        )

        context.update(
            {
                "logs": logs_dict_list,
                "config_tabs": config_tabs,
                "active_tab": active_tab,
                "current_log": selected_log,
                "log_content": log_content,
                "is_current_log_active": is_current_log_active,
            },
        )
        return context


MAX_INITIAL_LINES = 1000


def _stream_log_file(log_path: Path) -> Generator[str, None, None]:
    """
    Stream new lines from a log file.

    Sends the last N lines as initial_content, then streams new lines.
    Tracks file position and yields new lines as they're written.
    Uses a polling approach to check for new content.

    Args:
        log_path: Path to the log file to stream

    Yields:
        Server-Sent Events formatted strings with new log lines

    """
    try:
        last_position = 0
        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                initial_content = "".join(deque(f, maxlen=MAX_INITIAL_LINES))
                last_position = f.tell()
            initial_payload = json.dumps(
                {"type": "initial_content", "content": initial_content},
            )
            yield f"data: {initial_payload}\n\n"
        except FileNotFoundError:
            pass

        connected_payload = json.dumps({"type": "connected"})
        yield f"data: {connected_payload}\n\n"

        while True:
            sleep(0.5)  # Poll every 500ms

            try:
                with log_path.open(encoding="utf-8", errors="replace") as f:
                    f.seek(0, 2)  # Seek to end
                    current_size = f.tell()

                    if current_size < last_position:
                        # File was rotated or truncated, reset position
                        last_position = 0
                        f.seek(0)
                    elif current_size > last_position:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()

                        for line in new_lines:
                            payload = {
                                "type": "log_line",
                                "content": line.rstrip("\n\r"),
                            }
                            yield f"data: {json.dumps(payload)}\n\n"
            except FileNotFoundError:
                error_payload = json.dumps(
                    {"type": "error", "message": "Log file no longer exists"},
                )
                yield f"data: {error_payload}\n\n"
                break
            except Exception as e:
                error_payload = json.dumps(
                    {"type": "error", "message": f"Error reading log: {e!s}"},
                )
                yield f"data: {error_payload}\n\n"
                break

    except Exception as e:
        error_payload = json.dumps(
            {"type": "error", "message": f"Stream error: {e!s}"},
        )
        yield f"data: {error_payload}\n\n"


@login_required
def log_stream_view(
    request: HttpRequest,
) -> HttpResponse | StreamingHttpResponse:
    """
    Stream log file content using Server-Sent Events.

    Args:
        request: HTTP request with 'log' query parameter

    Returns:
        StreamingHttpResponse with SSE content, or HttpResponse for
        errors

    """
    selected_log = request.GET.get("log", None)
    if not selected_log:
        return HttpResponse("Missing log parameter", status=400)

    # Validate log path using LogService
    log_path = LogService.validate_log_path(selected_log)
    if not log_path:
        # Check if it's path traversal (403) or file not found (404)
        if LogService.is_safe_log_path(selected_log):
            return HttpResponse("Log file not found", status=404)
        return HttpResponse("Invalid log file path", status=403)

    # Only allow streaming for active log files
    if not LogService.can_stream_log(selected_log):
        return HttpResponse(
            "Streaming only available for active log files",
            status=400,
        )

    # Create streaming response with SSE headers
    response = StreamingHttpResponse(
        _stream_log_file(log_path),
        content_type="text/event-stream",
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # Disable nginx buffering
    return response

class NewsSchedulerDashboardView(TemplateView):
    """
    View for displaying the news scheduler dashboard.

    Anonymous users can view the schedule but not news sources.
    Only authenticated users can subscribe; only staff can update
    the schedule (POST / drag-and-drop).
    """

    template_name = "newsserver/news_scheduler_calendar.html"

    def get_context_data(self, **kwargs) -> dict:
        """Add subscriber, request state, and edit flag for template."""
        context = super().get_context_data(**kwargs)
        user = self.request.user
        context["user_can_edit_schedule"] = (
            user.is_authenticated and getattr(user, "is_staff", False)
        )
        subscriber = (
            cast("Subscriber", user) if user.is_authenticated else None
        )
        # Show "request to be added" when subscriber has no configs
        subscriber_request_pending = False
        if subscriber is not None:
            has_configs = subscriber.configs.filter(
                is_active=True,
                published_for_subscription=True,
            ).exists()
            subscriber_request_pending = not has_configs
        context["subscriber"] = subscriber
        context["subscriber_request_pending"] = subscriber_request_pending
        context["subscriber_request_already_sent"] = False
        if subscriber is not None and subscriber_request_pending:
            context["subscriber_request_already_sent"] = (
                SubscriberRequest.objects.filter(
                    email__iexact=subscriber.email,
                ).exists()
            )
        context["subscribable_configs"] = NewsConfig.objects.filter(
            is_active=True,
            published_for_subscription=True,
        ).order_by("display_name")
        if subscriber is not None:
            context["subscribed_config_ids"] = set(
                subscriber.configs.filter(
                    is_active=True,
                    published_for_subscription=True,
                ).values_list("pk", flat=True),
            )
        else:
            context["subscribed_config_ids"] = set()
        context["calendar_timezone"] = settings.TIME_ZONE
        return context

    def get(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        """Handle GET request."""
        # Handle AJAX data request
        is_js_request = (
            request.headers.get("x-requested-with") == "XMLHttpRequest"
            or request.GET.get("ajax") == "1"
        )
        if is_js_request:
            return self._get_calendar_data()
        # Handle standard page request
        return super().get(request, *args, **kwargs)

    def _parse_calendar_range(
        self,
    ) -> tuple[datetime | None, datetime | None]:
        """
        Parse start and end datetime from request parameters.

        Returns:
            Tuple of (start_limit, end_limit) or (None, None) if invalid

        """
        start_param = self.request.GET.get("start")
        end_param = self.request.GET.get("end")

        if start_param and end_param:
            # FullCalendar sends ISO strings for the range
            start_limit = parse_datetime(start_param)
            end_limit = parse_datetime(end_param)
            return start_limit, end_limit

        # Fallback for manual testing or missing params
        now = timezone.now()
        return now - timedelta(days=7), now + timedelta(days=14)

    def _get_subscribed_config_ids(self) -> set[int]:
        """Return subscribed config PKs (published only)."""
        user = self.request.user
        subscriber = (
            cast("Subscriber", user) if user.is_authenticated else None
        )
        if subscriber is None:
            return set()
        return set(
            subscriber.configs.filter(
                is_active=True,
                published_for_subscription=True,
            ).values_list("pk", flat=True),
        )

    def _generate_calendar_events(
        self,
        start_limit: datetime,
        end_limit: datetime,
    ) -> list[dict[str, str | int | dict[str, bool | list[str] | str]]]:
        """
        Generate calendar events for the given date range.

        Args:
            start_limit: Start of the date range
            end_limit: End of the date range

        Returns:
            List of event dictionaries for FullCalendar.

        """
        events = []
        # Staff see all configs; others only published-for-subscription.
        if self.request.user.is_authenticated and getattr(
            self.request.user, "is_staff", False,
        ):
            active_configs = NewsConfig.objects.filter(is_active=True)
        else:
            active_configs = NewsConfig.objects.filter(
                is_active=True,
                published_for_subscription=True,
            )
        # Inactive configs are shown to everyone as non-clickable
        # placeholders, excluding internal test configs.
        inactive_configs = NewsConfig.objects.filter(
            is_active=False,
        ).exclude(key__icontains="test")

        is_staff = self.request.user.is_authenticated and getattr(
            self.request.user, "is_staff", False,
        )
        include_sources = self.request.user.is_authenticated
        if include_sources:
            active_configs = active_configs.prefetch_related(
                "news_sources",
            )
            inactive_configs = inactive_configs.prefetch_related(
                "news_sources",
            )
        configs = list(active_configs) + list(inactive_configs)
        subscribed_ids = self._get_subscribed_config_ids()

        # Generate events for each week in the visible range
        # We start at the beginning of the week for start_limit
        curr_week_start = start_limit - timedelta(days=start_limit.weekday())

        while curr_week_start < end_limit:
            for config in configs:
                if not config.scheduler_weekly_analysis_enabled:
                    continue

                event = self._create_analysis_event(
                    config,
                    curr_week_start,
                    is_subscribed=config.pk in subscribed_ids,
                    include_sources=include_sources,
                    is_staff=is_staff,
                )
                if event:
                    events.append(event)

            curr_week_start += timedelta(days=7)

        return events

    def _create_analysis_event(
        self,
        config: NewsConfig,
        week_start: datetime,
        *,
        is_subscribed: bool = False,
        include_sources: bool = True,
        is_staff: bool = False,
    ) -> dict[str, str | int | dict[str, bool | list[str] | str]] | None:
        """
        Create a calendar event for a config's weekly analysis.

        Args:
            config: NewsConfig instance
            week_start: Start of the week (Monday)
            is_subscribed: True if the user is subscribed to this config
            include_sources: If True, include news source names
            is_staff: True if the requesting user is staff, who may
                drag-and-drop inactive configs to reschedule them

        Returns:
            Event dictionary for FullCalendar, or None if invalid

        """
        analysis_day_idx = DAY_NAME_TO_PYTHON_WEEKDAY.get(
            config.scheduler_weekly_analysis_day_of_week,
            0,
        )
        analysis_event_date = (
            week_start + timedelta(days=analysis_day_idx)
        ).date()

        # Configured hour/minute in constants.TZ
        analysis_ev_time = time(
            hour=config.scheduler_weekly_analysis_hour,
            minute=config.scheduler_weekly_analysis_minute,
        )
        analysis_start_iso = TZ.localize(datetime.combine(
            analysis_event_date,
            analysis_ev_time,
        )).isoformat()

        source_names = (
            sorted({s.name for s in config.news_sources.all()})
            if include_sources
            else []
        )
        return {
            "id": config.pk,
            "title": config.display_name,
            "start": analysis_start_iso,
            "backgroundColor": config.hero_color_primary,
            "borderColor": (
                config.hero_color_secondary or config.hero_color_primary
            ),
            "editable": config.is_active or is_staff,
            "extendedProps": {
                "active": config.is_active,
                "subscribed": is_subscribed,
                "sourceNames": source_names,
                "colorPrimary": config.hero_color_primary,
                "colorSecondary": config.hero_color_secondary,
                "colorMiddle": config.hero_color_middle or "",
            },
        }

    def _get_calendar_data(self) -> JsonResponse:
        """Prepare JSON events for FullCalendar."""
        start_limit, end_limit = self._parse_calendar_range()

        if not start_limit or not end_limit:
            return JsonResponse([], safe=False)

        events = self._generate_calendar_events(start_limit, end_limit)
        return JsonResponse(events, safe=False)

    def post(self, request: HttpRequest, *_args, **_kwargs) -> JsonResponse:
        """Update the specific scheduler fields (staff only)."""
        if not request.user.is_authenticated or not getattr(
            request.user, "is_staff", False,
        ):
            return JsonResponse(
                {"status": "error", "message": "Forbidden"},
                status=403,
            )
        try:
            data = json.loads(request.body)
            config_id = data.get("id")
            start_str = data.get("start")

            if not config_id or not start_str:
                return JsonResponse(
                    {"status": "error", "message": "Missing id or start"},
                    status=400,
                )

            config = NewsConfig.objects.get(pk=config_id)
            new_dt = parse_datetime(start_str)

            if new_dt:
                # FullCalendar v6 uses floating wall-time (named tz, no
                # TZ plugin). Frontend naive ISO is in that TZ; localize
                # to keep wall time. Aware inputs: astimezone.
                if new_dt.tzinfo is None:
                    local_dt = TZ.localize(new_dt)
                else:
                    local_dt = new_dt.astimezone(TZ)
                config.scheduler_weekly_analysis_day_of_week = (
                    PYTHON_WEEKDAY_TO_DAY_NAME.get(local_dt.weekday())
                )
                config.scheduler_weekly_analysis_hour = local_dt.hour
                config.scheduler_weekly_analysis_minute = local_dt.minute
                config.save(
                    update_fields=[
                        "scheduler_weekly_analysis_day_of_week",
                        "scheduler_weekly_analysis_hour",
                        "scheduler_weekly_analysis_minute",
                    ],
                )

                return JsonResponse({"status": "success"})
        except (
            NewsConfig.DoesNotExist,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as e:
            return JsonResponse(
                {"status": "error", "message": str(e)},
                status=400,
            )

        return JsonResponse(
            {"status": "error", "message": "Invalid data"},
            status=400,
        )


class EmailPreviewView(View):
    """Render the email template with mock data for admin preview."""

    def get(self, request: HttpRequest, config_key: str) -> HttpResponse:
        """Return rendered preview HTML for staff users."""
        if not request.user.is_authenticated:
            return HttpResponse(status=403)
        if not cast("Subscriber", request.user).is_staff:
            return HttpResponse(status=403)

        config = get_object_or_404(NewsConfig, key=config_key)
        primary = request.GET.get("primary") or config.hero_color_primary
        secondary = request.GET.get("secondary") or config.hero_color_secondary
        middle = request.GET.get("middle") or config.hero_color_middle or None

        if primary and not primary.startswith("#"):
            primary = f"#{primary}"
        if secondary and not secondary.startswith("#"):
            secondary = f"#{secondary}"
        if middle and not middle.startswith("#"):
            middle = f"#{middle}"

        html = ReportGeneratorAgent.render_preview(
            topic=config.display_name,
            primary=primary,
            secondary=secondary,
            middle=middle,
        )
        response = HttpResponse(html)
        response["X-Frame-Options"] = "SAMEORIGIN"
        return response


def signup_google_only(request: HttpRequest) -> HttpResponse:
    """
    Signup page: only Google signup is allowed (no email/password form).

    GET: show the signup page with "Sign up with Google".
    POST: return 405 (email/password signup is disabled).
    """
    if request.method != "GET":
        return HttpResponse(
            "Sign up is only available with Google.",
            status=405,
        )
    return render(
        request,
        "account/signup.html",
        {
            "redirect_field_name": "next",
            "redirect_field_value": request.GET.get("next", ""),
        },
    )


class SubscriptionRequestedView(LoginRequiredMixin, TemplateView):
    """Page after user submitted a subscription request (logged in)."""

    template_name = "newsserver/subscription_requested.html"


def subscription_request_from_social(request: HttpRequest) -> HttpResponse:
    """
    Handle first-time social login: create SubscriberRequest, confirm.

    Called after Google (or other) login when no Subscriber exists. Data
    is stashed in session by SocialAccountAdapter; we create
    SubscriberRequest, notify admin, show "request received" (user not
    logged in).
    """
    key = SESSION_KEY_SUBSCRIPTION_REQUEST_FROM_SOCIAL
    data = request.session.pop(key, None)
    if not data:
        return redirect("newsserver:news_schedule")
    email = (data.get("email") or "").strip().lower()
    if not email:
        return redirect("newsserver:news_schedule")
    first_name = (data.get("first_name") or "").strip()
    last_name = (data.get("last_name") or "").strip()
    obj = SubscriberRequest.objects.filter(email__iexact=email).first()
    if obj:
        obj.first_name = first_name
        obj.last_name = last_name
        obj.save(update_fields=["first_name", "last_name"])
    else:
        obj = SubscriberRequest.objects.create(
            email=email,
            first_name=first_name,
            last_name=last_name,
        )
        notify_admin_subscriber_request(obj)
    return render(request, "newsserver/subscription_requested.html", {})


def _staff_required(user: object) -> bool:
    return bool(
        getattr(user, "is_authenticated", False)
        and getattr(user, "is_staff", False),
    )


@user_passes_test(_staff_required, login_url=None)
def pending_subscription_requests(request: HttpRequest) -> HttpResponse:
    """
    Staff-only page listing users who want to subscribe.

    Informs the admin so they can add the user as a Subscriber and
    assign configs.
    """
    pending = list(
        SubscriberRequest.objects.all().order_by("-created_at")[:100],
    )
    return render(
        request,
        "newsserver/pending_subscription_requests.html",
        {"pending_requests": pending},
    )


@login_required
def subscriber_request_create(request: HttpRequest) -> JsonResponse:
    """
    Create or refresh a subscriber request for the current subscriber.

    Logged-in subscribers with no configs can submit a request; admin is
    notified and can add configs. If a request exists, we refresh it.
    """
    if request.method != "POST":
        return JsonResponse(
            {"status": "error", "message": "Method not allowed"},
            status=405,
        )
    subscriber = cast("Subscriber", request.user)
    email = (getattr(subscriber, "email", None) or "").strip().lower()
    if not email:
        return JsonResponse(
            {"status": "error", "message": "No email on account"},
            status=400,
        )
    has_configs = subscriber.configs.filter(
        is_active=True,
        published_for_subscription=True,
    ).exists()
    if has_configs:
        return JsonResponse(
            {"status": "error", "message": "Already have subscriptions"},
            status=400,
        )
    obj = SubscriberRequest.objects.filter(email__iexact=email).first()
    if obj:
        obj.first_name = getattr(subscriber, "first_name", "") or ""
        obj.last_name = getattr(subscriber, "last_name", "") or ""
        obj.user = subscriber
        obj.save(update_fields=["first_name", "last_name", "user"])
    else:
        obj = SubscriberRequest.objects.create(
            email=email,
            first_name=getattr(subscriber, "first_name", "") or "",
            last_name=getattr(subscriber, "last_name", "") or "",
            user=subscriber,
        )
        notify_admin_subscriber_request(obj)
    redirect_url = reverse("newsserver:subscription_requested")
    return JsonResponse({
        "status": "success",
        "message": "We've received your request; the admin will add you.",
        "redirect_url": redirect_url,
    })


@login_required
def subscriber_subscriptions(request: HttpRequest) -> JsonResponse:
    """
    Get or update the current subscriber's config subscriptions.

    GET: return subscribed config IDs (published only).
    POST: set configs to given config_ids (published only).
    """
    subscriber = cast("Subscriber", request.user)
    published = NewsConfig.objects.filter(
        is_active=True,
        published_for_subscription=True,
    )
    if request.method == "GET":
        ids = list(
            subscriber.configs.filter(
                is_active=True,
                published_for_subscription=True,
            ).values_list("pk", flat=True),
        )
        return JsonResponse({"config_ids": ids})
    if request.method != "POST":
        return JsonResponse(
            {"status": "error", "message": "Method not allowed"},
            status=405,
        )
    try:
        data = json.loads(request.body)
        raw_ids = data.get("config_ids", [])
        if not isinstance(raw_ids, list):
            return JsonResponse(
                {"status": "error", "message": "config_ids must be a list"},
                status=400,
            )
        valid_ids = [x for x in raw_ids if isinstance(x, int)]
        valid_configs = list(published.filter(pk__in=valid_ids))
        subscriber.configs.set(valid_configs)
        return JsonResponse({
            "status": "success",
            "config_ids": [c.pk for c in valid_configs],
        })
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"status": "error", "message": str(e)},
            status=400,
        )


@login_required
def suggest_config_view(request: HttpRequest) -> HttpResponse:
    """
    GET: show the config suggestion form.

    POST: create a ConfigSuggestion and notify admin.
    """
    if request.method == "POST":
        name = (request.POST.get("name") or "").strip()
        sources = (request.POST.get("sources") or "").strip()
        note = (request.POST.get("note") or "").strip()

        errors: dict[str, str] = {}
        if not name:
            errors["name"] = "Please provide a name for the config."
        if not sources:
            errors["sources"] = "Please list at least one source."

        if errors:
            return render(
                request,
                "newsserver/suggest_config.html",
                {
                    "errors": errors,
                    "name": name,
                    "sources": sources,
                    "note": note,
                },
                status=400,
            )

        subscriber = cast("Subscriber", request.user)
        suggestion = ConfigSuggestion.objects.create(
            name=name,
            sources=sources,
            note=note,
            email=(getattr(subscriber, "email", None) or "").strip().lower(),
            submitted_by=subscriber,
        )
        notify_admin_config_suggestion(suggestion)
        return render(request, "newsserver/suggest_config_thanks.html", {})

    return render(request, "newsserver/suggest_config.html", {})


@user_passes_test(_staff_required, login_url=None)
def config_suggestions_list(request: HttpRequest) -> HttpResponse:
    """Staff-only page listing all config suggestions."""
    suggestions = list(
        ConfigSuggestion.objects.all().order_by("-created_at")[:200],
    )
    return render(
        request,
        "newsserver/config_suggestions.html",
        {"suggestions": suggestions},
    )


def _valid_unsubscribe_token(email: str, token: str) -> bool:
    expected = _generate_unsubscribe_token(email)
    if not expected:
        logger.warning(
            "UNSUBSCRIBE_TOKEN_SECRET not configured — "
            "unsubscribe endpoint accepts any request without verification",
        )
        return True
    if not token:
        return False
    return hmac.compare_digest(token, expected)


@csrf_exempt
def unsubscribe(request: HttpRequest) -> HttpResponse:
    """
    One-click unsubscribe endpoint (RFC 8058).

    GET:  render form pre-filled with ?email= and ?token= query params.
    POST: validate HMAC token (if UNSUBSCRIBE_TOKEN_SECRET is set),
          deactivate the subscriber. Token from the email link is kept
          in a hidden form field so the submit path is also signed.
    """
    if request.method == "POST":
        email = (
            request.POST.get("email") or request.GET.get("email") or ""
        ).strip().lower()
        token = request.POST.get("token") or request.GET.get("token") or ""

        if not email or not _valid_unsubscribe_token(email, token):
            return render(
                request,
                "newsserver/unsubscribe.html",
                {"invalid": True, "email": email},
            )

        subscriber = Subscriber.objects.filter(email=email).first()
        if subscriber:
            subscriber.configs.clear()
        return render(
            request,
            "newsserver/unsubscribe.html",
            {
                "unsubscribed": True,
                "email": email,
                "found": subscriber is not None,
            },
        )

    email = request.GET.get("email", "")
    token = request.GET.get("token", "")
    return render(
        request,
        "newsserver/unsubscribe.html",
        {"email": email, "token": token},
    )


def shuffle_theme(request: HttpRequest) -> HttpResponse:
    """Clear the session theme so the next page load picks a new one."""
    request.session.pop("site_theme", None)
    referer = request.META.get("HTTP_REFERER", "/")
    return redirect(referer)
