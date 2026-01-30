"""Views for displaying NewsBot reports."""

import json
from collections.abc import Generator
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, ClassVar, cast

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.http import (
    FileResponse,
    Http404,
    HttpRequest,
    HttpResponse,
    JsonResponse,
    StreamingHttpResponse,
)
from django.utils import timezone
from django.utils.dateparse import parse_datetime
from django.views.generic import TemplateView

from newsbot.constants import TZ

from .models import AnalysisSummary, NewsConfig, ScrapeSummary
from .services.config_service import ConfigService
from .services.log_service import LogService
from .services.report_service import ReportService
from .utils import get_date_range, parse_date_or_default

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser


class ConfigOverviewView(TemplateView):
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


class ConfigReportView(TemplateView):
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


class RunListView(TemplateView):
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


class LogsView(TemplateView):
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
        if log_path.exists():
            with log_path.open(encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                initial_content = "".join(lines[-MAX_INITIAL_LINES:])
                last_position = f.tell()
            initial_payload = json.dumps(
                {"type": "initial_content", "content": initial_content},
            )
            yield f"data: {initial_payload}\n\n"

        # Send initial keepalive
        connected_payload = json.dumps({"type": "connected"})
        yield f"data: {connected_payload}\n\n"

        # Poll for new content
        while True:
            sleep(0.5)  # Poll every 500ms

            # Check if file still exists and is readable
            if not log_path.exists():
                error_payload = json.dumps(
                    {"type": "error", "message": "Log file no longer exists"},
                )
                yield f"data: {error_payload}\n\n"
                break

            # Open file and check for new content
            try:
                with log_path.open(encoding="utf-8", errors="replace") as f:
                    f.seek(0, 2)  # Seek to end
                    current_size = f.tell()

                    if current_size < last_position:
                        # File was rotated or truncated, reset position
                        last_position = 0
                        f.seek(0)
                    elif current_size > last_position:
                        # New content available
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()

                        # Send each new line as an SSE event
                        for line in new_lines:
                            # Strip trailing newlines
                            # (we'll add them back in display)
                            line_content = line.rstrip("\n\r")
                            # Create JSON payload and encode it properly
                            payload = {
                                "type": "log_line",
                                "content": line_content,
                            }
                            json_payload = json.dumps(payload)
                            yield f"data: {json_payload}\n\n"
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

class NewsSchedulerDashboardView(
    LoginRequiredMixin,
    UserPassesTestMixin,
    TemplateView,
):
    """View for displaying the news scheduler dashboard."""

    template_name = "newsserver/news_scheduler_calendar.html"

    # Mapping your model's DayOfWeek choices to python's weekday
    # (Mon=0, Sun=6)
    DAY_MAP: ClassVar[dict[str, int]] = {
        "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
    }
    # Reverse map for saving back to DB
    REV_DAY_MAP: ClassVar[dict[int, str]] = {v: k for k, v in DAY_MAP.items()}

    def test_func(self) -> bool:
        """Test if the user is staff for access."""
        user = self.request.user
        if user.is_authenticated:
            return cast("AbstractUser", user).is_staff
        return False

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

    def _generate_calendar_events(
        self,
        start_limit: datetime,
        end_limit: datetime,
    ) -> list[dict[str, str | int]]:
        """
        Generate calendar events for the given date range.

        Args:
            start_limit: Start of the date range
            end_limit: End of the date range

        Returns:
            List of event dictionaries for FullCalendar

        """
        events = []
        configs = NewsConfig.objects.filter(is_active=True)

        # Generate events for each week in the visible range
        # We start at the beginning of the week for start_limit
        curr_week_start = start_limit - timedelta(days=start_limit.weekday())

        while curr_week_start < end_limit:
            for config in configs:
                if not config.scheduler_weekly_analysis_enabled:
                    continue

                event = self._create_analysis_event(config, curr_week_start)
                if event:
                    events.append(event)

            curr_week_start += timedelta(days=7)

        return events

    def _create_analysis_event(
        self,
        config: NewsConfig,
        week_start: datetime,
    ) -> dict[str, str | int] | None:
        """
        Create a calendar event for a config's weekly analysis.

        Args:
            config: NewsConfig instance
            week_start: Start of the week (Monday)

        Returns:
            Event dictionary for FullCalendar, or None if invalid

        """
        analysis_day_idx = self.DAY_MAP.get(
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

        return {
            "id": config.pk,
            "title": f"{config.display_name} [{config.key}]",
            "start": analysis_start_iso,
            "backgroundColor": "#3788d8",
            "borderColor": "#2c3e50",
        }

    def _get_calendar_data(self) -> JsonResponse:
        """Prepare JSON events for FullCalendar."""
        start_limit, end_limit = self._parse_calendar_range()

        if not start_limit or not end_limit:
            return JsonResponse([], safe=False)

        events = self._generate_calendar_events(start_limit, end_limit)
        return JsonResponse(events, safe=False)

    def post(self, request: HttpRequest, *_args, **_kwargs) -> JsonResponse:
        """Update the specific scheduler fields."""
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
                # Frontend sends UTC. Convert to TZ so stored H:M match
                # what the user sees on the calendar.
                if new_dt.tzinfo is None:
                    new_dt = new_dt.replace(tzinfo=UTC)
                local_dt = new_dt.astimezone(TZ)
                config.scheduler_weekly_analysis_day_of_week = (
                    self.REV_DAY_MAP.get(local_dt.weekday())
                )
                config.scheduler_weekly_analysis_hour = local_dt.hour
                config.scheduler_weekly_analysis_minute = local_dt.minute
                config.save()

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
