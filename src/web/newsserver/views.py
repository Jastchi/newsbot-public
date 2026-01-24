"""Views for displaying NewsBot reports."""

import json
import re
from collections.abc import Generator
from datetime import date, datetime, time, timedelta
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, ClassVar, cast

from django.conf import settings
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
from supabase import Client

from newsbot.constants import TZ
from utilities.storage import (
    download_from_supabase,
    get_supabase_client,
    list_supabase_reports,
    should_use_supabase_for_config,
)

from .models import AnalysisSummary, NewsConfig, ScrapeSummary

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser


def _is_active_log_file(log_filename: str) -> bool:
    """
    Check if a log file is currently active (being written to).

    Active log files don't have a date suffix pattern like .YYYY-MM-DD.
    For example:
    - newsbot.log -> active
    - newsbot.log.2025-12-13 -> inactive (rotated)

    Args:
        log_filename: Name of the log file

    Returns:
        True if the log file is active, False otherwise

    """
    # Pattern matches rotated log files: filename.log.YYYY-MM-DD
    rotated_pattern = re.compile(r"\.\d{4}-\d{2}-\d{2}$")
    return not bool(rotated_pattern.search(log_filename))


class ConfigOverviewView(TemplateView):
    """Overview page showing all available configs."""

    template_name = "newsserver/config_overview.html"

    def get_context_data(self, **kwargs) -> dict:
        """Get all configs with their latest reports."""
        context = super().get_context_data(**kwargs)
        configs_data = []

        # Get Supabase client (if available)
        supabase_client = get_supabase_client()

        # Query all active configs from the database
        news_configs = NewsConfig.objects.filter(is_active=True).order_by(
            "display_name",
        )

        for news_config in news_configs:
            config_key = news_config.key
            config_name = news_config.display_name

            # Check if this config uses Supabase
            use_supabase = should_use_supabase_for_config(config_key)

            if use_supabase and supabase_client:
                # List reports from Supabase
                reports = list_supabase_reports(
                    supabase_client,
                    "Reports",
                    config_key,
                )
                if reports:
                    # Parse timestamp from filename
                    latest_report = max(
                        reports,
                        key=lambda x: x.get("updated_at", ""),
                    )
                    configs_data.append(
                        {
                            "name": config_name,
                            "key": config_key,
                            "report_count": len(reports),
                            "latest_report": latest_report["name"],
                            "last_modified": datetime.fromisoformat(
                                latest_report["updated_at"],
                            ),
                            "storage": "supabase",
                        },
                    )
            else:
                # List reports from local filesystem
                config_dir = settings.REPORTS_DIR / config_key
                if config_dir.exists() and config_dir.is_dir():
                    html_reports = sorted(
                        config_dir.glob("*.html"),
                        key=lambda x: x.stat().st_mtime,
                        reverse=True,
                    )

                    if html_reports:
                        latest_report = html_reports[0]
                        configs_data.append(
                            {
                                "name": config_name,
                                "key": config_key,
                                "report_count": len(html_reports),
                                "latest_report": latest_report.name,
                                "last_modified": datetime.fromtimestamp(
                                    latest_report.stat().st_mtime,
                                    TZ,
                                ),
                                "storage": "local",
                            },
                        )

        # Sort by display name
        configs_data.sort(key=lambda x: x["name"])

        context["configs"] = configs_data
        return context


class ConfigReportView(TemplateView):
    """
    Page showing a specific config report.

    Showing reports for a specific config with dropdown to select
    different reports
    """

    template_name = "newsserver/config_report.html"

    def _get_supabase_reports_context(
        self,
        context: dict,
        config_key: str,
        config_display_name: str,
        selected_report: str | None,
        supabase_client: Client,
    ) -> dict:
        """Get reports context from Supabase."""
        reports = list_supabase_reports(
            supabase_client,
            "Reports",
            config_key,
        )

        if not reports:
            context["error"] = (
                f"No reports found for config '{config_display_name}'"
            )
            return context

        # Sort by updated_at (most recent first)
        reports.sort(
            key=lambda x: x.get("updated_at", ""),
            reverse=True,
        )

        # Prepare report list for dropdown
        reports_list = [
            {
                "filename": report["name"],
                "modified": datetime.fromisoformat(
                    report["updated_at"],
                ),
                "size": report.get("metadata", {}).get("size", 0),
            }
            for report in reports
        ]

        current_report_name = (
            selected_report or reports[0]["name"]
        )  # Latest

        # Download report content from Supabase
        file_path = f"{config_key}/{current_report_name}"
        content = download_from_supabase(
            supabase_client,
            "Reports",
            file_path,
        )

        if content:
            report_content = content.decode("utf-8")
            context.update(
                {
                    "config_name": config_display_name,
                    "reports": reports_list,
                    "current_report": current_report_name,
                    "report_content": report_content,
                    "storage": "supabase",
                },
            )
        else:
            context["error"] = "Failed to load report from Supabase"

        return context

    def _get_local_reports_context(
        self,
        context: dict,
        config_key: str,
        config_display_name: str,
        selected_report: str | None,
    ) -> dict:
        """Get reports context from local filesystem."""
        config_dir = settings.REPORTS_DIR / config_key

        if not config_dir.exists():
            context["error"] = f"Config '{config_display_name}' not found"
            return context

        # Get all HTML reports for this config
        html_reports = sorted(
            config_dir.glob("*.html"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )

        if not html_reports:
            context["error"] = (
                f"No reports found for config '{config_display_name}'"
            )
            return context

        # Prepare report list for dropdown
        reports_list = [
            {
                "filename": report_file.name,
                "modified": datetime.fromtimestamp(
                    report_file.stat().st_mtime,
                    TZ,
                ),
                "size": report_file.stat().st_size,
            }
            for report_file in html_reports
        ]

        # Determine which report to show
        if selected_report:
            current_report_path = config_dir / selected_report
            if not current_report_path.exists():
                current_report_path = html_reports[0]
        else:
            current_report_path = html_reports[0]  # Latest report

        # Read the report content
        with Path(current_report_path).open(encoding="utf-8") as f:
            report_content = f.read()

        context.update(
            {
                "config_name": config_display_name,
                "reports": reports_list,
                "current_report": current_report_path.name,
                "report_content": report_content,
                "storage": "local",
            },
        )

        return context

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
            # Check if config uses Supabase
            use_supabase = should_use_supabase_for_config(config_key)
            supabase_client = get_supabase_client()

            if use_supabase and supabase_client:
                # Download from Supabase
                file_path = f"{config_key}/{selected_report}"
                content = download_from_supabase(
                    supabase_client,
                    "Reports",
                    file_path,
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
                raise Http404("Report file not found in Supabase")
            # Download from local filesystem
            config_dir = settings.REPORTS_DIR / config_key
            report_path = config_dir / selected_report
            if report_path.exists() and report_path.is_file():
                response = FileResponse(report_path.open("rb"))
                response["Content-Disposition"] = (
                    f'attachment; filename="{selected_report}"'
                )
                return cast("HttpResponse", response)
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
        try:
            news_config = NewsConfig.objects.get(
                key=config_key,
                is_active=True,
            )
            config_display_name = news_config.display_name
        except NewsConfig.DoesNotExist:
            context["error"] = f"Config '{config_key}' not found"
            return context

        # Check if config uses Supabase
        use_supabase = should_use_supabase_for_config(config_key)
        supabase_client = get_supabase_client()

        if use_supabase and supabase_client:
            return self._get_supabase_reports_context(
                context,
                config_key,
                config_display_name,
                selected_report,
                supabase_client,
            )

        return self._get_local_reports_context(
            context,
            config_key,
            config_display_name,
            selected_report,
        )


class RunListView(TemplateView):
    """View for displaying run summaries."""

    template_name = "newsserver/runs_list.html"

    def get_context_data(self, **kwargs) -> dict:
        """Get run summaries for the selected date."""
        context = super().get_context_data(**kwargs)

        # Get date from query param or default to today
        date_str = self.request.GET.get("date")
        if date_str:
            try:
                selected_date = date.fromisoformat(date_str)
            except ValueError:
                selected_date = timezone.now().date()
        else:
            selected_date = timezone.now().date()

        # Calculate next/prev dates
        prev_date = selected_date - timedelta(days=1)
        next_date = selected_date + timedelta(days=1)

        # Don't allow next date if it's in the future
        if next_date > timezone.now().date():
            next_date = None

        # Fetch runs for the selected date
        # We filter by range to handle timezone differences correctly
        start_of_day = datetime.combine(selected_date, datetime.min.time())
        end_of_day = datetime.combine(selected_date, datetime.max.time())

        # Make them timezone aware using configured timezone
        if timezone.is_aware(timezone.now()):
            start_of_day = start_of_day.replace(tzinfo=TZ)
            end_of_day = end_of_day.replace(tzinfo=TZ)

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

        logs_dir = settings.BASE_DIR / "logs"

        if download and selected_log:
            # Handle file download with path traversal protection
            log_path = (logs_dir / selected_log).resolve()
            # Ensure the resolved path is within the logs directory
            if not log_path.is_relative_to(logs_dir.resolve()):
                raise Http404("Invalid log file path")
            if log_path.exists() and log_path.is_file():
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

        logs_dir = settings.BASE_DIR / "logs"
        selected_log = self.request.GET.get("log", None)

        if not logs_dir.exists():
            context["error"] = "Logs directory not found"
            return context

        # Get all log files: active log first, then by date descending.
        # Active logs get higher priority when sorting descending.
        # Rotated logs sort by filename desc so newest dates come first.
        log_files = sorted(
            logs_dir.glob("*.log*"),
            key=lambda x: (_is_active_log_file(x.name), x.name),
            reverse=True,
        )

        if not log_files:
            context["error"] = "No log files found"
            return context

        # Prepare log list for dropdown
        logs_list = []
        for log_file in log_files:
            # Extract config name from filename
            # e.g., "technology.log" -> "technology"
            # e.g., "technology.log.2026-01-11" -> "technology"
            filename = log_file.name
            config_name = filename.split(".log")[0]
            logs_list.append(
                {
                    "filename": filename,
                    "config_name": config_name,
                    "modified": datetime.fromtimestamp(
                        log_file.stat().st_mtime,
                        tz=TZ,
                    ),
                    "size": log_file.stat().st_size,
                    "is_active": _is_active_log_file(filename),
                },
            )

        # Build config tabs from active log files (files ending in .log)
        # Each tab represents a config's log file
        config_tabs = []
        for log_file in log_files:
            if _is_active_log_file(log_file.name):
                # Extract config name from filename
                # (e.g., "technology" from "technology.log")
                config_name = log_file.name.rsplit(".log", 1)[0]
                # Create display name with title case and underscores
                # replaced
                display_name = config_name.replace("_", " ").title()
                config_tabs.append(
                    {
                        "name": config_name,
                        "display_name": display_name,
                        "filename": log_file.name,
                    },
                )
        # Sort tabs alphabetically by display name
        config_tabs.sort(key=lambda x: x["display_name"])

        # Determine which log to show with path traversal protection
        if selected_log:
            current_log_path = (logs_dir / selected_log).resolve()
            # Ensure the resolved path is within the logs directory
            if (
                not current_log_path.is_relative_to(
                    logs_dir.resolve(),
                )
                or not current_log_path.exists()
            ):
                current_log_path = log_files[0]
        else:
            current_log_path = log_files[0]  # Latest log

        # Determine active tab based on current log
        # For rotated logs like "technology.log.2026-01-11", extract the
        # base name
        current_log_name = current_log_path.name
        if _is_active_log_file(current_log_name):
            active_tab = current_log_name.rsplit(".log", 1)[0]
        else:
            # Extract base config name from rotated log
            # e.g., "technology.log.2026-01-11" -> "technology"
            active_tab = current_log_name.split(".log")[0]

        # Read the log content (last 1000 lines to avoid memory issues)
        try:
            with current_log_path.open(
                encoding="utf-8",
                errors="replace",
            ) as f:
                lines = f.readlines()
                # Get last 1000 lines
                log_content = "".join(lines[-1000:])
        except Exception as e:
            log_content = f"Error reading log file: {e}"

        # Check if current log is active
        is_current_log_active = _is_active_log_file(current_log_path.name)

        context.update(
            {
                "logs": logs_list,
                "config_tabs": config_tabs,
                "active_tab": active_tab,
                "current_log": current_log_path.name,
                "log_content": log_content,
                "is_current_log_active": is_current_log_active,
            },
        )
        return context


def _stream_log_file(log_path: Path) -> Generator[str, None, None]:
    """
    Stream new lines from a log file.

    Tracks file position and yields new lines as they're written.
    Uses a polling approach to check for new content.

    Args:
        log_path: Path to the log file to stream

    Yields:
        Server-Sent Events formatted strings with new log lines

    """
    # Initialize position to end of file
    try:
        # Get initial file size
        if log_path.exists():
            with log_path.open(encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)  # Seek to end
                last_position = f.tell()
        else:
            last_position = 0

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


class LogStreamView:
    """View for streaming active log files using Server-Sent Events."""

    def __call__(
        self,
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

        logs_dir = settings.BASE_DIR / "logs"
        log_path = (logs_dir / selected_log).resolve()

        # Path traversal protection
        if not log_path.is_relative_to(logs_dir.resolve()):
            return HttpResponse("Invalid log file path", status=403)

        if not log_path.exists() or not log_path.is_file():
            return HttpResponse("Log file not found", status=404)

        # Only allow streaming for active log files
        if not _is_active_log_file(selected_log):
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

    def _get_calendar_data(self) -> JsonResponse:
        """Prepare JSON events for FullCalendar."""
        start_param = self.request.GET.get("start")
        end_param = self.request.GET.get("end")

        if start_param and end_param:
            # FullCalendar sends ISO strings for the range
            start_limit = parse_datetime(start_param)
            end_limit = parse_datetime(end_param)
        else:
            # Fallback for manual testing or missing params
            now = timezone.now()
            start_limit = now - timedelta(days=7)
            end_limit = now + timedelta(days=14)

        if not start_limit or not end_limit:
            return JsonResponse([], safe=False)

        events = []
        configs = NewsConfig.objects.filter(is_active=True)

        # Generate events for each week in the visible range
        # We start at the beginning of the week for start_limit
        curr_week_start = start_limit - timedelta(days=start_limit.weekday())

        while curr_week_start < end_limit:
            # Get analysis events
            for config in configs:
                if not config.scheduler_weekly_analysis_enabled:
                    continue

                analysis_day_idx = self.DAY_MAP.get(
                    config.scheduler_weekly_analysis_day_of_week,
                    0,
                )
                analysis_event_date = (
                    curr_week_start + timedelta(days=analysis_day_idx)
                ).date()

                # Use the configured hour/minute
                analysis_ev_time = time(
                    hour=config.scheduler_weekly_analysis_hour,
                    minute=config.scheduler_weekly_analysis_minute,
                )

                # Construct aware datetime if possible, or just treat
                # as UTC for now to be consistent with how the browser
                # sends it (toISOString uses UTC). Adding 'Z' ensures
                # FullCalendar knows it's UTC and converts to local.
                analysis_start_iso = (
                    f"{analysis_event_date.isoformat()}"
                    f"T{analysis_ev_time.isoformat()}Z"
                )

                events.append({
                    "id": config.id,
                    "title": f"{config.display_name} [{config.key}]",
                    "start": analysis_start_iso,
                    "backgroundColor": "#3788d8",
                    "borderColor": "#2c3e50",
                })
            curr_week_start += timedelta(days=7)

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
                # Update the specific scheduler fields in your model
                # Note: new_dt is usually UTC if sent via toISOString()
                config.scheduler_weekly_analysis_day_of_week = (
                    self.REV_DAY_MAP.get(new_dt.weekday())
                )
                config.scheduler_weekly_analysis_hour = new_dt.hour
                config.scheduler_weekly_analysis_minute = new_dt.minute
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
