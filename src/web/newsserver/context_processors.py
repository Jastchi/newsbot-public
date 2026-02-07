"""Template context processors for the newsserver app."""

from django.http import HttpRequest

from .services.log_service import LogService


def has_logs(_request: HttpRequest) -> dict[str, bool]:
    """Add has_logs to template context: True if any log files exist."""
    logs = LogService.get_log_files()
    return {"has_logs": bool(logs)}
