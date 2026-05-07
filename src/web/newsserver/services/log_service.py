"""Service for handling log file operations."""

from collections import deque
from datetime import datetime
from pathlib import Path

from django.conf import settings

from newsbot.constants import TZ
from web.newsserver.datatypes import LogFileInfo
from web.newsserver.utils import (
    is_active_log_file,
    validate_path_within_directory,
)


class LogService:
    """Service class for handling log file operations."""

    @staticmethod
    def get_log_files() -> list[LogFileInfo]:
        """
        Get all log files from the logs directory.

        Returns:
            List of LogFileInfo objects, sorted with active logs first,
            then by filename descending (newest dates first)

        """
        logs_dir = settings.BASE_DIR / "logs"

        if not logs_dir.exists():
            return []

        log_files = sorted(
            logs_dir.glob("*.log*"),
            key=lambda x: (is_active_log_file(x.name), x.name),
            reverse=True,
        )

        logs_list = []
        for log_file in log_files:
            filename = log_file.name
            config_name = filename.split(".log")[0]
            stat = log_file.stat()
            logs_list.append(
                LogFileInfo(
                    filename=filename,
                    config_name=config_name,
                    modified=datetime.fromtimestamp(stat.st_mtime, tz=TZ),
                    size=stat.st_size,
                    is_active=is_active_log_file(filename),
                ),
            )

        return logs_list

    @staticmethod
    def get_log_content(log_name: str, max_lines: int = 1000) -> str:
        """
        Get the content of a log file (last N lines).

        Args:
            log_name: Name of the log file
            max_lines: Maximum number of lines to return (default: 1000)

        Returns:
            Log content as string, or error message if file cannot
            be read

        """
        log_path = LogService.validate_log_path(log_name)
        if not log_path:
            return f"Error: Invalid log file path: {log_name}"

        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                return "".join(deque(f, maxlen=max_lines))
        except Exception as e:
            return f"Error reading log file: {e}"

    @staticmethod
    def is_safe_log_path(log_name: str) -> bool:
        """
        Check if a log path is safe (not a path traversal attack).

        Args:
            log_name: Name of the log file

        Returns:
            True if the path is safe (within logs directory),
            False otherwise

        """
        logs_dir = settings.BASE_DIR / "logs"
        return validate_path_within_directory(log_name, logs_dir) is not None

    @staticmethod
    def validate_log_path(log_name: str) -> Path | None:
        """
        Validate and return a safe log file path.

        Includes path traversal protection to prevent directory escape.

        Args:
            log_name: Name of the log file

        Returns:
            Path object if valid, safe, and exists, None otherwise

        """
        logs_dir = settings.BASE_DIR / "logs"
        log_path = validate_path_within_directory(log_name, logs_dir)

        if log_path is None or not log_path.is_file():
            return None

        return log_path

    @staticmethod
    def get_config_tabs() -> list[dict[str, str]]:
        """
        Build config tabs from active log files.

        Returns:
            List of dictionaries with 'name', 'display_name', and
            'filename' keys, sorted alphabetically by display name

        """
        logs_dir = settings.BASE_DIR / "logs"

        if not logs_dir.exists():
            return []

        log_files = sorted(
            logs_dir.glob("*.log*"),
            key=lambda x: (is_active_log_file(x.name), x.name),
            reverse=True,
        )

        config_tabs = []
        for log_file in log_files:
            if is_active_log_file(log_file.name):
                config_name = log_file.name.rsplit(".log", 1)[0]
                display_name = config_name.replace("_", " ").title()
                config_tabs.append(
                    {
                        "name": config_name,
                        "display_name": display_name,
                        "filename": log_file.name,
                    },
                )

        config_tabs.sort(key=lambda x: x["display_name"])
        return config_tabs

    @staticmethod
    def get_active_tab_for_log(log_name: str) -> str:
        """
        Get the active tab name for a given log file.

        Args:
            log_name: Name of the log file

        Returns:
            Config name for the active tab

        """
        if is_active_log_file(log_name):
            return log_name.rsplit(".log", 1)[0]

        return log_name.split(".log", maxsplit=1)[0]

    @staticmethod
    def can_stream_log(log_name: str) -> bool:
        """
        Check if a log file can be streamed.

        Only active log files can be streamed.

        Args:
            log_name: Name of the log file

        Returns:
            True if the log can be streamed, False otherwise

        """
        return is_active_log_file(log_name)
