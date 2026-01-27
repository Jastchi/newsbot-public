"""Service for handling log file operations."""

import re
from datetime import datetime
from pathlib import Path

from django.conf import settings

from newsbot.constants import TZ
from web.newsserver.datatypes import LogFileInfo


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

        # Get all log files: active log first, then by date descending.
        # Active logs get higher priority when sorting descending.
        # Rotated logs sort by filename desc so newest dates come first.
        log_files = sorted(
            logs_dir.glob("*.log*"),
            key=lambda x: (_is_active_log_file(x.name), x.name),
            reverse=True,
        )

        logs_list = []
        for log_file in log_files:
            # Extract config name from filename
            # e.g., "technology.log" -> "technology"
            # e.g., "technology.log.2026-01-11" -> "technology"
            filename = log_file.name
            config_name = filename.split(".log")[0]
            logs_list.append(
                LogFileInfo(
                    filename=filename,
                    config_name=config_name,
                    modified=datetime.fromtimestamp(
                        log_file.stat().st_mtime,
                        tz=TZ,
                    ),
                    size=log_file.stat().st_size,
                    is_active=_is_active_log_file(filename),
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
            with log_path.open(
                encoding="utf-8",
                errors="replace",
            ) as f:
                lines = f.readlines()
                # Get last max_lines lines
                return "".join(lines[-max_lines:])
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
        log_path = (logs_dir / log_name).resolve()
        return log_path.is_relative_to(logs_dir.resolve())

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
        if not LogService.is_safe_log_path(log_name):
            return None

        logs_dir = settings.BASE_DIR / "logs"
        log_path = (logs_dir / log_name).resolve()

        if not log_path.exists() or not log_path.is_file():
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

        # Get all log files
        log_files = sorted(
            logs_dir.glob("*.log*"),
            key=lambda x: (_is_active_log_file(x.name), x.name),
            reverse=True,
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
        if _is_active_log_file(log_name):
            return log_name.rsplit(".log", 1)[0]

        # Extract base config name from rotated log
        # e.g., "technology.log.2026-01-11" -> "technology"
        return log_name.split(".log")[0]

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
        return _is_active_log_file(log_name)
