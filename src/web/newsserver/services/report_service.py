"""
Service for handling report retrieval from Supabase and local storage.

Handles both Supabase cloud storage and local filesystem storage.
"""

from datetime import datetime

from django.conf import settings
from supabase import Client

from newsbot.constants import TZ
from utilities.storage import (
    download_from_supabase,
    get_reports_bucket,
    get_supabase_client,
    list_supabase_reports,
    should_use_supabase_for_config,
)
from web.newsserver.datatypes import ReportInfo


class ReportService:
    """Service class for handling report operations."""

    @staticmethod
    def get_reports_for_config(config_key: str) -> list[ReportInfo]:
        """
        Get all reports for a specific config.

        Retrieves from Supabase or local storage based on config
        settings.

        Args:
            config_key: Configuration key

        Returns:
            List of ReportInfo objects sorted by modification date
            (newest first)

        """
        use_supabase = should_use_supabase_for_config(config_key)
        supabase_client = get_supabase_client()

        if use_supabase and supabase_client:
            return ReportService._get_supabase_reports(
                supabase_client,
                config_key,
            )

        return ReportService._get_local_reports(config_key)

    @staticmethod
    def _get_supabase_reports(
        supabase_client: Client,
        config_key: str,
    ) -> list[ReportInfo]:
        """Get reports from Supabase storage."""
        reports = list_supabase_reports(
            supabase_client,
            get_reports_bucket(),
            config_key,
        )

        if not reports:
            return []

        # Sort by updated_at (most recent first)
        reports.sort(
            key=lambda x: x.get("updated_at", ""),
            reverse=True,
        )

        return [
            ReportInfo(
                filename=report["name"],
                modified=datetime.fromisoformat(report["updated_at"]),
                size=report.get("metadata", {}).get("size", 0),
                storage="supabase",
            )
            for report in reports
        ]

    @staticmethod
    def _get_local_reports(config_key: str) -> list[ReportInfo]:
        """Get reports from local filesystem."""
        config_dir = settings.REPORTS_DIR / config_key
        html_reports = [
            (p, p.stat()) for p in config_dir.glob("*.html")
        ]
        html_reports.sort(key=lambda x: x[1].st_mtime, reverse=True)
        return [
            ReportInfo(
                filename=p.name,
                modified=datetime.fromtimestamp(s.st_mtime, TZ),
                size=s.st_size,
                storage="local",
            )
            for p, s in html_reports
        ]

    @staticmethod
    def get_report_content(
        config_key: str,
        report_name: str,
    ) -> str | None:
        """
        Get the content of a specific report as a string.

        Args:
            config_key: Configuration key
            report_name: Name of the report file

        Returns:
            Report content as string, or None if not found

        """
        use_supabase = should_use_supabase_for_config(config_key)
        supabase_client = get_supabase_client()

        if use_supabase and supabase_client:
            return ReportService._get_supabase_report_content(
                supabase_client,
                config_key,
                report_name,
            )

        return ReportService._get_local_report_content(config_key, report_name)

    @staticmethod
    def _get_supabase_report_content(
        supabase_client: Client,
        config_key: str,
        report_name: str,
    ) -> str | None:
        """Get report content from Supabase."""
        file_path = f"{config_key}/{report_name}"
        content = download_from_supabase(
            supabase_client,
            get_reports_bucket(),
            file_path,
        )

        if content:
            return content.decode("utf-8")
        return None

    @staticmethod
    def _get_local_report_content(
        config_key: str,
        report_name: str,
    ) -> str | None:
        """Get report content from local filesystem."""
        report_path = settings.REPORTS_DIR / config_key / report_name
        try:
            return report_path.read_text(encoding="utf-8")
        except OSError:
            return None

    @staticmethod
    def download_report(
        config_key: str,
        report_name: str,
    ) -> bytes | None:
        """
        Download a report file as bytes.

        Args:
            config_key: Configuration key
            report_name: Name of the report file

        Returns:
            Report content as bytes, or None if not found

        """
        use_supabase = should_use_supabase_for_config(config_key)
        supabase_client = get_supabase_client()

        if use_supabase and supabase_client:
            return ReportService._download_supabase_report(
                supabase_client,
                config_key,
                report_name,
            )

        return ReportService._download_local_report(config_key, report_name)

    @staticmethod
    def _download_supabase_report(
        supabase_client: Client,
        config_key: str,
        report_name: str,
    ) -> bytes | None:
        """Download report from Supabase."""
        file_path = f"{config_key}/{report_name}"
        return download_from_supabase(
            supabase_client,
            get_reports_bucket(),
            file_path,
        )

    @staticmethod
    def _download_local_report(
        config_key: str,
        report_name: str,
    ) -> bytes | None:
        """Download report from local filesystem."""
        report_path = settings.REPORTS_DIR / config_key / report_name
        try:
            return report_path.read_bytes()
        except OSError:
            return None
