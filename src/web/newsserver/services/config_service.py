"""Service for handling config-related operations."""

from datetime import datetime

from django.conf import settings

from newsbot.constants import TZ
from utilities.storage import (
    get_supabase_client,
    list_supabase_reports,
    should_use_supabase_for_config,
)
from web.newsserver.datatypes import ConfigWithReports
from web.newsserver.models import NewsConfig


class ConfigService:
    """Service class for handling config operations."""

    @staticmethod
    def get_active_configs_with_reports() -> list[ConfigWithReports]:
        """
        Get all active configs with their latest report information.

        Returns:
            List of ConfigWithReports objects, sorted by display name

        """
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
                        ConfigWithReports(
                            name=config_name,
                            key=config_key,
                            report_count=len(reports),
                            latest_report=latest_report["name"],
                            last_modified=datetime.fromisoformat(
                                latest_report["updated_at"],
                            ),
                            storage="supabase",
                        ),
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
                            ConfigWithReports(
                                name=config_name,
                                key=config_key,
                                report_count=len(html_reports),
                                latest_report=latest_report.name,
                                last_modified=datetime.fromtimestamp(
                                    latest_report.stat().st_mtime,
                                    TZ,
                                ),
                                storage="local",
                            ),
                        )

        # Sort by display name
        configs_data.sort(key=lambda x: x.name)

        return configs_data

    @staticmethod
    def get_config_by_key(config_key: str) -> NewsConfig | None:
        """
        Get a NewsConfig by its key.

        Args:
            config_key: Configuration key

        Returns:
            NewsConfig instance if found, None otherwise

        """
        try:
            return NewsConfig.objects.get(
                key=config_key,
                is_active=True,
            )
        except NewsConfig.DoesNotExist:
            return None
