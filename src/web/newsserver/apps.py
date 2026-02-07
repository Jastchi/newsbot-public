"""Configuration for the newsserver app used in the main (web) app."""

from django.apps import AppConfig


class NewsserverConfig(AppConfig):
    """Configuration for the newsserver app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "web.newsserver"

    def ready(self) -> None:
        """Run when Django starts; used for signal registration etc."""
