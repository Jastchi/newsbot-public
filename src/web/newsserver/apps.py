"""Configuration for the newsserver app used in the main (web) app."""

from django.apps import AppConfig
from django.conf import settings
from django.db.backends.base.base import BaseDatabaseWrapper
from django.db.backends.signals import connection_created


def _set_search_path(
    connection: BaseDatabaseWrapper,
    **_kwargs: object,
) -> None:
    """
    Pin every new Postgres connection to the env's schema.

    Done here instead of via DATABASES["OPTIONS"]["options"] because
    Supabase's Supavisor pooler rejects the `-c search_path=...`
    startup parameter and closes the connection.
    """
    if connection.vendor != "postgresql":
        return
    # settings.DB_SCHEMA was validated against ^[a-z_][a-z0-9_]{0,62}$
    # at import time, so interpolation is safe.
    schema = settings.DB_SCHEMA
    with connection.cursor() as cursor:
        cursor.execute(f'SET search_path TO "{schema}", public')


class NewsserverConfig(AppConfig):
    """Configuration for the newsserver app."""

    default_auto_field = "django.db.models.BigAutoField"
    name = "web.newsserver"

    def ready(self) -> None:
        """Run when Django starts; used for signal registration etc."""
        connection_created.connect(_set_search_path)
