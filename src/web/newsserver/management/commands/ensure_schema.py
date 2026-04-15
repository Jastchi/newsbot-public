"""
Create the Postgres schema for the current ENVIRONMENT if missing.

Django's `migrate` will fail if the target schema does not exist
(`django_migrations` has nowhere to live). Run this once before
`migrate` so the dev schema is provisioned automatically.
"""

from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    """Provision the configured Postgres schema if it does not exist."""

    help = "Ensure the configured Postgres DB_SCHEMA exists."

    def handle(self, *_args: object, **_options: object) -> None:
        """Run the schema-creation statement."""
        if connection.vendor != "postgresql":
            self.stdout.write(
                f"Skipping: backend is {connection.vendor}, not postgresql.",
            )
            return

        # settings.DB_SCHEMA was validated at import time against
        # ^[a-z_][a-z0-9_]{0,62}$, so interpolation is safe here.
        schema = settings.DB_SCHEMA
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')

        self.stdout.write(
            self.style.SUCCESS(f"Schema '{schema}' is ready."),
        )
