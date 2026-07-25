from django.core.management import call_command
from django.db import migrations


CACHE_TABLE = "newsbot_cache"


def create_cache_table(apps, schema_editor):
    """Create the DatabaseCache table (see settings.CACHES).

    ``createcachetable`` is idempotent — it skips tables that already
    exist — so this is safe to re-run.
    """
    call_command(
        "createcachetable",
        CACHE_TABLE,
        database=schema_editor.connection.alias,
    )


def drop_cache_table(apps, schema_editor):
    schema_editor.execute(f'DROP TABLE IF EXISTS "{CACHE_TABLE}"')


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0035_update_deprecated_judge_model"),
    ]

    operations = [
        migrations.RunPython(create_cache_table, drop_cache_table),
    ]
