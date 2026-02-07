# Migration: Fix socialaccount_socialaccount.user_id FK to reference newsserver_subscriber
#
# If socialaccount was migrated when AUTH_USER_MODEL was still auth.User, the
# socialaccount_socialaccount.user_id FK points at auth_user. After switching
# to newsserver.Subscriber, we must point it at newsserver_subscriber so
# social logins can link to Subscriber rows.

from django.db import migrations


def fix_socialaccount_user_fk_forward(apps, schema_editor):
    """Point socialaccount_socialaccount.user_id at newsserver_subscriber."""
    vendor = schema_editor.connection.vendor
    if vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        # Find FK constraint on socialaccount_socialaccount that references auth_user
        cursor.execute(
            """
            SELECT c.conname
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'socialaccount_socialaccount'
              AND c.contype = 'f'
              AND pg_get_constraintdef(c.oid) LIKE '%auth_user%'
            """
        )
        row = cursor.fetchone()
        if not row:
            return
        old_constraint = row[0]
        quoted = schema_editor.connection.ops.quote_name(old_constraint)
        cursor.execute(
            "ALTER TABLE socialaccount_socialaccount "
            "DROP CONSTRAINT IF EXISTS " + quoted,
        )
        cursor.execute(
            """
            ALTER TABLE socialaccount_socialaccount
            ADD CONSTRAINT socialaccount_socialaccount_user_id_fk_subscriber
            FOREIGN KEY (user_id) REFERENCES newsserver_subscriber(id) ON DELETE CASCADE
            DEFERRABLE INITIALLY DEFERRED
            """
        )


def fix_socialaccount_user_fk_reverse(apps, schema_editor):
    """Drop the subscriber FK (reverse of forward). Does not re-add auth_user FK."""
    vendor = schema_editor.connection.vendor
    if vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            ALTER TABLE socialaccount_socialaccount
            DROP CONSTRAINT IF EXISTS socialaccount_socialaccount_user_id_fk_subscriber
            """
        )


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0021_subscriber_as_auth_user"),
        ("socialaccount", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(fix_socialaccount_user_fk_forward, fix_socialaccount_user_fk_reverse),
    ]
