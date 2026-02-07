# Migration: Fix account_emailaddress.user_id FK to reference newsserver_subscriber
#
# If allauth account was migrated when AUTH_USER_MODEL was still auth.User, the
# account_emailaddress.user_id FK points at auth_user. After switching to
# newsserver.Subscriber, we must point it at newsserver_subscriber so
# email addresses can link to Subscriber rows.

from django.db import migrations


def fix_account_emailaddress_user_fk_forward(apps, schema_editor):
    """Point account_emailaddress.user_id at newsserver_subscriber."""
    vendor = schema_editor.connection.vendor
    if vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT c.conname
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'account_emailaddress'
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
            "ALTER TABLE account_emailaddress "
            "DROP CONSTRAINT IF EXISTS " + quoted,
        )
        cursor.execute(
            """
            ALTER TABLE account_emailaddress
            ADD CONSTRAINT account_emailaddress_user_id_fk_subscriber
            FOREIGN KEY (user_id) REFERENCES newsserver_subscriber(id) ON DELETE CASCADE
            DEFERRABLE INITIALLY DEFERRED
            """
        )


def fix_account_emailaddress_user_fk_reverse(apps, schema_editor):
    """Drop the subscriber FK (reverse of forward)."""
    vendor = schema_editor.connection.vendor
    if vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            ALTER TABLE account_emailaddress
            DROP CONSTRAINT IF EXISTS account_emailaddress_user_id_fk_subscriber
            """
        )


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0023_fix_admin_log_user_fk"),
        ("account", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(
            fix_account_emailaddress_user_fk_forward,
            fix_account_emailaddress_user_fk_reverse,
        ),
    ]
