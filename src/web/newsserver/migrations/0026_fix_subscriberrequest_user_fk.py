# Migration: Fix newsserver_subscriberrequest.user_id FK to reference newsserver_subscriber
#
# If the model was migrated when AUTH_USER_MODEL was still auth.User (or during the switch),
# the FK points at auth_user. After switching to newsserver.Subscriber, we must 
# point it at newsserver_subscriber.

from django.db import migrations


def fix_subscriber_request_user_fk_forward(apps, schema_editor):
    """Point newsserver_subscriberrequest.user_id at newsserver_subscriber."""
    vendor = schema_editor.connection.vendor
    if vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        # Find FK constraint on newsserver_subscriberrequest that references auth_user
        cursor.execute(
            """
            SELECT c.conname
            FROM pg_constraint c
            JOIN pg_class t ON c.conrelid = t.oid
            WHERE t.relname = 'newsserver_subscriberrequest'
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
            "ALTER TABLE newsserver_subscriberrequest DROP CONSTRAINT IF EXISTS " + quoted,
        )
        cursor.execute(
            """
            ALTER TABLE newsserver_subscriberrequest
            ADD CONSTRAINT newsserver_subscriberrequest_user_id_fk_subscriber
            FOREIGN KEY (user_id) REFERENCES newsserver_subscriber(id) ON DELETE SET NULL
            DEFERRABLE INITIALLY DEFERRED
            """
        )


def fix_subscriber_request_user_fk_reverse(apps, schema_editor):
    """Drop the subscriber FK (reverse of forward)."""
    vendor = schema_editor.connection.vendor
    if vendor != "postgresql":
        return
    with schema_editor.connection.cursor() as cursor:
        cursor.execute(
            """
            ALTER TABLE newsserver_subscriberrequest
            DROP CONSTRAINT IF EXISTS newsserver_subscriberrequest_user_id_fk_subscriber
            """
        )


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0025_subscriberrequest_admin_notified_at_and_more"),
    ]

    operations = [
        migrations.RunPython(
            fix_subscriber_request_user_fk_forward, 
            fix_subscriber_request_user_fk_reverse
        ),
    ]
