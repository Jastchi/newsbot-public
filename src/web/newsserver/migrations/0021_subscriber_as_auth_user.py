# Migration: Subscriber as AUTH_USER_MODEL (add password, last_login, is_staff, is_superuser)

from django.conf import settings
from django.contrib.auth.hashers import UNUSABLE_PASSWORD_PREFIX
from django.db import migrations, models
from django.utils.crypto import get_random_string


def set_unusable_passwords(apps, schema_editor):
    """Set unusable password for existing subscribers."""
    Subscriber = apps.get_model("newsserver", "Subscriber")
    for sub in Subscriber.objects.all():
        sub.password = UNUSABLE_PASSWORD_PREFIX + get_random_string(40)
        sub.save(update_fields=["password"])


def noop(apps, schema_editor):
    pass


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0020_newsconfig_published_for_subscription"),
    ]

    operations = [
        migrations.AddField(
            model_name="subscriber",
            name="password",
            field=models.CharField(default="!", max_length=128, verbose_name="password"),
            preserve_default=False,
        ),
        migrations.RunPython(set_unusable_passwords, noop),
        migrations.AddField(
            model_name="subscriber",
            name="last_login",
            field=models.DateTimeField(blank=True, null=True, verbose_name="last login"),
        ),
        migrations.AddField(
            model_name="subscriber",
            name="is_staff",
            field=models.BooleanField(
                default=False,
                help_text="Designates whether the user can log into the admin site.",
            ),
        ),
        migrations.AddField(
            model_name="subscriber",
            name="is_superuser",
            field=models.BooleanField(
                default=False,
                help_text="Designates that this user has all permissions.",
            ),
        ),
        migrations.AlterField(
            model_name="subscriberrequest",
            name="user",
            field=models.ForeignKey(
                blank=True,
                null=True,
                on_delete=models.SET_NULL,
                related_name="subscriber_requests",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
    ]
