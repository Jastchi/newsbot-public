# Migration adding SubscriberRequest model

from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0018_newsconfig_exclude_articles_from_configs"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="SubscriberRequest",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("email", models.EmailField(max_length=254, unique=True)),
                (
                    "first_name",
                    models.CharField(blank=True, max_length=100),
                ),
                (
                    "last_name",
                    models.CharField(blank=True, max_length=100),
                ),
                (
                    "created_at",
                    models.DateTimeField(auto_now_add=True),
                ),
                (
                    "included_in_daily_email_at",
                    models.DateTimeField(
                        blank=True,
                        help_text="When this request was included in a daily admin digest.",
                        null=True,
                    ),
                ),
                (
                    "user",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=models.SET_NULL,
                        related_name="subscriber_requests",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "Subscriber request",
                "verbose_name_plural": "Subscriber requests",
                "ordering": ["-created_at"],
            },
        ),
    ]
