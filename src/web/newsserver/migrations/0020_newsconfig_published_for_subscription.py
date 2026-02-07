# Migration adding published_for_subscription to NewsConfig

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0019_subscriber_request"),
    ]

    operations = [
        migrations.AddField(
            model_name="newsconfig",
            name="published_for_subscription",
            field=models.BooleanField(
                default=False,
                help_text=(
                    "If set, this config appears in \"My subscriptions\" and can be "
                    "chosen by subscribers. Only published configs are subscribable."
                ),
            ),
        ),
    ]
