# Generated migration to remove scheduler_daily_scrape_hour and scheduler_daily_scrape_minute
# (model uses DAILY_SCRAPE_HOUR/DAILY_SCRAPE_MINUTE constants instead)

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0016_remove_scheduler_timezone"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="newsconfig",
            name="scheduler_daily_scrape_hour",
        ),
        migrations.RemoveField(
            model_name="newsconfig",
            name="scheduler_daily_scrape_minute",
        ),
    ]
