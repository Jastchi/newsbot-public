# Migration adding exclude_articles_from_configs M2M (self-referential)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0017_remove_newsconfig_scheduler_daily_scrape_hour_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="newsconfig",
            name="exclude_articles_from_configs",
            field=models.ManyToManyField(
                blank=True,
                help_text=(
                    "Optional. Exclude articles whose URL exists in any of these "
                    "configs' articles."
                ),
                related_name="configs_excluding_my_articles",
                to="newsserver.newsconfig",
            ),
        ),
    ]
