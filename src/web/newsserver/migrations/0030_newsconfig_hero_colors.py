from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0029_remove_logging_config"),
    ]

    operations = [
        migrations.AddField(
            model_name="newsconfig",
            name="hero_color_primary",
            field=models.CharField(
                default="#5b6ee8",
                help_text="Primary brand color for email hero (hex, e.g. #5b6ee8)",
                max_length=7,
            ),
        ),
        migrations.AddField(
            model_name="newsconfig",
            name="hero_color_secondary",
            field=models.CharField(
                default="#8b52d4",
                help_text="Secondary brand color for email hero gradient end (hex, e.g. #8b52d4)",
                max_length=7,
            ),
        ),
    ]
