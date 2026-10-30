from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('newsserver', '0037_analysissummary_validation_stats'),
    ]

    operations = [
        migrations.AddField(
            model_name='newsconfig',
            name='llm_judge_check_source',
            field=models.BooleanField(
                default=False,
                help_text=(
                    'Whether the judge also checks outputs against the '
                    'source articles. Uses more tokens.'
                ),
            ),
        ),
    ]
