from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('newsserver', '0036_create_cache_table'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysissummary',
            name='validation_stats',
            field=models.JSONField(
                blank=True,
                default=dict,
                help_text='Judge and name-validation counters',
            ),
        ),
    ]
