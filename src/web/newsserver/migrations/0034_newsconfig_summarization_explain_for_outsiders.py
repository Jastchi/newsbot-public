from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('newsserver', '0033_configsuggestion'),
    ]

    operations = [
        migrations.AddField(
            model_name='newsconfig',
            name='summarization_explain_for_outsiders',
            field=models.BooleanField(
                default=False,
                help_text=(
                    'When enabled, the LLM briefly explains key institutions, '
                    'organisations, political figures, and local concepts that a '
                    'reader unfamiliar with the topic might not know.'
                ),
            ),
        ),
    ]
