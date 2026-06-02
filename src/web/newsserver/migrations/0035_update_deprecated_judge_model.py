from django.db import migrations, models


DEFAULT_JUDGE_MODEL = "gemini-2.5-flash-lite"


class Migration(migrations.Migration):

    dependencies = [
        ("newsserver", "0034_newsconfig_summarization_explain_for_outsiders"),
    ]

    operations = [
        migrations.AlterField(
            model_name="newsconfig",
            name="llm_judge_model",
            field=models.CharField(
                default=DEFAULT_JUDGE_MODEL,
                help_text="Model identifier for the judge LLM",
                max_length=200,
            ),
        ),
    ]
