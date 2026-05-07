# Generated for NewsConfig LLM default updates (gemini-first)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('newsserver', '0027_alter_newssource_type'),
    ]

    operations = [
        migrations.AlterField(
            model_name='newsconfig',
            name='llm_provider',
            field=models.CharField(choices=[('ollama', 'Ollama'), ('gemini', 'Gemini')], default='gemini', help_text='LLM provider name', max_length=50),
        ),
        migrations.AlterField(
            model_name='newsconfig',
            name='llm_model',
            field=models.CharField(default='gemini-3-flash', help_text='LLM model identifier', max_length=200),
        ),
        migrations.AlterField(
            model_name='newsconfig',
            name='llm_judge_model',
            field=models.CharField(default='gemini-3-flash', help_text='Model identifier for the judge LLM', max_length=200),
        ),
        migrations.AlterField(
            model_name='newsconfig',
            name='llm_base_url',
            field=models.CharField(blank=True, default='http://localhost:11434', help_text='Base URL for the local LLM service. Only used when provider = ollama.', max_length=500),
        ),
    ]
