# Generated migration to remove scheduler_timezone field

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('newsserver', '0015_topic_newsconfig_topics'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='newsconfig',
            name='scheduler_timezone',
        ),
    ]
