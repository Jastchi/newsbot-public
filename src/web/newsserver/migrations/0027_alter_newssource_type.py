# Generated for HTML source type addition

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('newsserver', '0026_fix_subscriberrequest_user_fk'),
    ]

    operations = [
        migrations.AlterField(
            model_name='newssource',
            name='type',
            field=models.CharField(choices=[('rss', 'RSS'), ('html', 'HTML')], default='rss', help_text='Source type', max_length=50),
        ),
    ]
