# Generated by Django 3.2.5 on 2022-07-23 10:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0002_auto_20220620_1412'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='parameter',
            name='patch_size',
        ),
        migrations.AlterField(
            model_name='image',
            name='image',
            field=models.FileField(upload_to='images/'),
        ),
    ]
