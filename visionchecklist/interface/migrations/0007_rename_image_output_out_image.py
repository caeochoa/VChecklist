# Generated by Django 3.2.5 on 2022-07-25 10:33

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('interface', '0006_output_image'),
    ]

    operations = [
        migrations.RenameField(
            model_name='output',
            old_name='image',
            new_name='out_image',
        ),
    ]