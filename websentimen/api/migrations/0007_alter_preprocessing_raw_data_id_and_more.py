# Generated by Django 4.1 on 2022-08-26 02:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0006_alter_preprocessing_raw_data_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='preprocessing',
            name='raw_data_id',
            field=models.BigIntegerField(blank=True, default=None, null=True),
        ),
        migrations.AlterField(
            model_name='preprocessing',
            name='result',
            field=models.TextField(blank=True, default=None, null=True),
        ),
    ]
