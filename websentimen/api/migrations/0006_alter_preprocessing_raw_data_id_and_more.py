# Generated by Django 4.1 on 2022-08-26 01:54

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0005_alter_preprocessing_raw_data_id_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='preprocessing',
            name='raw_data_id',
            field=models.IntegerField(null=True),
        ),
        migrations.AlterField(
            model_name='preprocessing',
            name='result',
            field=models.CharField(max_length=1000),
        ),
    ]