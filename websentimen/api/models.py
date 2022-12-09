from django.db import models

# Create your models here.
class Data(models.Model):
    username = models.CharField(max_length=255)
    content = models.TextField()
    def __str__(self):
        return self.username
    
class Preprocessing(models.Model):
    username = models.CharField(max_length=255, blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    review_tokens_stemmed = models.TextField(blank=True, null=True)
    def __str___(self):
        return self.username

class Kaenen(models.Model):
    username = models.CharField(max_length=255, blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    cluster = models.IntegerField(blank=True, null=True)
    def __str___(self):
        return self.username
    
class SmoteKnn(models.Model):
    username = models.CharField(max_length=255, blank=True, null=True)
    content = models.TextField(blank=True, null=True)
    cluster = models.IntegerField(blank=True, null=True)
    def __str___(self):
        return self.username