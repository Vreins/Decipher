from django.db import models
from django.contrib.auth.models import User

class Session(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

# Create your models here
class Chat(models.Model):
    agent = models.CharField(max_length=100000)
    message = models.CharField(max_length=100000)
    response = models.CharField(max_length=100000)
    sources = models.CharField(max_length=100000)
    suggestive_question = models.CharField(max_length=100000)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.message
    
    
