# Create your models here.
from django.db import models

class Customer(models.Model):
    CLASS_CHOICES = ((1, 'plastic'), (2, 'glass'), (3, 'metal'))
    class_of_garbage = models.IntegerField(choices=CLASS_CHOICES)
    image = models.ImageField(upload_to='images/')

    def __str__(self):
        return self.image



