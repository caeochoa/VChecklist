from django.contrib import admin

# Register your models here.

from .models import Image, Parameter

admin.site.register(Image)
admin.site.register(Parameter)