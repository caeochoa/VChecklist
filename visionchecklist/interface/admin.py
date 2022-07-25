from django.contrib import admin

# Register your models here.

from .models import Image, Output

admin.site.register(Image)
admin.site.register(Output)