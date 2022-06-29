from distutils.command.upload import upload
from django.db import models
from django.utils.translation import gettext_lazy as _

# Create your models here.

class Image(models.Model):
    image = models.ImageField(upload_to='images/')
    pub_date = models.DateTimeField('date published')

    def __str__(self) -> str:
        return self.image.name

    

class Parameter(models.Model):

    class RotationDist(models.TextChoices):
        RANDOM = "R", _("Random Rotation")
        CENTRAL = "C", _("Central Rotation")
        OUTER = "O", _("Outer Rotation")
    
    class Angle(models.IntegerChoices):
        A0 = 0, _('0º')
        A1 = 1, _('90º')
        A2 = 2, _('180º')
        A3 = 3, _('270º')
    
    image = models.ForeignKey(Image, on_delete=models.CASCADE)
    patch_size = models.IntegerField() # PATCH_SIZE (INT)
    rotation_distribution = models.CharField(max_length=1, choices=RotationDist.choices, default=RotationDist.RANDOM) # Distribution of rotation (Choice)
    rot_proportion = models.FloatField('Rotation proportion') # Proportion (Float/int)
    angle = models.IntegerField(choices=Angle.choices, default=Angle.A0) # Angle (INT Choice)

    def __str__(self):
        name = [str(self.image), str(self.patch_size), self.rotation_distribution, str(self.rot_proportion), str(self.angle)]
        return "_".join(name)