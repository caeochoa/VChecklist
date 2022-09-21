from distutils.command.upload import upload
from django.db import models
from django.utils.translation import gettext_lazy as _

# Create your models here.

class Image(models.Model):
    image = models.FileField(upload_to='images/')
    pub_date = models.DateTimeField('date published')

    class RotationDist(models.TextChoices):
        RANDOM = "R", _("Random Rotation")
        CENTRAL = "C", _("Central Rotation")
        OUTER = "O", _("Outer Rotation")
    
    class Angle(models.IntegerChoices):
        A0 = 0, _('0º')
        A1 = 1, _('90º')
        A2 = 2, _('180º')
        A3 = 3, _('270º')

    #patch_size = models.IntegerField() # PATCH_SIZE (INT)
    rotation_distribution = models.CharField(max_length=1, choices=RotationDist.choices, default=RotationDist.RANDOM) # Distribution of rotation (Choice)
    rot_proportion = models.FloatField('Rotation proportion', default=0) # Proportion (Float/int)
    angle = models.IntegerField(choices=Angle.choices, default=Angle.A0) # Angle (INT Choice)

    def __str__(self):
        name = [str(self.image.name), self.rotation_distribution, str(int(self.rot_proportion*100)), str(self.angle)]
        return "_".join(name)

    
class Output(models.Model):

    input = models.ForeignKey(Image, on_delete=models.CASCADE)
    out_image = models.FileField(upload_to='images/', default=None)
    result = models.IntegerField(default=0)
    mean_diff = models.FloatField(default=0)
    dice_score = models.FloatField()
    og_dice_score = models.FloatField()
