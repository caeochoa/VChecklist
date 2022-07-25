#from .visionchecklist.interface.models import Image, Output
from django.utils import timezone
import os
import pandas as pd
import argparse


def listf(dir, type):
    if type == "file":
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    elif type == "folder":
        return [f for f in os.listdir(dir) if not os.path.isfile(os.path.join(dir, f))]
    elif not type:
        return [f for f in os.listdir(dir)]

def main(evals, dir, Image):
    evals_path = os.path.abspath(evals)
    evals = pd.read_csv(evals_path)

    og_dice = evals["og_dice"][0]

    images_path = dir #os.path.abspath(dir)

    for image in listf(images_path, "file"):
        imtype, conf = image.split("_")
        conf = conf.split(".")[0]
        if imtype == "input":
            if conf == "original":
                print(images_path, image)
                im = Image(image=os.path.join(images_path, image), pub_date=timezone.now())
                print(im.image.url)
                im.save()
                #assert im.id == 1, "OG input doesn't have id 1"
                out = im.output_set.create(out_image=os.path.join(images_path, "output_original.png"), result=1, mean_diff = 0, dice_score = og_dice, og_dice_score = og_dice)
                #assert out.id == 1, "OG output doesn't have id 1"
    
    for image in listf(images_path, "file"):
        imtype, conf = image.split("_")
        conf = conf.split(".")[0]
        mode_choices = {"rr":Image.RotationDist.RANDOM, "cr":Image.RotationDist.CENTRAL, "or":Image.RotationDist.OUTER}
        angle_choices = {"0":Image.Angle.A0, "1":Image.Angle.A1, "2":Image.Angle.A2, "3":Image.Angle.A3}
        if imtype == "input":
            if not conf == "original":
                mode, prop, angle = conf.split("-")
                print(im.image.url)
                im = Image(image=os.path.join(images_path, image),
                pub_date=timezone.now(),
                rotation_distribution=mode_choices[mode],
                rot_proportion=float(prop)/100,
                angle=angle_choices[angle])
                im.save()

                # now just add also corresponding output
                # maybe by using the conf and adding "output" to it
                im_eval = evals.loc[evals['name']==conf]
                im.output_set.create(out_image=os.path.join(images_path, "output_"+ conf + ".png"),
                result=int(im_eval['similarity']),
                mean_diff=im_eval['mean_diff'],
                dice_score=im_eval['pert_dice'],
                og_dice_score=im_eval['og_dice'])
        
    #assert len(listf(images_path, "file")) == len(Image.objects.all())*2 +1, f"Number of files: {len(listf(images_path, 'file'))}, Number of Image+Output objects: {len(Image.objects.all())*2 +1}" 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--evals", help="The path to the input data to perturb and feed the model")
    parser.add_argument("-d", "--dir", help="Survey images folder")
    args = parser.parse_args()

    main(args.evals, args.dir)