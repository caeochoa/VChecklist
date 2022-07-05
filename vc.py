from ImageMods.image_processing import SampleImage3D
import os
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import network_training_output_dir, default_trainer, default_plans_identifier
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.evaluation.evaluator import evaluate_folder
import json
import numpy as np

#### This whole file assumes the images are in 3D format ".nii.gz", 
#### but the nnunet file naming format can be easily adjusted

def load_images(data_path):
    # load a batch of samples

    ## would be nice to input the location of these samples as an argument
    ## but idk how to do that so for now I'll just put it here

    data_path = os.path.abspath(data_path)

    ## assume all the images are in right format in one folder and we will create a convert function later

    ### make a list of only files, excluding folders
    all_files = [file for file in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, file))]

    for file in all_files:
        assert file.split(".")[-2:] == ["nii", "gz"] # make sure all files are the right format

    ### finally, return a list with the path of each file

    return [os.path.join(data_path, file) for file in all_files]






def perturb(images, mode:str="manual", config=None, nnunet=False):
    '''Take a batch of images and perturb each of them with the same configuration
    :param images: a list of image paths to be perturbed
    :param mode: a setting that establishes whether the perturbation is chosen mmanually or in a config file
    :param config: path to the config file if mode = file
    :param nnunet: a boolean establishing whether to use nnunet format for files'''
    # perturb each of them with the same configuration (manually chosen once, or based on config file)
    ## mode will establish wether the config is chosen manually once or on config file

    if mode == "manual" or mode == "m":
        im = SampleImage3D(images[0])
        
        pert = "-"
        while pert != "":
            pert = input("Type of perturbation (rr, cr, or) - leave blank to continue:")
            if pert == "rr":
                try:
                    conf = []
                    prop = input("Introduce proportion of perturbed patches (0-1):")
                    assert 0 <= float(prop) <= 1, "Proportion has to be a decimal between 0 and 1"
                    conf.append(float(prop))
                    k = input("Introduce angle of perturbation (multiple of 90):")
                    assert float(k)%90 == 0, "Angle has to be a multiple of 90"
                    conf.append(k//90)
                    im.random_rotation(conf[0], conf[1])
                except AssertionError:
                    continue
            if pert == "cr":
                try:
                    conf = []
                    prop = input("Introduce proportion of perturbed patches (0-1):")
                    assert 0 <= float(prop) <= 1, "Proportion has to be a decimal between 0 and 1"
                    conf.append(float(prop))
                    k = input("Introduce angle of perturbation (multiple of 90):")
                    assert float(k)%90 == 0, "Angle has to be a multiple of 90"
                    conf.append(k//90)
                    im.central_rotation(conf[0], conf[1])
                except AssertionError:
                    continue
            if pert == "or":
                try:
                    conf = []
                    prop = input("Introduce proportion of perturbed patches (0-1):")
                    assert 0 <= float(prop) <= 1, "Proportion has to be a decimal between 0 and 1"
                    conf.append(float(prop))
                    k = input("Introduce angle of perturbation (multiple of 90):")
                    assert float(k)%90 == 0, "Angle has to be a multiple of 90"
                    conf.append(k//90)
                    im.outer_rotation(conf[0], conf[1])
                except AssertionError:
                    continue
        im.save(nnunet=nnunet) # save in nnunet format, and save the config file
        config_path = images[0].split(".")[0] + "_perturbation_configs.csv"

        for image_path in images[1:]:
            im = SampleImage3D(image_path)
            im.apply_config(os.path.abspath(config_path), nnunet=nnunet)
    
    if mode == "file" or mode == "f":
        for image_path in images:
            im = SampleImage3D(image_path)
            im.apply_config(os.path.abspath(config_path), save_config=image_path==images[0], nnunet=nnunet) # this should save the config only if using the first image

            




def predict(input_folder, output_folder):
    # make a prediction for the original samples and the perturbed samples

    model = "3d_fullres" # -m
    folds = None
    save_npz = False
    num_threads_preprocessing = 6
    num_threads_nifti_save = 2
    lowres_segmentations = None
    part_id = 0
    num_parts = 1
    tta = 1
    mixed_precision = True
    overwrite_existing = True
    mode = "normal"
    overwrite_all_in_gpu = None
    step_size = 0.5

    task_name = convert_id_to_task_name(1)
    trainer = default_trainer

    model_folder_name = os.path.join(network_training_output_dir, model, task_name, trainer + "__" +
                              default_plans_identifier)

    predict_from_folder(model=model_folder_name, input_folder=input_folder, output_folder=output_folder, folds=folds, save_npz=save_npz,
                        num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=num_threads_nifti_save,
                        lowres_segmentations=lowres_segmentations, part_id=part_id, num_parts=num_parts, tta=tta, mixed_precision=mixed_precision,
                        overwrite_existing=overwrite_existing, mode=mode , overwrite_all_in_gpu=overwrite_all_in_gpu, step_size=step_size)

# evaluate each of these against true labels
# test definition of property [1] based on these evaluations

def compare_mean(dict1, dict2, label, criteria):
    return dict1["results"]["mean"][str(label)][criteria] - dict2["results"]["mean"][str(label)][criteria]

def compare_sample(dict1, dict2, sample, label, criteria):
    return dict1["results"]["all"][sample][str(label)][criteria] - dict2["results"]["mean"][str(label)][criteria]

def test_property1(output_folder_og, output_folder_perturbed_path):

    eval_og_csv = os.path.join(output_folder_og, "summary.json")
    eval_pert_csv = os.path.join(output_folder_perturbed_path, "summary.json")

    with open(eval_og_csv) as file:
        eval_og = json.load(file)
    
    with open(eval_pert_csv) as file:
        eval_pert = json.load(file)
    
    assert len(eval_og["results"]["all"]) == len(eval_pert["results"]["all"]), f"The number of samples is different between original and perturbed folders"
    
    eval_difference = []
    for sample in range(len(eval_og_csv["results"]["all"])):
        eval_difference.append([compare_sample(eval_og, eval_pert, sample, label, "Accuracy") for label in (0,1,2,4)])
    
    eval_difference = np.array(eval_difference)
    avg_accuracy = np.mean(eval_difference, axis=1)
    similarity = avg_accuracy[avg_accuracy <= 0.01].shape[0]/avg_accuracy.shape[0]*100

    return similarity
    
    
    



if __name__ == "__main__":

    data_path = "nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data"
    data_path = os.path.abspath(data_path)
    images = load_images(data_path) # load a batch of samples
    perturb(images=images, mode="manual", nnunet=True) # perturb each of them with the same configuration (manually chosen once, or based on config file)

    # predict original
    input_folder_og = data_path # -i
    output_folder_og = os.path.abspath("nn-UNet/outputs/BraTS2021/original") # would be nice to also input this with an argument
    predict(input_folder_og, output_folder_og)
    
    # predict perturbed
    input_folder_perturbed = os.path.join(data_path, "perturbed")
    output_folder_perturbed = os.path.abspath("nn-UNet/outputs/BraTS2021/perturbed")

    folders = [folder for folder in os.listdir(input_folder_perturbed) if not os.path.isfile(os.path.join(input_folder_perturbed, folder))]

    for folder in folders:
        input_folder_perturbed_path = os.path.join(input_folder_perturbed, folder)
        output_folder_perturbed_path = os.path.join(output_folder_perturbed, folder)
        predict(input_folder_perturbed_path, output_folder_perturbed_path)

    # evaluate each of these against true labels
    labels = os.path.abspath("nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data/labels")
    ## original 
    ### (assuming nnunet and BraTS)
    evaluate_folder(folder_with_gts=labels, folder_with_predictions=output_folder_og, labels = (0,1,2,4))
    
    ## perturbed 
    ### (assuming nnunet and BraTS)
    for folder in folders:
        output_folder_perturbed_path = os.path.join(output_folder_perturbed, folder)
        evaluate_folder(folder_with_gts=labels, folder_with_predictions=output_folder_perturbed_path, labels = (0,1,2,4))
        
        # test definition of property [1] based on these evaluations
        similarity = test_property1(output_folder_og, output_folder_perturbed_path)
        print(f"For configuration {folder}, {similarity}% of perturbed samples show behaviour that agrees with original samples")

