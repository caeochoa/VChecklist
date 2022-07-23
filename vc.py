from ImageMods.image_processing import SampleImage3D
import os
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import network_training_output_dir, default_trainer, default_plans_identifier
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.evaluation.evaluator import evaluate_folder
import json
import numpy as np
import argparse

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
        if not file.split(".")[-2:] == ["nii", "gz"]: # Make sure all files are the right format (format is {file})"
            all_files.pop(all_files.index(file))
    ### finally, return a list with the path of each file

    return [os.path.join(data_path, file) for file in all_files]






def perturb(images, config=None, nnunet=False):
    '''Take a batch of images and perturb each of them with the same configuration
    :param images: a list of image paths to be perturbed
    :param mode: {manual | file} a setting that establishes whether the perturbation is chosen manually or in a config file
    :param config: path to the config file if mode = file
    :param nnunet: a boolean establishing whether to use nnunet format for files'''
    # perturb each of them with the same configuration (manually chosen once, or based on config file)
    ## mode will establish wether the config is chosen manually once or on config file

    if "perturbed" in os.listdir(os.path.dirname(images[0])):
        pert_path = os.path.join(os.path.dirname(images[0]), "perturbed")
        applied_perts = [folder for folder in os.listdir(os.path.dirname(pert_path)) if not os.path.isfile(os.path.join(os.path.dirname(pert_path), folder))]

    if not config:
        im = SampleImage3D(images[0])
        
        pert = "-"
        while pert != "":

            try:
                pert = input("Type of perturbation (rr, cr, or) - leave blank to continue:")
                assert pert in ["rr", "cr", "or"], "Type of perturbation has to be one of rr (Random Rotation), cr (Central Rotation) and or (Outer Rotation)"
                conf = []
                prop = input("Introduce proportion of perturbed patches (0-1):")
                assert 0 <= float(prop) <= 1, "Proportion has to be a decimal between 0 and 1"
                conf.append(float(prop))
                k = input("Introduce angle of perturbation (multiple of 90):")
                assert float(k)%90 == 0, "Angle has to be a multiple of 90"
                conf.append(float(k)//90)

                if f"{pert}-{int(conf[0]*100)}-{conf[1]}" not in applied_perts:
                    if pert == "rr":
                        im.random_rotation(conf[0], conf[1])
                    if pert == "cr":
                        im.central_rotation(conf[0], conf[1])
                    if pert == "or":
                        im.outer_rotation(conf[0], conf[1])
            
            except AssertionError:
                continue

        im.save(nnunet=nnunet) # save in nnunet format, and save the config file
        images_path, filename = os.path.split(images[0])
        config_path = os.path.join(os.path.join(images_path, "perturbed"), filename.split(".")[0] + "_perturbation_configs.csv")

        for image_path in images[1:]:
            im = SampleImage3D(image_path)
            im.apply_config(os.path.abspath(config_path), nnunet=nnunet)
    else:
        for image_path in images:
            im = SampleImage3D(image_path)
            if image_path == images[0]:
                im.load_config(os.path.abspath(config))
                for pert in im.config.keys():
                    for conf in im.config[pert]:
                        if f"{pert}-{int(conf[0]*100)}-{conf[1]}" in applied_perts:
                            print(f"{pert}-{int(conf[0]*100)}-{conf[1]} already performed on this dataset")
                            im.config[pert].pop(im.config[pert].index(conf))

            print(f"Applying configuration to {os.path.split(image_path)[1]}")
            im.apply_config(os.path.abspath(config), save_config=image_path==images[0], nnunet=nnunet) # this should save the config only if using the first image

            




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

    if not os.path.exists(output_folder):

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

def test_property1(output_folder_og, output_folder_perturbed_path, criteria):

    eval_og_json = os.path.join(output_folder_og, "summary.json")
    eval_pert_json = os.path.join(output_folder_perturbed_path, "summary.json")

    with open(eval_og_json) as file:
        eval_og = json.load(file)
    
    with open(eval_pert_json) as file:
        eval_pert = json.load(file)
    
    assert len(eval_og["results"]["all"]) == len(eval_pert["results"]["all"]), f"The number of samples is different between original and perturbed folders"
    
    eval_difference = []
    for sample in range(len(eval_og["results"]["all"])):
        eval_difference.append([compare_sample(eval_og, eval_pert, sample, label, criteria) for label in (0,1,2,4)])
    
    eval_difference = np.array(eval_difference)
    avg_diff = np.abs(np.mean(eval_difference, axis=1))
    similarity = avg_diff[avg_diff <= 0.01].shape[0]/avg_diff.shape[0]*100
    
    # means = [np.mean(np.array([dic["results"]["mean"][str(label)][criteria] for label in (0,1,2,4)])) for dic in [eval_og, eval_pert]]
    means = np.mean(np.abs(np.array([compare_mean(eval_og, eval_pert, label, criteria) for label in (0,1,2,4)])))

    return similarity, means
    

def write_report(output_folder_perturbed, labels_perturbed, output_folder_og):

    folders = [folder for folder in os.listdir(output_folder_perturbed) if not os.path.isfile(os.path.join(output_folder_perturbed, folder))]

    report = []
    for folder in folders:
        output_folder_perturbed_path = os.path.join(output_folder_perturbed, folder)
        evaluate_folder(folder_with_gts=labels_perturbed, folder_with_predictions=output_folder_perturbed_path, labels = (0,1,2,4))
        
        # test definition of property [1] based on these evaluations
        similarity, means = test_property1(output_folder_og, output_folder_perturbed_path, "Dice")
        result = f"For configuration {folder}, {similarity}% of perturbed samples show behaviour that agrees with original samples. (Mean: {means})"
        print(result)
        report.append(result)

    with open(os.path.join(output_folder_perturbed, "property_results.txt"), mode="w") as report_file:
        report_file.write("\n".join(report))

    



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visual Checklist for nn_UNet", description="Applies the specified perturbations to a dataset, feeds them to the nn_UNet model and reports the results on a established property.")
    parser.add_argument("-i", "--input", help="The path to the input data to perturb and feed the model")
    parser.add_argument("-o", "--output", help="The path to save the output of the model for the original and perturbed data.")
    parser.add_argument("-c", "--config", help="Choose configuration file for perturbation.")
    parser.add_argument("-l", "--perturb_labels", help="A flag that establishes that labels should be perturbed too.", action="store_true")
    args = parser.parse_args()

    data_path = args.input
    data_path = os.path.abspath(data_path)
    images = load_images(data_path) # load a batch of samples
    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        config_path = None
    
    perturb(images=images, config=os.path.abspath(args.config), nnunet=True) # perturb each of them with the same configuration (manually chosen once, or based on config file)

    # predict original
    input_folder_og = data_path # -i
    output_folder_og = os.path.abspath(args.output) # would be nice to also input this with an argument
    predict(input_folder_og, output_folder_og)
    
    # predict perturbed
    input_folder_perturbed = os.path.join(data_path, "perturbed")
    output_folder_perturbed = os.path.join(output_folder_og, "perturbed")

    folders = [folder for folder in os.listdir(input_folder_perturbed) if not os.path.isfile(os.path.join(input_folder_perturbed, folder))]

    for folder in folders:
        input_folder_perturbed_path = os.path.join(input_folder_perturbed, folder)
        output_folder_perturbed_path = os.path.join(output_folder_perturbed, folder)
        predict(input_folder_perturbed_path, output_folder_perturbed_path)

    # evaluate each of these against true labels
    labels = os.path.join(os.path.dirname(input_folder_og), "labels")

    if args.perturb_labels:
        labels_files = load_images(labels)
        if not config_path:
            config_path = os.path.join(input_folder_perturbed, os.path.split(labels_files[0])[1].split(".")[0] + "_perturbed.csv")
        perturb(images=labels_files, config=config_path, nnunet=True)
        labels_perturbed = os.path.join(labels, "perturbed")
    else:
        labels_perturbed = labels

    ## original 
    ### (assuming nnunet and BraTS)
    evaluate_folder(folder_with_gts=labels, folder_with_predictions=output_folder_og, labels = (0,1,2,4))
    
    ## perturbed 
    ### (assuming nnunet and BraTS)
    
    write_report(output_folder_perturbed, labels_perturbed, output_folder_og)

