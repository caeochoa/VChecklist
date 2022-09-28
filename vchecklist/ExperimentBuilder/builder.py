import os
#from nnunet.inference.predict import predict_from_folder
#from nnunet.paths import network_training_output_dir, default_trainer, default_plans_identifier
#from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
#from nnunet.evaluation.evaluator import evaluate_folder
import json
import numpy as np
import tests.perturbations as perts
import tests.patches as patches
from . import utils
from datetime import datetime




class ExperimentBuilder():

    def __init__(self, img_folder, out_folder, tests_config = None, save_config=False, experiment_name=None) -> None:
        '''
        :: img_folder :: path to folder with the input images
        :: out_folder :: path to folder for output (test results and output images)
        :: tests_config :: path to file with configuration for tests.
            - tests should specify what ground truth to use (original, perturbed, clinician's...)
        
        '''
        
        self.img_folder = os.path.abspath(img_folder)
        self.out_folder = os.path.abspath(out_folder)
        if not experiment_name:
            experiment_name = datetime.now().strftime("%Y%m%d%H") + "_experiment"
        self.out_folder = utils.try_mkdir(os.path.join(self.out_folder, experiment_name))
        

        if tests_config:
            self.tests_config = os.path.abspath(tests_config)
            self.load_config(self.tests_config)

        else:
            pert = "-"
            self.tests = {}
            while pert != "":
                try:
                    print("Type of perturbation - leave blank to continue:")
                    print("Available options:", list(perts.TestType().pert_rules.keys()))
                    pert = input("::")
                    if pert == "":
                        break
                    assert pert in perts.TestType().pert_rules.keys(), "Input has to be in available perturbations"
                    
                    
                    patch_shape = input("Introduce shape of patches. Each dimension should be separated by a comma. For example, if patch has 3 dimensions of size 50, input '50,50,50':")
                    patch_shape = (int(a) for a in patch_shape.split(","))
                    
                    print("Introduce patch selection rule:")
                    print("Options:", list(perts.TestType().ps_rules.keys()))
                    selection = input("::")
                    assert selection in perts.TestType().ps_rules.keys(), "Input has to be in available selection rules"

                    prop = input("Introduce proportion of perturbed patches (0-1):")
                    assert 0 <= float(prop) <= 1, "Proportion has to be a decimal between 0 and 1"
                    k = input("Introduce degree of perturbation:")
                    #assert float(k)%90 == 0, "Angle has to be a multiple of 90"
                    self.tests[selection.capitalize()+pert.capitalize()] = {"TestType":perts.TestType(patch_selection=selection, perturbation=pert), 
                    "proportion":float(prop), "degree":float(k), "patch_shape":tuple(patch_shape)}
                except AssertionError:
                    continue
            
        if save_config:
            self.save_config()
        


    def load_images(self, folder):
        # load a batch of samples

        folder = os.path.abspath(folder)

        ### make a list of only files, excluding folders
        all_files = utils.listf(folder, "file")

        for file in all_files:
            if not file.split(".")[-2:] == ["nii", "gz"]: # Make sure all files are the right format (format is {file})"
                all_files.pop(all_files.index(file))
        ### finally, return a list with the path of each file

        return [os.path.join(folder, file) for file in all_files]
    
    def load_config(self, path):
        with open(path, "r") as f:
            self.tests = json.load(f)
        for test in self.tests:
            select = self.tests[test]["TestType"]["patch_selection_function"]
            pert = self.tests[test]["TestType"]["perturbation_function"]
            self.tests[test]["TestType"] = perts.TestType(patch_selection = select, perturbation = pert)
            self.tests[test]["patch_shape"] = tuple(self.tests[test]["patch_shape"])

    def save_config(self):
        with open(os.path.join(self.out_folder, "tests.json"), "w") as f:
            json.dump(self.tests, f, indent=2, cls=perts.TestTypeJSONEncoder)
        
    def perturb(self):
        self.pert_img_folder = utils.try_mkdir(os.path.join(self.out_folder, "images"))
        for image_path in self.load_images(self.img_folder):
            image = patches.SampleImage(img_path=image_path)
            for test in self.tests:
                out_folder = utils.try_mkdir(os.path.join(self.pert_img_folder, test), verbose=False)
                test = self.tests[test]
                image.split_into_patches(test["patch_shape"])
                image.patches = test["TestType"].apply(patches=image.patches, probability=test["proportion"], k=test["degree"])
                image.paste_patches()
                image.save_img(out_path=out_folder)




    def predict(self):
        '''
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

        if not os.path.exists(self.out_folder):

            predict_from_folder(model=model_folder_name, input_folder=self.img_folder, output_folder=self.out_folder, folds=folds, save_npz=save_npz,
                                num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=num_threads_nifti_save,
                                lowres_segmentations=lowres_segmentations, part_id=part_id, num_parts=num_parts, tta=tta, mixed_precision=mixed_precision,
                                overwrite_existing=overwrite_existing, mode=mode , overwrite_all_in_gpu=overwrite_all_in_gpu, step_size=step_size)'''
        raise NotImplementedError


    def predict_and_evaluate(self):
        pass
