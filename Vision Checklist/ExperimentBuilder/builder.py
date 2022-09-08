import os
from nnunet.inference.predict import predict_from_folder
from nnunet.paths import network_training_output_dir, default_trainer, default_plans_identifier
from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
from nnunet.evaluation.evaluator import evaluate_folder
import json
import numpy as np



class ExperimentBuilder():

    def __init__(self, img_folder, out_folder, tests_config = None) -> None:
        '''
        :: img_folder :: path to folder with the input images
        :: out_folder :: path to folder for output (test results and output images)
        :: tests_config :: path to file with configuration for tests.
            - tests should specify what ground truth to use (original, perturbed, clinician's...)
        
        '''
        
        self.img_folder = os.path.abspath(img_folder)
        self.out_folder = os.path.abspath(out_folder)
        if tests_config:
            self.tests_config = os.path.abspath(tests_config)
    
    def load_config(self):
        
        pass

    
    def perturb(self):
        pass

    def predict(self):

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
                                overwrite_existing=overwrite_existing, mode=mode , overwrite_all_in_gpu=overwrite_all_in_gpu, step_size=step_size)


    def predict_and_evaluate(self):
        pass
