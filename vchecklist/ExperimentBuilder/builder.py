import os
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
        self.pred_folder = os.path.join(self.out_folder, "outputs")
        

        if tests_config:
            self.tests_config = os.path.abspath(tests_config)
            self.load_config(self.tests_config)

        else:
            self.prompt_for_tests()
            
        if save_config:
            self.save_config()
        

    def prompt_for_tests(self):
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
                
                if selection == "InsideManual" or selection == "OutsideManual":
                    manual_path = os.path.abspath(input("Introduce path to manual selection for perturbations:"))
                    assert os.path.exists(manual_path), "File doesn't exist"
                    assert os.path.splitext(manual_path)[1] == ".npy", "Currently only .npy files are supported"
                    man = "_" + os.path.splitext(os.path.split(manual_path)[1])[0]
                else:
                    manual_path = None
                    man = ""

                prop = input("Introduce proportion of perturbed patches (0-1):")
                assert 0 <= float(prop) <= 1, "Proportion has to be a decimal between 0 and 1"
                k = input("Introduce degree of perturbation:")
                #assert float(k)%90 == 0, "Angle has to be a multiple of 90"
                parameters = {"patch_selection":selection, "perturbation":pert, "probability":float(prop), "degree":float(k), "manual_path":manual_path}
                self.tests[selection+pert.capitalize()+man] = {"TestType":perts.TestType(parameters=parameters), "patch_shape":tuple(patch_shape)}
            except AssertionError as e:
                print(e)
                continue

        return

    def load_images(self, folder):
        # load a batch of samples

        folder = os.path.abspath(folder)

        print("Loading images from:", folder)

        ### make a list of only files, excluding folders
        all_files = utils.listf(folder, "file")

        for file in all_files:
            if not file.split(".")[-2:] == ["nii", "gz"]: # Make sure all files are the right format (format is {file})"
                all_files.pop(all_files.index(file))
        ### finally, return a list with the path of each file

        samples = {}
        for file in all_files:
            filename = os.path.splitext(file)[0]
            sample_id = nnunet_get_sample_id(filename)
            try:
                samples[sample_id].append(os.path.join(folder, file))
            except KeyError:
                samples[sample_id] = [os.path.join(folder, file)]

        print(len(samples), "samples loaded")

        return samples
    
    def load_config(self, path):
        with open(path, "r") as f:
            self.tests = json.load(f)
        for test in self.tests:
            #select = self.tests[test]["TestType"]["patch_selection_function"]
            #pert = self.tests[test]["TestType"]["perturbation_function"]
            self.tests[test]["TestType"] = perts.TestType(parameters=self.tests[test]["TestType"])
            self.tests[test]["patch_shape"] = tuple(self.tests[test]["patch_shape"])
    

    def save_config(self):
        with open(os.path.join(self.out_folder, "tests.json"), "w") as f:
            json.dump(self.tests, f, indent=2, cls=perts.TestTypeJSONEncoder)
        
    def perturb(self):
        self.pert_img_folder = utils.try_mkdir(os.path.join(self.out_folder, "images"))
        for sample_id, samples in self.load_images(self.img_folder).items():
            image = patches.SampleImages(img_paths=samples)
            for test in self.tests:
                out_folder = utils.try_mkdir(os.path.join(self.pert_img_folder, test), verbose=True)
                test = self.tests[test]

                image.split_into_patches(test["patch_shape"])
                #print(image.patches)
                # clearly here theres a different selection for each image so each modality is getting different patches perturbed!!
                image.patches = test["TestType"].apply(patches=image.patches)
                image.paste_patches()
                image.save_imgs(out_path=out_folder)




    def predict_from_folder(self, input_folder, out_folder):
        
        # make a prediction for the original samples and the perturbed samples

        from nnunet.inference.predict import predict_from_folder
        from nnunet.paths import network_training_output_dir, default_trainer, default_plans_identifier
        from nnunet.utilities.task_name_id_conversion import convert_id_to_task_name
        
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

        if not os.path.exists(out_folder):

            predict_from_folder(model=model_folder_name, input_folder=input_folder, output_folder=out_folder, folds=folds, save_npz=save_npz,
                                num_threads_preprocessing=num_threads_preprocessing, num_threads_nifti_save=num_threads_nifti_save,
                                lowres_segmentations=lowres_segmentations, part_id=part_id, num_parts=num_parts, tta=tta, mixed_precision=mixed_precision,
                                overwrite_existing=overwrite_existing, mode=mode , overwrite_all_in_gpu=overwrite_all_in_gpu, step_size=step_size)
        


    def predict(self):

        # first predict on original images
        print("Starting predictions for original images")
        self.predict_from_folder(input_folder=self.img_folder, out_folder=os.path.join(self.pred_folder, "Original"))

        print("Finished predictions for original images")

        # now for each test predict from it
        for test in self.tests:
            print(f"Starting predictions for samples from test {test}")
            self.predict_from_folder(input_folder=os.path.join(self.pert_img_folder, test), out_folder=os.path.join(self.pred_folder, test))

            print(f"Finished predictions for samples from test {test}")

    def evaluate(self, labels_path, property):

        from nnunet.evaluation.evaluator import evaluate_folder
        
        labels = (0,1,2,4)

        # perform nn_unet evaluations for each set of images
        # ideally this would be possible with any set of labels: original, perturbed, given by the doctor... but for now I´ll focus on original labels
        og_preds = os.path.join(self.pred_folder, "Original")

        print("Performing evaluation of original predictions")
        print("Folder: ", og_preds)
        evaluate_folder(folder_with_gts=labels_path, folder_with_predictions=og_preds, labels = labels)

        with open(os.path.join(og_preds, "summary.json"), "r") as f:
            eval_og = json.load(f)

        self.results = self.tests.copy()

        # labels need to be padded in order to have the same size as perturbed outputs

        if not self.pert_img_folder:
            self.pert_img_folder = utils.try_mkdir(os.path.join(self.out_folder, "images"))
        
        pert_labels_path = utils.try_mkdir(os.path.join(self.pert_img_folder, "Labels"))


        for test in self.results:
            print(f"Performing evaluation of {test} predictions")

            # this still doesn't work -- we need a patch_shape and that has to be extracted from each test type
            test_labels_path = utils.try_mkdir(os.path.join(pert_labels_path, test))
            print(f"Padding labels at {test_labels_path}")
            for sample_id, samples in self.load_images(labels_path).items():
                image = patches.SampleImages(img_paths=samples).imgs[0] # assuming just on labels image per sample id
                image.out_img = np.pad(image.img, patches.pad_img(image.img.shape, self.results[test]["patch_shape"]))
                print(image.img.shape, image.out_img.shape)
                image.save_img(test_labels_path)
                assert os.path.exists(os.path.join(test_labels_path,image.filename)), f"{image.filename} wasn't saved correctly"
            print("Done")


            pred_folder = os.path.join(self.pred_folder, test)
            print("Folder: ", pred_folder)
            evaluate_folder(folder_with_gts=test_labels_path, folder_with_predictions=pred_folder, labels = labels)

            # now load the evaluation files
            
            with open(os.path.join(pred_folder, "summary.json"), "r") as f:
                eval_pert = json.load(f)
            
            assert len(eval_og["results"]["all"]) == len(eval_pert["results"]["all"]), f"The number of samples is different between original and perturbed folders"
            
            print(f"Comparing the results of {test} with the original predictions")
            # now a property should be loaded from the properties file and applied to evals
            ## right now this is much more specific than I would like because I only have one property
            similarity = property(eval_og, eval_pert, labels, "Dice")

            for label in labels:
                self.results[test][label] = [similarity[0, labels.index(label)], similarity[1, labels.index(label)]]
                mean = np.mean(similarity, 1)
                self.results[test]["summary"] = f"For configuration {test}, {mean[0]}% of perturbed samples show behaviour that agrees with original samples. (Mean: {mean[1]})"


        with open(os.path.join(self.out_folder, "results.json"), "w") as f:
            json.dump(self.results, f, indent=2, cls=perts.TestTypeJSONEncoder)

def nnunet_get_sample_id(f:str):
    return f.split("_")[1]

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
        eval_difference.append([utils.compare_sample(eval_og, eval_pert, sample, label, criteria) for label in (0,1,2,4)])
    
    eval_difference = np.array(eval_difference)
    avg_diff = np.abs(np.mean(eval_difference, axis=1))
    similarity = avg_diff[avg_diff <= 0.01].shape[0]/avg_diff.shape[0]*100
    
    # means = [np.mean(np.array([dic["results"]["mean"][str(label)][criteria] for label in (0,1,2,4)])) for dic in [eval_og, eval_pert]]
    means = np.mean(np.abs(np.array([utils.compare_mean(eval_og, eval_pert, label, criteria) for label in (0,1,2,4)])))

    return similarity, means
    

def write_report(output_folder_perturbed, labels_perturbed, output_folder_og):
    '''
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
        report_file.write("\n".join(report))'''
