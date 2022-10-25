from ExperimentBuilder.builder import ExperimentBuilder
import timeit

path = "/Users/caeochoa/Documents/GUH20/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/BraTS2021_00495"
out_path = "/Users/caeochoa/Documents/GUH20/VChecklist/vchecklist"
tests = "/Users/caeochoa/Documents/GUH20/VChecklist/vchecklist/2022092721_experiment/tests.json"
e = ExperimentBuilder(img_folder=path, out_folder=out_path, tests_config=None, save_config=True, experiment_name="2022092721_experiment")
print(timeit.timeit(e.perturb, number=1))