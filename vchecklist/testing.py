from ExperimentBuilder.builder import ExperimentBuilder
import timeit
from ExperimentBuilder import utils
from tests.perturbations import TestType
from inspect import getmembers, isfunction
import sys
import argparse
import numpy as np


def Test_perturb_config_file():
    path = "/Users/caeochoa/Documents/GUH20/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/BraTS2021_00495"
    out_path = "/Users/caeochoa/Documents/GUH20/VChecklist/vchecklist"
    tests = "/Users/caeochoa/Documents/GUH20/VChecklist/vchecklist/2022092721_experiment/tests.json"
    e = ExperimentBuilder(img_folder=path, out_folder=out_path, tests_config=tests, save_config=True, experiment_name="2022092721_experiment")
    print(timeit.timeit(e.perturb, number=1))

    return True

def Test_perturb_noconfig_file():
    path = "/Users/caeochoa/Documents/GUH20/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/BraTS2021_00495"
    out_path = "/Users/caeochoa/Documents/GUH20/VChecklist/vchecklist"
    e = ExperimentBuilder(img_folder=path, out_folder=out_path, tests_config=None, save_config=True, experiment_name="2022092721_experiment")
    print(timeit.timeit(e.perturb, number=1))

    return True

def Test_clean_dict():
    parameters = {"perturbation":"rotation"}
    acc_params = ["patch_selection","perturbation"]

    test = utils.clean_dict(parameters, acc_params) == {'patch_selection': None, 'perturbation': 'rotation'}
    return test

def Test_TestType_params():
    parameters = {"perturbation":"rotation"}
    t =  TestType(parameters=parameters)

    test = t.perturbation == "rotation"

    return test

def Test_Perturbations_well_defined():
    '''Check that all perturbations are well-defined for the first 4 degrees i.e. the output and the input have the same dimensions'''

    perturbations = TestType().pert_rules
    patch =  np.ones((50,50,50))
    test = np.array([])

    for pert in perturbations:
        print("Testing", pert)
        degrees = [perturbations[pert](patch,i).shape == patch.shape for i in range(1,5)]
        np.append(test, np.all(np.array(degrees)))
    
    return (np.all(test))


def full_test_config_file():
    return True

def full_test_noconfig_file():
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Testing Suite for Visual Checklist")
    parser.add_argument("-t", "--test", help="Test to be run")
    parser.add_argument("-a", "--all", help="Run  all tests", action="store_true")
    
    
    args = parser.parse_args()
    
    tests = {name:obj for name,obj in getmembers(sys.modules[__name__]) if isfunction(obj) and obj.__module__ == __name__ and name.split("_")[0] == "Test"}

    if args.all:
        for test in tests:
            print(test + ": ",tests[test]())
        full_test_config_file()
    elif args.test in tests:
        print(args.test + ": ", tests[args.test]())
    elif args.test == "full_test_config_file":
        full_test_config_file()
    elif args.test == "full_test_noconfig_file":
        full_test_noconfig_file()
    
