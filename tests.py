from ImageMods.image_processing import SampleImage3D
from vc import load_images, perturb, predict, compare_sample
import argparse, os, json
import numpy as np

def SampleImageTest(input_folder, config_path):
    im = SampleImage3D(input_folder)
    im.apply_config("ImageMods/configs/test.csv", save_config=True, nnunet=True)

    return True



def PredictTest(input_folder, output_folder):
    
    predict(input_folder=input_folder, output_folder=output_folder)

    return True

def PredictPerturbTest(input_folder, output_folder, config_path):
    
    images = load_images(input_folder)
    perturb(images=images, config=config_path, nnunet=True)

    input_folder_perturbed = os.path.join(input_folder, "perturbed")
    output_folder_perturbed = os.path.join(output_folder, "perturbed")

    folders = [folder for folder in os.listdir(input_folder_perturbed) if not os.path.isfile(os.path.join(input_folder_perturbed, folder))]

    for folder in folders:
        input_folder_perturbed_path = os.path.join(input_folder_perturbed, folder)
        output_folder_perturbed_path = os.path.join(output_folder_perturbed, folder)
        predict(input_folder_perturbed_path, output_folder_perturbed_path)

def Property1Test(output_folder_og, output_folder_perturbed_path):

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
    parser = argparse.ArgumentParser(description="Perform tests on vchecklist")
    parser.add_argument("-t", "--test", help="Chooses a test from SampleImageTest and PredictTest")
    parser.add_argument("-i", "--input_folder", help="Input folder for predict test.")
    parser.add_argument("-o", "--output_folder", help="Output folder for predict test.")
    parser.add_argument("-c", "--config", help="Path to config file for perturb test")

    args = parser.parse_args()

    input_folder = os.path.abspath(args.input_folder)
    output_folder = os.path.abspath(args.output_folder)
    config_path = os.path.abspath(args.config)

    available_tests = ["SampleImageTest", "PredictTest", "PredictPerturbTest"]

    if args.test == "SampleImageTest" or args.test == "0":
        SampleImageTest(input_folder, config_path)
        print("SampleImageTest worked!")
    elif args.test == "PredictTest" or args.test == "1":
        PredictTest(input_folder, output_folder)
        print("PredictTest worked!")
    elif args.test == "PredictPerturbTest" or args.test == "2":
        PredictPerturbTest(input_folder, output_folder, config_path)
        print("PredictPerturbTest worked!")
    else:
        print(f"Choose a test from: {available_tests}")
