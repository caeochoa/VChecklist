from ImageMods.image_processing import SampleImage3D
from vc import load_images, perturb, predict
import argparse
import os

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

    available_tests = ["SampleImageTest", "PredictTest"]

    if args.test == "SampleImageTest" or args.test == "0":
        SampleImageTest(input_folder, config_path)
        print("SampleImageTest worked!")
    if args.test == "PredictTest" or args.test == "1":
        PredictTest(input_folder, output_folder)
        print("PredictTest worked!")
    if args.test == "PredictPerturbTest" or args.test == "2":
        PredictPerturbTest(input_folder, output_folder, config_path)
        print("PredictPerturbTest worked!")
    else:
        print(f"Choose a test from: {available_tests}")
