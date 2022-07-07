from ImageMods.image_processing import SampleImage3D
from vc import load_images, predict
import argparse

def SampleImageTest():
    im = SampleImage3D("/Users/caeochoa/Documents/GUH20/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/BraTS2021_00495/imagesTs/BraTS2021_00495_0000.nii.gz")
    im.apply_config("ImageMods/configs/test.csv", save_config=True, nnunet=True)

    return True



def PredictTest(input_folder, output_folder):
    
    predict(input_folder=input_folder, output_folder=output_folder)

    return True



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform tests on vchecklist")
    parser.add_argument("-t", "--test", help="Chooses a test from SampleImageTest and PredictTest")
    parser.add_argument("-i", "--input_folder", help="Input folder for predict test.")
    parser.add_argument("-o", "--output_folder", help="Output folder for predict test.")
    args = parser.parse_args()

    available_tests = ["SampleImageTest", "PredictTest"]

    if args.test == "SampleImageTest" or "0":
        SampleImageTest()
        print("SampleImageTest worked!")
    if args.test == "PredictTest" or "1":
        PredictTest(args.input_folder, args.output_folder)
        print("PredictTest worked!")
    else:
        print(f"Choose a test from: {available_tests}")