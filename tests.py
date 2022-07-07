from ImageMods.image_processing import SampleImage3D
from vc import load_images, predict
import argparse

def SampleImageTest(input_folder):
    im = SampleImage3D(input_folder)
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

    if args.test == "SampleImageTest" or args.test == "0":
        SampleImageTest(args.input_folder)
        print("SampleImageTest worked!")
    if args.test == "PredictTest" or args.test == "1":
        PredictTest(args.input_folder, args.output_folder)
        print("PredictTest worked!")
    else:
        print(f"Choose a test from: {available_tests}")
