from vc import predict, load_images
import argparse
import os
from ImageMods.convert_nifti import *
import numpy as np

def predict_simplified(input_path, output_path):
    predict(input_path, output_path)
    output_images = load_images(output_path)
    outputs = Nifti2Numpy(output_images[0])[0]
    outputs = np.expand_dims(outputs, axis=0)
    for image in output_images[1:]:
        array = Nifti2Numpy(image)
        outputs = np.append(outputs, np.expand_dims(array, 0), axis=0)
    
    return outputs




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Visual Checklist for nn_UNet", description="Applies the specified perturbations to a dataset, feeds them to the nn_UNet model and reports the results on a established property.")
    parser.add_argument("-i", "--input", help="The path to the input data to perturb and feed the model")
    parser.add_argument("-o", "--output", help="The path to save the output of the model for the original and perturbed data.")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    print(predict_simplified(input_path, output_path).shape)

