import os, json, argparse
from ImageMods.image_processing import SampleImage3D
import vc

def perturb(images):

    '''Take a batch of images and perturb all of them with all possible configurations'''

    for image_path in images:
        image = SampleImage3D(image_path)
        for prop in range(100):
            for angle in [1,2,3]:
                image.random_rotation(prop/100, angle)
                image.central_rotation(prop/100, angle)
                image.outer_rotation(prop/100, angle)
        image.save()


def save_perturbed(image):

    '''Alternative to the SampleImage method for saving images, main difference is that
    all images will be saved into the same folder...?'''

    raise NotImplemented



if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visual Checklist for nn_UNet", description="Applies the specified perturbations to a dataset, feeds them to the nn_UNet model and reports the results on a established property.")
    parser.add_argument("-i", "--input", help="The path to the input data to perturb and feed the model")
    parser.add_argument("-o", "--output", help="The path to save the output of the model for the original and perturbed data.")
    parser.add_argument("-l", "--perturb_labels", help="A flag that establishes that labels should be perturbed too.", action="store_true")
    args = parser.parse_args()

    input_folder_og = args.input
    input_folder_og = os.path.abspath(input_folder_og)
    images = vc.load_images(input_folder_og) # load a batch of samples
    
    perturb(images=images) # perturb each of them with the same configuration (manually chosen once, or based on config file)

    # predict original
    output_folder_og = os.path.abspath(args.output) # would be nice to also input this with an argument
    vc.predict(input_folder_og, output_folder_og)
    
    # predict perturbed
    input_folder_perturbed = os.path.join(input_folder_og, "perturbed")
    output_folder_perturbed = os.path.join(output_folder_og, "perturbed")

    folders = [folder for folder in os.listdir(input_folder_perturbed) if not os.path.isfile(os.path.join(input_folder_perturbed, folder))]

    for folder in folders:
        input_folder_perturbed_path = os.path.join(input_folder_perturbed, folder)
        output_folder_perturbed_path = os.path.join(output_folder_perturbed, folder)
        vc.predict(input_folder_perturbed_path, output_folder_perturbed_path)

    # evaluate each of these against true labels
    labels = os.path.join(os.path.dirname(input_folder_og), "labels")

    if args.perturb_labels:
        labels_files = vc.load_images(labels)
        perturb(images=labels_files)
        labels_perturbed = os.path.join(labels, "perturbed")
    else:
        labels_perturbed = labels

    ## original 
    ### (assuming nnunet and BraTS)
    vc.evaluate_folder(folder_with_gts=labels, folder_with_predictions=output_folder_og, labels = (0,1,2,4))
    
    ## perturbed 
    ### (assuming nnunet and BraTS)
    
    vc.write_report(output_folder_perturbed, labels_perturbed, output_folder_og)