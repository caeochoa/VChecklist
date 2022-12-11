from ExperimentBuilder.builder import ExperimentBuilder
import os
import argparse
from tests.test_properties import agrees_Property

def evaluate(labels_path):
    assert args.labels, "A path for labels is needed to evaluate"
    labels_path = os.path.abspath(args.labels)
    builder.evaluate(labels_path=labels_path, property=agrees_Property)

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Visual Checklist for nn_UNet", description="Applies the specified perturbations to a dataset, feeds them to the nn_UNet model and reports the results on a established property.")
    parser.add_argument("-i", "--input", help="The path to the input data to perturb and feed the model")
    parser.add_argument("-o", "--output", help="The path to save the output of the model for the original and perturbed data.")
    parser.add_argument("-c", "--config", help="Choose configuration file for perturbation.")
    parser.add_argument("-s", "--save", help="Save configuration file for perturbation.", action="store_true")
    #parser.add_argument("-l", "--perturb_labels", help="A flag that establishes that labels should be perturbed too.", action="store_true")
    parser.add_argument("--perturb", help="Just perform perturbations on images", action="store_true")
    parser.add_argument("--predict", help="Just perform predictions with nn-UNet", action="store_true")
    parser.add_argument("--evaluate", help="Just evaluate VC results of nn-UNet", action="store_true")
    parser.add_argument("--name", help="Name for experiment folder")
    parser.add_argument("--labels", help="Path to labels folder for evaluation")
    args = parser.parse_args()

    img_path = os.path.abspath(args.input)
    out_path = os.path.abspath(args.output)
    config_path = os.path.abspath(args.config) if args.config else None

    builder = ExperimentBuilder(img_folder=img_path, 
                                out_folder=out_path, 
                                tests_config=config_path, 
                                save_config=args.save, 
                                experiment_name=args.name)
    
    if args.perturb:
        builder.perturb()
    elif args.predict:
        builder.predict()
    elif args.evaluate:
        evaluate(args.labels)
    else:
        builder.perturb()
        builder.predict()
        evaluate(args.labels)