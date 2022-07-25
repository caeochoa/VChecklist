import os, json, csv
import numpy as np

def compare_mean(dict1, dict2, label, criteria):
    return dict1["results"]["mean"][str(label)][criteria] - dict2["results"]["mean"][str(label)][criteria]

def compare_sample(dict1, dict2, sample, label, criteria):
    return dict1["results"]["all"][sample][str(label)][criteria] - dict2["results"]["mean"][str(label)][criteria]

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
        eval_difference.append([compare_sample(eval_og, eval_pert, sample, label, criteria) for label in (0,1,2,4)])
    
    eval_difference = np.array(eval_difference)
    avg_diff = np.abs(np.mean(eval_difference, axis=1))
    similarity = avg_diff[avg_diff <= 0.01].shape[0]/avg_diff.shape[0]*100
    
    # means = [np.mean(np.array([dic["results"]["mean"][str(label)][criteria] for label in (0,1,2,4)])) for dic in [eval_og, eval_pert]]
    mean_difference = np.mean(np.abs(np.array([compare_mean(eval_og, eval_pert, label, criteria) for label in (0,1,2,4)])))
    pert_mean_dice = np.mean(np.array([eval_pert["results"]["mean"][str(label)][criteria] for label in (0,1,2,4)]))
    og_mean_dice = np.mean(np.array([eval_og["results"]["mean"][str(label)][criteria] for label in (0,1,2,4)]))

    return similarity, mean_difference, pert_mean_dice, og_mean_dice

def write_report(output_folder_perturbed, output_folder_og):

    folders = [folder for folder in os.listdir(output_folder_perturbed) if not os.path.isfile(os.path.join(output_folder_perturbed, folder))]

    report = []
    for folder in folders:
        output_folder_perturbed_path = os.path.join(output_folder_perturbed, folder)
        
        # test definition of property [1] based on these evaluations
        similarity, mean_diff, pert_dice, og_dice = test_property1(output_folder_og, output_folder_perturbed_path, "Dice")
        result = {"name":folder, "similarity":similarity, "mean_diff": mean_diff, "pert_dice":pert_dice, "og_dice":og_dice}
        print(result)
        report.append(result)

    with open(os.path.join(output_folder_perturbed, f"property_results.csv"), mode="w") as csvfile:
        fields = list(report[0].keys()) #["name", "similarity", "mean_diff", "pert_dice", "og_dice"]
        writer = csv.DictWriter(csvfile, fields)
        writer.writeheader()
        for result in report:
            writer.writerow(result)