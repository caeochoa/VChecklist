from ExperimentBuilder import utils
import numpy as np


def agrees_Property(eval_og, eval_perturbed, labels, criteria):

    eval_difference = []
    for sample in range(len(eval_og["results"]["all"])):
        eval_difference.append([utils.compare_sample(eval_og, eval_perturbed, sample, label, criteria) for label in labels])
    
    eval_difference = np.abs(np.array(eval_difference))
    #avg_diff = np.abs(np.mean(eval_difference, axis=1))
    similarity = np.zeros(len(labels))
    for i in range(eval_difference.shape[1]):
        label_diff = eval_difference[:,i]
        similarity[i] = label_diff[label_diff <= 0.1].shape[0]/label_diff.shape[0]*100
    
    similarity = np.expand_dims(similarity, 0)
    similarity = np.append(similarity, [utils.compare_mean(eval_og, eval_perturbed, label, criteria) for label in labels], 0)

    return similarity