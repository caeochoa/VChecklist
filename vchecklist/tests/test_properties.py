from ExperimentBuilder import utils
import numpy as np


def agrees_Property(eval_og, eval_perturbed, labels, criteria):
    '''For each t ∈ T' the behaviour of M on t agrees with the behaviour of M on u ∈ U_t in 95% of cases.'''

    # first compute the difference in the specified criteria between each original and perturbed sample per label 
    eval_difference = []
    for sample in range(len(eval_og["results"]["all"])):
        eval_difference.append([utils.compare_sample(eval_og, eval_perturbed, sample, label, criteria) for label in labels])
    
    # make the differences absolute
    eval_difference = np.abs(np.array(eval_difference))
    #avg_diff = np.abs(np.mean(eval_difference, axis=1))
    
    # now use this difference to compute the percentage of samples that agree per label
    similarity = np.zeros(len(labels))
    for i in range(eval_difference.shape[1]):
        label_diff = eval_difference[:,i]
        similarity[i] = label_diff[label_diff <= 0.1].shape[0]/label_diff.shape[0]*100
    
    similarity = np.expand_dims(similarity, 0)
    means = np.expand_dims(np.array([utils.compare_mean(eval_og, eval_perturbed, label, criteria) for label in labels]))
    similarity = np.append(similarity, means, 0)

    return similarity