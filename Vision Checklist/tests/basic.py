#import matplotlib.pyplot as plt
#import imageio
import numpy as np
#import nibabel as nib
from random import random
#from .convert_nifti import Nifti2Numpy, Numpy2Nifti
import os
import csv

class BasicTestType():

    def __init__(image:np.array) -> None:
        pass

def hi(x):
    return x*2

class RobustnessTestType(BasicTestType):

    def __init__(self, image: np.array, patch_rule:function, patch_select_rule:function, perturbations:list) -> None:
        super().__init__(image)
        self.patch_rule = patch_rule # shape-like... or some other way to write complex and unequal patch sizes
        self.patch_select_rule = patch_select_rule # boolean array?
        self.perturbations = perturbations #list of functions

        self.patches = patch_rule(image)

    def perturb(self):
        perturbed = [self.patch_select_rule(self.patches, perturbation) for perturbation in self.perturbations]

