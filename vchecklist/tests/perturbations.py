import numpy as np
from scipy.ndimage import zoom
from . import patches
from json import JSONEncoder
from inspect import getmembers, isfunction
import sys

class TestType():

    def __init__(self, patch_selection:str=None, perturbation:str=None) -> None:

        self.pert_rules = {name:obj for name,obj in getmembers(sys.modules["tests.perturbations"]) if isfunction(obj) and obj.__module__ == "tests.perturbations"}
        self.ps_rules = {name.split("_")[0]:obj for name,obj in getmembers(patches) if (isfunction(obj) and name.endswith("PatchSelection"))}

        if perturbation and patch_selection:
            assert perturbation in self.pert_rules.keys()
            assert patch_selection in self.ps_rules.keys()

        self.patch_selection_function = patch_selection
        self.perturbation_function = perturbation
    
    def apply(self, patches, probability, k, manual_path=None):

        select = self.patch_selection(patches[0], probability, manual_path)

        perturbed_images = patches.copy()

        for image in perturbed_images:
            for i in range(select.shape[0]):
                for j in range(select.shape[1]):
                    for k in range(select.shape[2]):
                        if select[i,j,k]:
                            image[i,j,k] = self.perturbation(image[i,j,k], k)

        return perturbed_images
    
    def patch_selection(self, patches, probability, manual_path=None):
        if self.patch_selection_function:
            if self.patch_selection_function == "Manual":
                select = self.ps_rules[self.patch_selection_function](patches, probability, manual_path)
            else:
                select = self.ps_rules[self.patch_selection_function](patches, probability)
            return select
        else:
            raise NotImplementedError

    def perturbation(self, img, k):
        if self.perturbation_function:
            perturbed = self.pert_rules[self.perturbation_function](img, k)
            return perturbed
        else:
            raise NotImplementedError

class TestTypeJSONEncoder(JSONEncoder):
    def default(self, o):
        d = o.__dict__.copy()
        if "pert_rules" in d.keys():
            d.pop("pert_rules")
            d.pop("ps_rules")
        return d

def rotation(img:np.ndarray, k=1):
    if k > 4:
        k = k // 90
    return np.rot90(img, k)

def occlude(img:np.ndarray, *args):
    return np.zeros(img.shape)

def blurr(img, k):
    return zoom(zoom(img, (1/k, 1/k, 1/k)), (k,k,k))

# no shuffle for now

def intensity(img:np.ndarray, k):
    
    if np.any(img != 0):
        max = img[img!=0].max()

        img[img!=0]+=k/100*max
        img[img>max]=max

    return img
    
    