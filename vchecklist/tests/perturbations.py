import numpy as np
from scipy.ndimage import zoom
from . import patches
from json import JSONEncoder
from inspect import getmembers, isfunction
import sys
from ExperimentBuilder.utils import get_value, clean_dict
from os.path import abspath

class TestType():

    def __init__(self, **parameters) -> None:

        ''' parameters =  patch_selection:str=None, perturbation:str=None, probability:float=None, k=None, manual_path=None'''
        accepted_parameters =  ["patch_selection", "perturbation", "probability", "k", "manual_path"]
        self.__dict__ = clean_dict(parameters, accepted_parameters)
        self.accepted_parameters = accepted_parameters
        self.pert_rules = {name:obj for name,obj in getmembers(sys.modules["tests.perturbations"]) if isfunction(obj) and obj.__module__ == "tests.perturbations"}
        self.ps_rules = {name.split("_")[0]:obj for name,obj in getmembers(patches) if (isfunction(obj) and name.endswith("PatchSelection"))}

        #perturbation = get_value(parameters, "perturbation")
        #patch_selection = get_value(parameters, "patch_selection")
        if self.perturbation and self.patch_selection:
            assert self.perturbation in self.pert_rules.keys()
            assert self.patch_selection in self.ps_rules.keys()
        
        self.manual_path = abspath(self.manual_path) if self.manual_path else None
        

        '''self.patch_selection_function = patch_selection
        self.perturbation_function = perturbation
        self.probability = get_value(parameters, "probability")
        self.k = get_value(parameters, "k")
        self.manual_path = get_value(parameters, "manual_path") '''
    
    def apply(self, patches):

        select = self.__patch_selection__(patches[0])

        perturbed_images = patches.copy()

        for image in perturbed_images:
            for i in range(select.shape[0]):
                for j in range(select.shape[1]):
                    for k in range(select.shape[2]):
                        if select[i,j,k]:
                            image[i,j,k] = self.__perturbation__(image[i,j,k])

        return perturbed_images
    
    def __patch_selection__(self, patches):
        if self.patch_selection_function:
            if self.patch_selection_function == "InsideManual" or self.patch_selection_function == "OutsideManual":
                select = self.ps_rules[self.patch_selection_function](patches, self.probability, self.manual_path)
            else:
                select = self.ps_rules[self.patch_selection_function](patches, self.probability)
            return select
        else:
            raise NotImplementedError

    def __perturbation__(self, img):
        if self.perturbation_function:
            perturbed = self.pert_rules[self.perturbation_function](img, self.k)
            return perturbed
        else:
            raise NotImplementedError

class TestTypeJSONEncoder(JSONEncoder):
    def default(self, o):
        d = o.__dict__.copy()
        if "pert_rules" in d.keys():
            d.pop("pert_rules")
            d.pop("ps_rules")
            d.pop("accepted_parameters")
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
    
    