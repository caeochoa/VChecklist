import numpy as np
from scipy.ndimage import zoom

class TestType():

    def __init__(self) -> None:

        pass
    
    def apply(self, patches, probability, **kwargs):

        select = self.patch_selection(patches, probability)

        perturbed = patches.copy()

        for i in range(select.shape[0]):
            for j in range(select.shape[1]):
                for k in range(select.shape[2]):
                    if select[i,j,k]:
                        perturbed[i,j,k] = self.perturbation(perturbed[i,j,k], kwargs)

        self.perturbed = perturbed
    
    def patch_selection(self):
        pass

    def perturbation(self):
        pass


def rotation(img:np.ndarray, **kwargs):
    return np.rot90(img, kwargs)

def occlude(img:np.ndarray, **kwargs):
    return np.zeros(img.shape)

def blurr(img, k):
    return zoom(zoom(img, (1/k, 1/k, 1/k)), (k,k,k))

# no shuffle for now

def intensity(img:np.ndarray, k):
    
    if np.any(img != 0):
        # normalize
        max = img[img!=0].max()
        img /= max

        img[img!=0]+=k/100
        img[img>1]=1

    return img
    