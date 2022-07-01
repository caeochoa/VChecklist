# load nifti with nibabel
# transform nifti to numpy array
# apply perturbations to numpy array
# transfrom numpy array back to nifti (with og affine transformation?) and then save

import nibabel as nib
import numpy as np

def Nifti2Numpy(file):
    nifti = nib.load(file)
    array = np.array(nifti.dataobj)

    return array, nifti.affine, nifti.header

def Numpy2Nifti(array, affine, header=None):
    nifti = nib.Nifti1Image(array, affine, header)

    return nifti