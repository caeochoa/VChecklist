import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from . import convert_nifti
import os
import nibabel as nib

class SampleImage():

    def __init__(self, img_path) -> None:
        
        self.img_path = os.path.abspath(img_path)
        self.filename = os.path.split(self.img_path)[1]
        self.img, self.affine, self.header = convert_nifti.Nifti2Numpy(self.img_path) 
        
    def show_slices_multiple(self):
        show_slices_multiple([central_slices(self.img)])
        plt.show()
    
    def split_into_patches(self, patch_shape:tuple):
        self.patch_shape = patch_shape
        assert len(patch_shape) == len(self.img.shape), "Wrong number of dimensions for patch"
        self.patches = split_into_patches(self.img, patch_shape)

    def paste_patches(self):
        self.out_img = paste_patches(self.patches)
    
    def save_img(self, out_path, out_img=None):
        if not out_img:
            out_img = self.out_img
        nifti = convert_nifti.Numpy2Nifti(out_img, self.affine, self.header)
        print("Saving ", self.filename, " in ",out_path)
        nib.save(nifti, os.path.join(out_path, self.filename))
        
class SampleImages():

    def __init__(self, img_paths:list) -> None:
        self.img_paths = [os.path.abspath(p) for p in img_paths]
        self.imgs = []
        for img_path in self.img_paths:
            self.imgs.append(SampleImage(img_path=img_path))
    
    def show_slices_multiple(self):
        imgs = [i.img for i in self.imgs]
        show_slices_multiple([central_slices(img) for img in imgs])
        plt.show()
    
    def split_into_patches(self, patch_shape:tuple):
        self.patch_shape = patch_shape
        self.patches = []
        for img in self.imgs:
            img.split_into_patches(patch_shape)
            self.patches.append(img.patches)
    
    def paste_patches(self):
        for i, img in enumerate(self.imgs):
            img.patches = self.patches[i]
            img.paste_patches()
    
    def save_imgs(self, out_path):
        for img in self.imgs:
            img.save_img(out_path)

def show_slices_multiple(list_of_slices):
    fig, axes = plt.subplots(1, len(list_of_slices[0]), figsize=(10,5))
    assert len(list_of_slices) <= 3, "Too many slices"
    cmap = ["gray", "plasma", "viridis"]
    for j, slices in enumerate(list_of_slices):
        for i, slice in enumerate(slices):
            if j == 0:
                axes[i].imshow(slice.T, cmap=cmap[j], origin="lower")#, alpha=alpha)
                axes[i].axis("off")
            else:
                alpha = slice.T > 0
                alpha = alpha.astype(np.uint8)
                im = axes[i].imshow(slice.T, cmap=cmap[j], origin="lower", alpha=alpha)
                values = np.unique(slice)[1:]
                colors = [ im.cmap(im.norm(value)) for value in values]
                labels = {"0": "Everything else", "1": "NCR", "2": "ED", "4": "ET", "3":"ET"}
                patches = [ mpatches.Patch(color=colors[i], label=labels[str(values[i])] ) for i in range(len(values)) ]
                plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
                
    fig.subplots_adjust(wspace=0, hspace=0)
    fig.tight_layout()
    
def central_slices(img:np.ndarray):
    assert len(img.shape) == 3, "Image should have 3 dimensions"
    s = img.shape
    return [img[s[0]//2, :, :], img[:, s[1]//2, :], img[:, :, s[2]//2]]


def pad_img(img_shape:tuple, patch_shape:tuple):

    pad = []
    for axis in range(len(img_shape)):
        mod = img_shape[axis]%patch_shape[axis]
        pad.append([(patch_shape[axis] - mod)//2,(patch_shape[axis] - mod)//2 + (patch_shape[axis] - mod)%2])

    return pad

def split_into_patches(img:np.ndarray, patch_shape:tuple):
    assert len(img.shape) == len(patch_shape), f"Patch shape {patch_shape}, doesn't fit number of dimensions ({len(img.shape)})"
    patchnum = []
    padimg = np.pad(img, pad_img(img.shape, patch_shape))
    for dim in range(len(img.shape)):
        assert padimg.shape[dim] % patch_shape[dim] == 0, f"{padimg.shape},{patch_shape},{dim}"
        patchnum.append(padimg.shape[dim]//patch_shape[dim])
    patches = np.zeros(tuple(patchnum) + patch_shape)
    
    for i in range(patchnum[0]):
        for j in range(patchnum[1]):
            for k in range(patchnum[2]):
                patches[i,j,k] = padimg[i*patch_shape[0]:(i+1)*patch_shape[0], j*patch_shape[1]:(j+1)*patch_shape[1], k*patch_shape[2]:(k+1)*patch_shape[2]]

    return patches

def paste_patches(patches:np.ndarray):
    dims = len(patches.shape)//2
    img = np.zeros([patches.shape[i+dims]*patches.shape[i] for i in range(dims)])
    print(img.shape)

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            for k in range(patches.shape[2]):
                img[i*patches.shape[3]:(i+1)*patches.shape[3], j*patches.shape[4]:(j+1)*patches.shape[4], k*patches.shape[5]:(k+1)*patches.shape[5]] = patches[i,j,k]
    
    return img

def pad_selection(patches:np.ndarray, selection:np.ndarray, axis, value=0):
    #assert len(img.shape) == len(patch_shape), f"Patch shape {patch_shape}, doesn't fit number of dimensions ({len(img.shape)})"
    rem = patches.shape[axis]-selection.shape[axis]
    shape = list(selection.shape)
    shape[axis] = rem//2
    selection = np.append(np.full(shape, value), selection, axis)
    shape[axis] = rem//2 + rem%2
    selection = np.append(selection, np.full(shape, value), axis)
    return selection.astype(np.bool8)

def patch_selection_intersection(select:np.ndarray, probability:float):
    new_select = np.zeros((select.shape[0],select.shape[1],select.shape[2]), np.bool8)
    selection = np.random.rand(select.shape[0],select.shape[1],select.shape[2])
    selection = selection <= probability
    for i in range(select.shape[0]):
        for j in range(select.shape[1]):
            for k in range(select.shape[2]):
                if np.any(select[i,j,k]) and selection[i,j,k]:
                    new_select[i,j,k] = True

    return new_select
    
def Central_PatchSelection(patches:np.ndarray, probability:float):
    selection = np.random.rand(patches.shape[0]//2,patches.shape[1]//2,patches.shape[2]//2)#, patches.shape[3], patches.shape[4], patches.shape[5])
    selection = selection <= probability
    for axis in range(len(selection.shape)):
        selection = pad_selection(patches, selection, axis)
    #selection = np.expand_dims(selection, 3)
    #selection = np.expand_dims(selection, 3)
    #selection = np.expand_dims(selection, 3)


    return selection.astype(np.bool8)

def Outer_PatchSelection(patches:np.ndarray, probability:float):
    selection = np.random.rand(patches.shape[0],patches.shape[1],patches.shape[2])
    selection = selection <= probability
    centre = Central_PatchSelection(patches, 1) 
    selection[centre] = 0

    return selection.astype(np.bool8)

def Random_PatchSelection(patches:np.ndarray, probability:float):
    select = np.random.rand(patches.shape[0], patches.shape[1], patches.shape[2])
    select = select <= probability
    return select.astype(np.bool8)

def InsideManual_PatchSelection(patches:np.ndarray, probability:float, manual_path:str):
    with open(os.path.abspath(manual_path), mode="rb") as file:
        select = np.load(file)
        select = split_into_patches(select, patches.shape[3:])
        select = patch_selection_intersection(select, probability)
    
    return select.astype(np.bool8)
def OutsideManual_PatchSelection(patches:np.ndarray, probability:float, manual_path:str):
    with open(os.path.abspath(manual_path), mode="rb") as file:
        select = np.load(file)
        select = np.invert(select.astype(np.bool8))
        select = split_into_patches(select, patches.shape[3:])
        select = patch_selection_intersection(select, probability)
    
    return select.astype(np.bool8)