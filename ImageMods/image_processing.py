from types import NoneType
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
import numpy as np
import nibabel as nib
from random import random
from convert_nifti import Nifti2Numpy, Numpy2Nifti
import os


class SampleImage():

    def __init__(self, image_path) -> None:
        self.ogimage = imageio.imread(image_path)
        self.image = self.ogimage.copy()
        self.IMAGE_SIZE = self.image.shape[0]
        

    def crop_image(self, dims:tuple|list):
        
        image = self.image

        h, w = image.shape[:2]
        assert dims < (h,w), "New dimensions have to be smaller for cropping to be possible"

        print(f"Cropping sample image from ({h},{w}) to {dims}")
        self.image = image[(h-dims[0])//2:h-(h-dims[0])//2, (w-dims[1])//2:w-(w-dims[1])//2]
        self.IMAGE_SIZE = self.image.shape[0]

        return 

    def square_image(self):

        image = self.image

        if image.shape[0] > image.shape[1]:
            self.crop_image((image.shape[1], image.shape[1]))
        elif image.shape[0] < image.shape[1]:
            self.crop_image((image.shape[0], image.shape[0]))
        
        print(f"Image changed from {image.shape} to {self.image.shape}")
        
        #self.image = squared
        assert self.image.shape[0] == self.image.shape[1], "Error: Image wasn't squared correctly"
        self.IMAGE_SIZE = self.image.shape[0]

        return
    
    def make_patches(self, PATCH_SIZE:int):

        self.PATCH_SIZE = PATCH_SIZE
        self.PATCH_PER_SIDE = self.IMAGE_SIZE//PATCH_SIZE
        new_size = self.PATCH_PER_SIDE*PATCH_SIZE

        self.square_image()

        self.crop_image((new_size, new_size))

        if len(self.image.shape) < 3:
            self.image = np.expand_dims(self.image, 2)

        assert len(self.image.shape) == 3, f"Image doesn't have right dimensions: {self.image.shape}"

        patches = np.zeros((self.PATCH_PER_SIDE, self.PATCH_PER_SIDE, PATCH_SIZE, PATCH_SIZE, self.image.shape[2]), dtype=int)
        for i,y in enumerate(range(0,self.IMAGE_SIZE, PATCH_SIZE)):
            for j,x in enumerate(range(0,self.IMAGE_SIZE,PATCH_SIZE)):
                patches[i,j] = self.image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

        print(f"Sample now contains {len(patches)} patches of {PATCH_SIZE}x{PATCH_SIZE}, at {self.PATCH_PER_SIDE} patches per side")

        self.patches = patches

        return
    
    def show_patches(self, patches=None, span:float=0, label:bool=False, save=False):

        if type(patches) == NoneType:
            patches = self.patches
        else:
            assert patches.shape == (self.PATCH_PER_SIDE, self.PATCH_PER_SIDE, self.PATCH_SIZE, self.PATCH_SIZE, self.image.shape[-1]), ""
            patches = patches

        fig, ax = plt.subplots(self.PATCH_PER_SIDE, self.PATCH_PER_SIDE, figsize=(5, 5))

        for idy in range(self.PATCH_PER_SIDE):
            for idx in range(self.PATCH_PER_SIDE):
                ax[idy, idx].imshow(patches[idy, idx])
                ax[idy, idx].axis('off')
                if label:
                    ax[idy, idx].text(0,25, f"{idy}, {idx}", color='red', weight='bold')


        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()


        plt.subplots_adjust(wspace=span, hspace=span)
        plt.show()

        if save:
            plt.savefig("SampleImage.png")
    
    def random_rotation(self, proportion:float = 0.5, k:int=1):

        print(f"Applying Random Rotation of {90*k}º to {proportion*100}% of patches (Size: {self.PATCH_SIZE})")

        rotated_patches = self.patches.copy()

        for idy in range(self.PATCH_PER_SIDE):
            for idx in range(self.PATCH_PER_SIDE):
                if random() <= proportion:
                    rotated_patches[idy, idx] = np.rot90(self.patches[idy, idx], k=k)
        
        return rotated_patches
    
    def central_rotation(self, proportion:float=0.5, k:int=1):

        print(f"Applying Central Rotation of {90*k}º to {proportion*100}% of central patches (Size: {self.PATCH_SIZE})")

        rotated_patches = self.patches.copy()

        for idy in range(self.PATCH_PER_SIDE):
            for idx in range(self.PATCH_PER_SIDE):
                if (self.PATCH_PER_SIDE-1)/4 <= idy < (self.PATCH_PER_SIDE-1)*3/4 and (self.PATCH_PER_SIDE-1)/4 <= idx < (self.PATCH_PER_SIDE-1)*3/4:
                    if random() <= proportion:
                        rotated_patches[idy, idx] = np.rot90(self.patches[idy, idx], k=k)

        return rotated_patches

    def outer_rotation(self, proportion:float=0.5, k:int=1):

        print(f"Applying Outer Rotation of {90*k}º to {proportion*100}% of outer patches (Size: {self.PATCH_SIZE})")

        rotated_patches = self.patches.copy()

        for idy in range(self.PATCH_PER_SIDE):
            for idx in range(self.PATCH_PER_SIDE):
                if not ((self.PATCH_PER_SIDE-1)/4 <= idy < (self.PATCH_PER_SIDE-1)*3/4 and (self.PATCH_PER_SIDE-1)/4 <= idx < (self.PATCH_PER_SIDE-1)*3/4):
                    if random() <= proportion:
                        rotated_patches[idy, idx] = np.rot90(self.patches[idy, idx], k=k)

        return rotated_patches

            
class SampleImage3D():

    def __init__(self, image_path) -> None:
        self.image_path = image_path
        self.data, self.affine, self.header = Nifti2Numpy(image_path)
        print(f"Image shape is {self.data.shape}")
        self.PATCH_SIZE = (20,20,31) # Assuming BRATS dataset with shape (240,240,155)
        print(f"Patches will have shape {self.PATCH_SIZE}")
        self.PATCH_PER_SIDE = np.array(self.data.shape)//np.array(self.PATCH_SIZE)
        self.perturbed = np.zeros(self.data.shape)
        self.perturbed = np.expand_dims(self.perturbed, 0)


    def show_slices(self, slices=None):
        """ Function to display row of image slices """
        # consider adding the option to easily choose other slices of self.data
        if type(slices) == NoneType:
            slices = [self.data[self.data.shape[0]//2,:,:], self.data[:,self.data.shape[0]//2,:], self.data[:,:,self.data.shape[0]//2]]
        elif type(slices) == np.ndarray:
            slices = [slices[slices.shape[0]//2,:,:], slices[:,slices.shape[0]//2,:], slices[:,:,slices.shape[0]//2]]
        # elif type(slices) == str:
        #    if slices == "perturbed":
        #        slices = 
        fig, axes = plt.subplots(1, len(slices), figsize=(5,5))
        if len(slices) == 1:
            axes.imshow(slices[0].T, cmap="gray", origin="lower")
        else:
            for i, slice in enumerate(slices):
                axes[i].imshow(slice.T, cmap="gray", origin="lower")
                axes[i].axis("off")
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.tight_layout()
        plt.suptitle(f"Center slices of image")
        plt.show()
    
    def random_rotation(self, proportion:float = 0.5, k:int=1):

        print(f"Applying Random Rotation of {90*k}º to {proportion*100}% of patches (Size: {self.PATCH_SIZE})")

        perturbed = self.data.copy()

        for i,x in enumerate(range(0, self.data.shape[0], self.PATCH_SIZE[0])):
            for j,y in enumerate(range(0,self.data.shape[1], self.PATCH_SIZE[1])):
                for k,z in enumerate(range(0,self.data.shape[2], self.PATCH_SIZE[2])):
                    if random() <= proportion:
                        perturbed[x:x+self.PATCH_SIZE[0], y:y+self.PATCH_SIZE[1], z:z+self.PATCH_SIZE[2]] = np.rot90(perturbed[x:x+self.PATCH_SIZE[0], y:y+self.PATCH_SIZE[1], z:z+self.PATCH_SIZE[2]], k=k)

        self.perturbed = np.append(self.perturbed, np.expand_dims(perturbed, 0), 0)
        
    
    def central_rotation(self, proportion:float=0.5, k:int=1):

        print(f"Applying Central Rotation of {90*k}º to {proportion*100}% of central patches (Size: {self.PATCH_SIZE})")

        perturbed = self.data.copy()

        for i,x in enumerate(range(0, self.data.shape[0], self.PATCH_SIZE[0])):
            for j,y in enumerate(range(0,self.data.shape[1], self.PATCH_SIZE[1])):
                for k,z in enumerate(range(0,self.data.shape[2], self.PATCH_SIZE[2])):
                    if (self.PATCH_PER_SIDE[0]-1)/4 <= i < (self.PATCH_PER_SIDE[0]-1)*3/4 and (self.PATCH_PER_SIDE[1]-1)/4 <= j < (self.PATCH_PER_SIDE[1]-1)*3/4 and (self.PATCH_PER_SIDE[2]-1)/4 <= k < (self.PATCH_PER_SIDE[2]-1)*3/4:
                        if random() <= proportion:
                            perturbed[x:x+self.PATCH_SIZE[0], y:y+self.PATCH_SIZE[1], z:z+self.PATCH_SIZE[2]] = np.rot90(perturbed[x:x+self.PATCH_SIZE[0], y:y+self.PATCH_SIZE[1], z:z+self.PATCH_SIZE[2]], k=k)

        self.perturbed = np.append(self.perturbed, np.expand_dims(perturbed, 0), 0)

    def outer_rotation(self, proportion:float=0.5, k:int=1):

        print(f"Applying Outer Rotation of {90*k}º to {proportion*100}% of outer patches (Size: {self.PATCH_SIZE})")

        perturbed = self.data.copy()

        for i,x in enumerate(range(0, self.data.shape[0], self.PATCH_SIZE[0])):
            for j,y in enumerate(range(0,self.data.shape[1], self.PATCH_SIZE[1])):
                for k,z in enumerate(range(0,self.data.shape[2], self.PATCH_SIZE[2])):
                    if not ((self.PATCH_PER_SIDE[0]-1)/4 <= i < (self.PATCH_PER_SIDE[0]-1)*3/4 and (self.PATCH_PER_SIDE[1]-1)/4 <= j < (self.PATCH_PER_SIDE[1]-1)*3/4 and (self.PATCH_PER_SIDE[2]-1)/4 <= k < (self.PATCH_PER_SIDE[2]-1)*3/4):
                        if random() <= proportion:
                            perturbed[x:x+self.PATCH_SIZE[0], y:y+self.PATCH_SIZE[1], z:z+self.PATCH_SIZE[2]] = np.rot90(perturbed[x:x+self.PATCH_SIZE[0], y:y+self.PATCH_SIZE[1], z:z+self.PATCH_SIZE[2]], k=k)

        self.perturbed = np.append(self.perturbed, np.expand_dims(perturbed, 0), 0)

    def save(self, path=None):
        if not path:
            path = "/".join(self.image_path.split("/")[:-1])
            filename = self.image_path.split("/")[-1].split(".")[0]
            
        path = os.path.join(path, "perturbed")

        try:
            print(f"Creating directory {path}")
            os.mkdir(path)
        except OSError:
            print("Directory already exists")

        for i, image in enumerate(self.perturbed[1:]):
            nifti = Numpy2Nifti(image, self.affine, self.header)
            nib.save(nifti, os.path.join(path, filename + "_perturbed_"+f"00{i}"))
