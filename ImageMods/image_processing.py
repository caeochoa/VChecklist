from types import NoneType
from scipy import ndimage
import matplotlib.pyplot as plt
import imageio
import numpy as np
from random import random


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

            
    