import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def pad_img(img:np.ndarray, patch_shape:int, axis):
    #assert len(img.shape) == len(patch_shape), f"Patch shape {patch_shape}, doesn't fit number of dimensions ({len(img.shape)})"
    mod = img.shape[axis]%patch_shape
    shape = list(img.shape)
    shape[axis] = (patch_shape - mod)//2
    pad_img = np.append(np.zeros(shape), img, axis)
    #assert np.all(pad_img[shape[0]:, :, :] == img), f"{axis}: {pad_img[shape[0]:, :, :].shape}, {img.shape}"
    shape[axis] = (patch_shape - mod)//2 + (patch_shape - mod)%2
    pad_img = np.append(pad_img, np.zeros(shape), axis)
    return pad_img

def split_into_patches(img:np.ndarray, patch_shape:tuple):
    assert len(img.shape) == len(patch_shape), f"Patch shape {patch_shape}, doesn't fit number of dimensions ({len(img.shape)})"
    patchnum = []
    padimg = img.copy()
    for dim in range(len(img.shape)):
        padimg = pad_img(padimg, patch_shape[dim], dim)
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

def CentralPatchSelection(patches:np.ndarray, probability:float):
    selection = np.random.rand(patches.shape[0]//2,patches.shape[1]//2,patches.shape[2]//2)#, patches.shape[3], patches.shape[4], patches.shape[5])
    selection = selection <= probability
    for axis in range(len(selection.shape)):
        selection = pad_selection(patches, selection, axis)
    #selection = np.expand_dims(selection, 3)
    #selection = np.expand_dims(selection, 3)
    #selection = np.expand_dims(selection, 3)


    return selection.astype(np.bool8)

def OuterPatchSelection(patches:np.ndarray, probability:float):
    selection = np.random.rand(patches.shape[0],patches.shape[1],patches.shape[2])
    selection = selection <= probability
    centre = CentralPatchSelection(patches, 1) 
    selection[centre] = 0

    return selection

