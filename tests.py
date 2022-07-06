from ImageMods.image_processing import SampleImage3D
def SampleImageTest():
    im = SampleImage3D("/Users/caeochoa/Documents/GUH20/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/BraTS2021_00495/imagesTs/BraTS2021_00495_0000.nii.gz")
    im.apply_config("ImageMods/configs/test.csv", save_config=True, nnunet=True)

if __name__ == "__main__":
    SampleImageTest()