#!/usr/bin/env bash
source /home/s2259310/.bashrc
conda activate dissertation

echo Creating directories...
rm -r /disk/scratch/s2259310/nnUNet_raw_data_base
mkdir /disk/scratch/s2259310 /disk/scratch/s2259310/nnUNet_raw_data_base /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
rm -r /disk/scratch/s2259310/outputs /disk/scratch/s2259310/outputs.zip 

echo Copying data...
#zip -r data.zip nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
cd nn-UNet
zip -ru models.zip nnUNet_trained_models
cd ..
rsync -u /home/s2259310/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data/data.zip /disk/scratch/s2259310
rsync -u /home/s2259310/VChecklist/ImageMods/configs/test.csv /disk/scratch/s2259310/test_config.csv
rsync -u /home/s2259310/VChecklist/nn-UNet/models.zip /disk/scratch/s2259310
echo Done!
echo Decompressing full data...
unzip -u /disk/scratch/s2259310/data.zip -d /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
unzip -u /disk/scratch/s2259310/models.zip -d /disk/scratch/s2259310/
echo Done!

echo Converting data...
python nn-UNet/convert_data.py /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021
echo Done!

echo Running survey_prep.py
export nnUNet_raw_data_base="/disk/scratch/s2259310/nnUNet_raw_data_base"
export RESULTS_FOLDER="/disk/scratch/s2259310/nnUNet_trained_models"
python survey_prep.py -i /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/images -o /disk/scratch/s2259310/outputs/Task500_BraTS2021
echo Done!!!! 

echo Compressing outputs and copying them to home directory
cd /disk/scratch/s2259310/
zip -r outputs.zip outputs
cd /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/
zip -r input_images.zip images labels
rsync /disk/scratch/s2259310/outputs.zip /home/s2259310/VChecklist/nn-UNet/
rsync /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/input_images.zip /home/s2259310/VChecklist/nn-UNet/
mkdir /home/s2259310/VChecklist/nn-UNet/outputs/survey_prep
unzip /home/s2259310/VChecklist/nn-UNet/outputs.zip -d /home/s2259310/VChecklist/nn-UNet/survey_prep
unzip /home/s2259310/VChecklist/nn-UNet/input_images.zip -d /home/s2259310/VChecklist/nn-UNet/outputs/survey_prep

echo All done!! Good job!!
