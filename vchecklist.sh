#!/usr/bin/env bash
source /home/s2259310/.bashrc
conda activate dissertation

echo Creating directories...
mkdir /disk/scratch/s2259310 /disk/scratch/s2259310/nnUNet_raw_data_base /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021

echo Copying data...
#zip -r data.zip nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
rsync -u /home/s2259310/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data/data.zip /disk/scratch/s2259310
rsync -u /home/s2259310/VChecklist/ImageMods/configs/test.csv /disk/scratch/s2259310/test_config.csv
echo Done!
echo Exporting data...
unzip -u /disk/scratch/s2259310/data.zip -d /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
echo Done!

echo Converting data...
python nn-UNet/convert_data.py /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021
echo Done!

echo Running vc.py
export nnUNet_raw_data_base="/disk/scratch/s2259310/nnUNet_raw_data_base"
python vc.py -i /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/images -o /disk/scratch/s2259310/outputs/Task500_BraTS2021 -c /disk/scratch/s2259310/test_config.csv
echo Done!!!! 

echo Compressing outputs and copying them to home directory
zip -r /disk/scratch/s2259310/outputs.zip /disk/scratch/s2259310/outputs
rsync /disk/scratch/s2259310/outputs.zip /home/s2259310/VChecklist/nn-UNet/
unzip -u /home/s2259310/VChecklist/nn-UNet/outputs.zip -d /home/s2259310/VChecklist/nn-UNet/

echo All done!! Good job!!
