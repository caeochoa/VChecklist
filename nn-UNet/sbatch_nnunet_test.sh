#!/usr/bin/env bash
source /home/s2259310/.bashrc
conda activate dissertation
mkdir /disk/scratch/s2259310
zip -r data.zip nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
rsync -u /home/s2259310/VChecklist/nn-UNet/data.zip /disk/scratch/s2259310
unzip -u /disk/scratch/s2259310/data.zip -d /disk/scratch/s2259310
export nnUNet_raw_data_base="/disk/scratch/s2259310/nnUNet_raw_data_base"
nnUNet_predict -i /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/imagesTs -o /disk/scratch/s2259310/outputs -t 1 -m 3d_fullres
zip -r /disk/scratch/s2259310/outputs.zip /disk/scratch/s2259310/outputs
rsync -u /disk/scratch/s2259310/outputs.zip /home/s2259310/VChecklist/nn-UNet/outputs.zip
mkdir /home/s2259310/VChecklist/nn-UNet/outputs/batch
unzip /home/s2259310/VChecklist/nn-UNet/outputs.zip -d /home/s2259310/VChecklist/nn-UNet/outputs/batch
 
