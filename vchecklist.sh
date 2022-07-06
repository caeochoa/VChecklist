#!/usr/bin/env bash
source /home/s2259310/.bashrc
conda activate dissertation

mkdir /disk/scratch/s2259310 /disk/scratch/s2259310/nnUNet_raw_data_base /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021

#zip -r data.zip nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
rsync -u /home/s2259310/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data/data.zip /disk/scratch/s2259310
unzip -u /disk/scratch/s2259310/data.zip -d /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021

python nn-UNet/convert_data.py /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021

export nnUNet_raw_data_base="/disk/scratch/s2259310/nnUNet_raw_data_base"
python vc.py -i /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/images -o /disk/scratch/s2259310/outputs/Task500_BraTS2021

zip -r /disk/scratch/s2259310/outputs.zip /disk/scratch/s2259310/outputs
unzip -u /home/s2259310/VChecklist/nn-UNet/outputs.zip -d /home/s2259310/VChecklist/nn-UNet/