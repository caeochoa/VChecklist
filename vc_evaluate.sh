#!/usr/bin/env bash
source /home/s2259310/.bashrc
conda activate dissertation


echo Experiment: $1
echo Copying data...
rsync -u /home/s2259310/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data/data.zip /disk/scratch/s2259310
rsync -u /home/s2259310/VChecklist/exp_outputs/$1.zip /disk/scratch/s2259310
echo Done!

echo Decompressing data...
unzip -u /disk/scratch/s2259310/$1.zip -d /disk/scratch/s2259310
unzip -u /disk/scratch/s2259310/data.zip \*seg.nii.gz -d /disk/scratch/s2259310/$1
echo Done!

echo Converting data...
python nn-UNet/convert_data.py /disk/scratch/s2259310/$1/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021
echo Done!


echo Running vc.py evaluate
export nnUNet_raw_data_base="/disk/scratch/s2259310/nnUNet_raw_data_base"
export RESULTS_FOLDER="/disk/scratch/s2259310/nnUNet_trained_models"
python vchecklist/vc.py -i /disk/scratch/s2259310/$1/images -o /disk/scratch/s2259310/ -c /disk/scratch/s2259310/$1/tests.json --name $1 -l /disk/scratch/s2259310/$1/labels 
echo Done!!!! 


echo Copying report to home directory
cd /disk/scratch/s2259310/

rsync /disk/scratch/s2259310/$1/results.json /home/s2259310/VChecklist/exp_outputs/$1/results.json


echo All done!! Good job!!
