#!/usr/bin/env bash
source /home/s2259310/.bashrc
conda activate dissertation

echo Creating directories...
mkdir /disk/scratch/s2259310 /disk/scratch/s2259310/nnUNet_raw_data_base /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
rm -r /disk/scratch/s2259310/outputs /disk/scratch/s2259310/outputs.zip
echo Copying output folder structure...
if [ -d "nn-UNet/outputs" ]
then
    cd nn-UNet/outputs
    find outputs -type d > /disk/scratch/s2259310/output_dirs.txt
    cd /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
    xargs mkdir -p < /disk/scratch/s2259310/output_dirs.txt
    cd /disk/scratch/s2259310/
    mkdir outputs
    cd outputs
    xargs mkdir -p < /disk/scratch/s2259310/output_dirs.txt
    cd /home/s2259310/VChecklist
fi


echo Copying data...
#zip -r data.zip nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
cd nn-UNet
zip -ru models.zip nnUNet_trained_models
cd ..
rsync -u /home/s2259310/VChecklist/nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data/data.zip /disk/scratch/s2259310
rsync -u /home/s2259310/VChecklist/nn-UNet/models.zip /disk/scratch/s2259310
echo Done!
echo Decompressing data...
unzip -u /disk/scratch/s2259310/data.zip -d /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
unzip -u /disk/scratch/s2259310/models.zip -d /disk/scratch/s2259310/
echo Done!

echo Converting data...
python nn-UNet/convert_data.py /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021
echo Done!

echo Running nnunet-basic.py
export nnUNet_raw_data_base="/disk/scratch/s2259310/nnUNet_raw_data_base"
export RESULTS_FOLDER="/disk/scratch/s2259310/nnUNet_trained_models"
python nnunet-basic.py -i /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/images -o /disk/scratch/s2259310/outputs/Task500_BraTS2021
echo Done!!!! 

echo Compressing outputs and copying them to home directory
cd /disk/scratch/s2259310/
zip -r outputs.zip outputs
rsync /disk/scratch/s2259310/outputs.zip /home/s2259310/VChecklist/nn-UNet/
unzip -u /home/s2259310/VChecklist/nn-UNet/outputs.zip -d /home/s2259310/VChecklist/nn-UNet/

echo All done!! Good job!!