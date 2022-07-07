#!/usr/bin/env bash
source /home/s2259310/.bashrc
conda activate dissertation

echo Compressing and copying all data
#zip -ru BraTS2021_00495.zip nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/BraTS2021_00495/imagesTs
mkdir /disk/scratch/s2259310 /disk/scratch/s2259310/nnUNet_raw_data_base /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
rsync -u nn-UNet/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/data/data.zip /disk/scratch/s2259310
unzip -u /disk/scratch/s2259310/data.zip -d /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021
echo Done!

echo Compressing and copying trained models
cd nn-UNet
zip -ru models.zip nnUNet_trained_models
cd ..
rsync -u /home/s2259310/VChecklist/nn-UNet/models.zip /disk/scratch/s2259310
unzip -u /disk/scratch/s2259310/models.zip -d /disk/scratch/s2259310/
echo Done!

echo Converting data...
python nn-UNet/convert_data.py /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021
echo Done!

echo Testing original shell script
export nnUNet_raw_data_base="/disk/scratch/s2259310/nnUNet_raw_data_base"
export RESULTS_FOLDER="/disk/scratch/s2259310/nnUNet_trained_models"
nnUNet_predict -i /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task500_BraTS2021/images -o /disk/scratch/s2259310/outputs/Task500_BraTS2021/original_script -t 1 -m 3d_fullres

echo Testing Predict function
python tests.py -t PredictTest -i /disk/scratch/s2259310/nnUNet_raw_data_base/nnUNet_raw_data/Task501_BraTS2021/images -o /disk/scratch/s2259310/outputs/Task500_BraTS2021/PredictTest

echo All done!
