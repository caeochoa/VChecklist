import os
import argparse

def listf(dir, type):
    if type == "file":
        return [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
    elif type == "folder":
        return [f for f in os.listdir(dir) if not os.path.isfile(os.path.join(dir, f))]
    elif not type:
        return [f for f in os.listdir(dir)]

def try_mkdir(new_dir):
    new_dir = os.path.abspath(new_dir)

    try:
        print(f"Creating directory {new_dir}")
        os.mkdir(new_dir)
    except OSError:
        print("Directory already exists")
    
    return new_dir

if __name__ == "__main__":

    ## argparse
    parser = argparse.ArgumentParser(description="Converts data from BraTS_2021 organisation format to a nn_UNet readable formar")
    parser.add_argument("dir", help="Path to folder with the data")
    args = parser.parse_args()


    dir = args.dir

    dir = os.path.abspath(dir)

    images_path = try_mkdir(os.path.join(os.path.dirname(dir), "images"))
    labels_path = try_mkdir(os.path.join(os.path.dirname(dir), "labels"))

    for folder in listf(dir, "folder"):
        folder_path = os.path.join(dir, folder)
        for file in listf(folder_path, "file"):
            file_path = os.path.join(folder_path, file)
            if file.split(".")[0].split("_")[-1] == "flair":
                os.rename(file_path, os.path.join(images_path, "_".join(file.split(".")[0].split("_")[:2]) + "_0000.nii.gz"))
            elif file.split(".")[0].split("_")[-1] == "t1":
                os.rename(file_path, os.path.join(images_path, "_".join(file.split(".")[0].split("_")[:2]) + "_0001.nii.gz"))
            elif file.split(".")[0].split("_")[-1] == "t1ce":
                os.rename(file_path, os.path.join(images_path, "_".join(file.split(".")[0].split("_")[:2]) + "_0002.nii.gz"))
            elif file.split(".")[0].split("_")[-1] == "t2":
                os.rename(file_path, os.path.join(images_path, "_".join(file.split(".")[0].split("_")[:2]) + "_0003.nii.gz"))
            elif file.split(".")[0].split("_")[-1] == "seg":
                os.rename(file_path, os.path.join(labels_path, "_".join(file.split(".")[0].split("_")[:2]) + ".nii.gz"))
            else:
                print(f"{file} is not a sample file")
    
    # test that four files for each case have been created
    case_ids = []
    inputs = listf(images_path, "file")
    for file in inputs:
        case_ids.append(file.split("_")[1])
    for case in case_ids:
        for i in range(0,4):
            try:
                assert f"BraTS2021_{case}_{str(i).zfill(4)}.nii.gz" in inputs, f"Missing modality {str(i).zfill(4)} for case {case}"
            except AssertionError:
                continue

