import os

def try_mkdir(path, verbose=True):
    try:
        if verbose:
            print(f"Creating directory {path}")
        os.mkdir(path)
    except OSError:
        if verbose:
            print("Directory already exists")
    return os.path.abspath(path)

def listf(path, type):
    if type == "file":
        return [file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))]
    if type == "folder":
        return [file for file in os.listdir(path) if not os.path.isfile(os.path.join(path, file))]
    else:
        return ValueError

def compare_mean(dict1, dict2, label, criteria):
    return dict1["results"]["mean"][str(label)][criteria] - dict2["results"]["mean"][str(label)][criteria]

def compare_sample(dict1, dict2, sample, label, criteria):
    return dict1["results"]["all"][sample][str(label)][criteria] - dict2["results"]["mean"][str(label)][criteria]