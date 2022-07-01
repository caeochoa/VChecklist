from image_processing import SampleImage3D

def select_files():
    path = "-"
    files = []
    while path != "":
        path = input("Input path for file:")
        files.append(path)
    return files[:-1]

files = select_files()

for file in files:
    image = SampleImage3D(file)
    image.random_rotation()
    image.save()

