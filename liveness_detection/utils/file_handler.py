from imutils import paths
import os
from random import sample

from settings import DATASET_FOLDERS


def sample_image_paths(dataset_used_real_path, sample_size=None):
    position = []
    imagePaths = []
    labels = []
    list_files = sorted(os.listdir(dataset_used_real_path))

    for j in range(len(list_files)):

        user_imagePaths = []
        user_position = []
        sample_user = list(paths.list_images(os.path.sep.join([dataset_used_real_path, list_files[j]])))
        if sample_size:
            user_imagePaths.append(sample(sample_user, sample_size))
        else:
            user_imagePaths.append(sample_user)

        for i in user_imagePaths[0]:
            user_position.append(i.split(os.path.sep)[-1].split('f')[-1].split('.')[0])

        imagePaths.append(user_imagePaths[0])
        position.append(user_position)
    imagePaths = [t for l in imagePaths for t in l]

    for i in imagePaths:
        labels.append(i.split(os.path.sep)[-3])

    return (imagePaths, position, labels)


def get_image_paths_and_labels(dataset_used_path, IMAGES_TO_PROCESS=60):
    image_paths = []
    data_labels = []
    print("[INFO] loading first %s real images" % IMAGES_TO_PROCESS)
    for folder in dataset_folders:
        print("extracting data for %s" % folder)
        folder_path = os.path.sep.join([dataset_used_path, folder])
        img_paths, position, labels = sample_image_paths(folder_path, IMAGES_TO_PROCESS)
        image_paths.extend(img_paths)
        data_labels.extend(labels)
    print("Extraction finished")
    return image_paths, data_labels


def create_new_folders(dataset_path, destiny_path):
    print("Creating new Folders")
    for folder in dataset_folders:
        model_files = sorted(os.listdir(os.path.sep.join([dataset_path, folder])))
        for model_file in model_files:
            new_folder = os.path.sep.join([destiny_path, folder, model_file])
            try:
                os.makedirs(new_folder)
                print(f"folder {new_folder} created.")
            except Exception as e:
                print(e)
