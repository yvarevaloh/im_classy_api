import os
from random import sample

import numpy as np
import skimage.io as image_io
from imutils import paths as impaths
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

from image import Image
from settings import DATASET_FOLDERS


def sample_image_paths(dataset_used_real_path, sample_size=None):
    position = []
    imagePaths = []
    labels = []
    list_files = sorted(os.listdir(dataset_used_real_path))

    for j in range(len(list_files)):

        user_imagePaths = []
        user_position = []
        sample_user = list(impaths.list_images(os.path.sep.join([dataset_used_real_path, list_files[j]])))
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


class ImagesSet:

    lbp_load_functions = {
        "histogram": np.loadtxt,
        "image": image_io.imread
    }
    dataset_folders = ("real", "digital fake", "printed fake")
    classes = None

    def __init__(self, folder_path):
        super().__init__()

    def save_data_and_labels(self, location, data, labels, data_name, labels_name):
        joblib.dump(data, os.path.join(location, data_name))
        joblib.dump(labels, os.path.join(location, labels_name))

    @staticmethod
    def get_images_from_path(images_paths):
        if isinstance(images_paths, str):
            images_paths = [images_paths]
        return[image_io.imread(path) for path in images_paths]

    @classmethod
    def get_lbp_feature(cls, images, lbp_type, images_paths=None, lbp=None):
        features = []
        if images_paths:
            images.extend(cls.get_images_from_path(images_paths))
        for image in images:
            lbp_image = Image(lbp)
            feature = lbp_image.get_lbp_image(image)
            if lbp_type == "histogram":
                feature = lbp_image.get_lbp_histogram(feature)
            features.append(feature)
        return features

    @classmethod
    def get_encoded_labels(cls, labels):
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        cls.classes = le.classes_
        return encoded_labels

    @classmethod
    def get_image_paths_and_labels(cls, dataset_used_path, IMAGES_TO_PROCESS=60):
        image_paths = []
        data_labels = []
        print("[INFO] loading %s images paths" % IMAGES_TO_PROCESS)
        for folder in DATASET_FOLDERS:
            print("extracting data for %s" % folder)
            folder_path = os.path.sep.join([dataset_used_path, folder])
            img_paths, position, labels = sample_image_paths(folder_path, IMAGES_TO_PROCESS)
            image_paths.extend(img_paths)
            data_labels.extend(labels)
        print("Extraction finished")
        return image_paths, data_labels

    @classmethod
    def get_lbp_data_pickle(cls, location, data_name, labels_name):
        data = joblib.load(os.path.join(location, data_name))
        labels = joblib.load(os.path.join(location, labels_name))
        return data, labels

    def get_lbp_data_from_folder(self, folder_path, lbp_type, save_all=True):
        data = []
        labels = []
        for classification_folder in cls.dataset_folders:
            total_path = os.path.sep.join([folder_path, classification_folder])
            files_list = sorted(os.listdir(total_path))
            for folder in files_list:
                data_paths = sorted(os.listdir(os.path.sep.join([total_path, folder])))
                for data_path in data_paths:
                    lbp_data = self.lbp_load_functions[lbp_type](os.path.join(total_path, folder, data_path))
                    data.append(lbp_data)
                labels.extend(np.full((len(data_paths)), classification_folder))

        labels = self.get_encoded_labels(labels)
        print(f"Labels are:\t{self.classes}")  # ['digital fake' 'printed fake' 'real']
        return data, labels

# if __name__ == "__main__":
#     self.data, self.labels = self.get_lbp_data(folder_path)
