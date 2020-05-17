import os

from sklearn.externals import joblib

#from images_set import ImagesSet
from liveness_detection_classifier import LivenessDetectionClassifier

# if __name__ == "__main__":
model = joblib.load(r"trained_models\svm_model_256.pkl")
lbp_data_path = r"data"
data = joblib.load(os.path.join(lbp_data_path, "lbp_histograms_256.pkl"))
labels = joblib.load(os.path.join(lbp_data_path, "lbp_labels_256.pkl"))
l_detection = LivenessDetectionClassifier(model=model)

dataset_path = r"data\20200318-174041"
#data, labels = ImagesSet.get_image_paths_and_labels(dataset_path)

image_path = r"data\test"
image_names = ("1r.jpg", "2r.jpg", "3p.png", "4p.png", "5d.png", "6d.png", "7d.jpg")
# for image_name in image_names:
#     classification = l_detection.classify(os.path.join(image_path, image_name), data, labels)
#     print(image_name, classification)
classification = l_detection.classify([os.path.join(image_path, image_name)
                                       for image_name in image_names])
print(classification)
