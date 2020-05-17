import os
import unittest

import joblib
from sklearn import metrics
from sklearn.model_selection import train_test_split

from images_set import ImagesSet
from liveness_detection_classifier import LivenessDetectionClassifier
from settings import DATASET_PATH
from settings import MIN_MODEL_ACCURACY


class TestLivenessDetectionClassifier(unittest.TestCase):
    def setUp(self):
        dataset_path = DATASET_PATH
        self.data, self.labels = ImagesSet.get_image_paths_and_labels(dataset_path)
        # self.model = joblib.load(r"trained_models\svm_model_256.pkl")
        self.l_detection = LivenessDetectionClassifier(self.data, self.labels)

    def test_SVM_accuracy(self):
        f"""Validación con un error de {MIN_MODEL_ACCURACY} o menos."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.data, self.labels, test_size=0.25, random_state=42, stratify=self.labels)
        classification = self.l_detection.classify(X_test)
        class_report = metrics.classification_report(classification, y_test, output_dict=True)
        self.assertGreaterEqual(class_report["accuracy"], MIN_MODEL_ACCURACY)

    def test_SVM_accuracy_1(self):
        """Full accuracy and roc_auc when all data is used"""
        classification = self.l_detection.classify(self.data)
        class_report = metrics.classification_report(classification, self.labels, output_dict=True)
        # roc_auc = metrics.roc_auc_score(classification, self.labels[:5])
        self.assertAlmostEquals(class_report["accuracy"], 1)
