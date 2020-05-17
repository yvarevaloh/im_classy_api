from sklearn import svm
from images_set import ImagesSet

from settings import SVM_C, SVM_KERNEL, SVM_LBP_TYPE

from numpy import ndarray


class LivenessDetectionClassifier:
    """
    Train and classify the model.
    """
    number_to_detect_type = {0: 'digital fake', 1: 'printed fake', 2: 'real'}

    def __init__(self, X_data=None, Y_vector=None, c=SVM_C, kernel=SVM_KERNEL, model=None):
        super().__init__()
        self.c = c
        self.kernel = kernel
        self._model = model
        self.lbp_type = SVM_LBP_TYPE
        self.X_data = None if X_data is None else ImagesSet.get_lbp_feature(X_data, self.lbp_type)
        self.Y_vector = None if Y_vector is None else ImagesSet.get_encoded_labels(Y_vector)
        if Y_vector is not None:
            LivenessDetectionClassifier.number_to_detect_type = {
                index: value for index, value in enumerate(ImagesSet.classes)}

    @classmethod
    def cast_classification(cls, classification):
        # if isinstance(classification, ndarray):
        #     classification = classification[0]
        return [cls.number_to_detect_type[element] for element in classification]

    def classify(self, data_to_classify):
        if self._model is None:
            self._model = self.fit_model(self.X_data, self.Y_vector)
        prediction = self._model.predict(ImagesSet.get_lbp_feature(data_to_classify, self.lbp_type))
        return self.cast_classification(prediction)

    def fit_model(self, X_data, Y_vector):
        clf = svm.SVC(C=self.c, kernel=self.kernel)
        clf.fit(X_data, Y_vector)
        return clf
