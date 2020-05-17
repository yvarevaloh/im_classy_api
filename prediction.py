import io
from base64 import b64decode
from copy import deepcopy
from os.path import splitext

import joblib
from flask import jsonify
from flask_restful import Resource
from flask_restful import reqparse
import skimage.io as image_io

from constants import FRAMES_KEY
from constants import MAX_FRAMES
from constants import VALID_EXTENSIONS
from errors import ExceededFramesNumberError
from errors import ImageExtensionError
from errors import MissingFramesKeyError
from liveness_detection.liveness_detection_classifier import LivenessDetectionClassifier

mock_response = {
    "type": "",
    "score": 1.0,
    "predictions": {
        "real": 0.0,
        "printed": 0.0,
        "digital": 0.0
    },
    "version": "1.0",
    "responseCode": "200",
    "responseMessage": "Success"
}


class Classify(Resource):
    @staticmethod
    def create_response(**args):
        response = deepcopy(mock_response)
        response.update(**args)
        return response

    @staticmethod
    def get_image_request_errors(files):
        if FRAMES_KEY not in files:
            raise MissingFramesKeyError
        file_names = [file_.filename for file_ in files.getlist(FRAMES_KEY)]
        if len(file_names) > MAX_FRAMES:
            raise ExceededFramesNumberError
        if not all(map(lambda name: splitext(name)[-1].lower() in VALID_EXTENSIONS, file_names)):
            raise ImageExtensionError

    @staticmethod
    def get_binary_request_errors(binaries):
        if FRAMES_KEY not in binaries:
            raise MissingFramesKeyError
        file_names = [file_ for file_ in binaries.getlist(FRAMES_KEY)]
        if len(file_names) > MAX_FRAMES:
            raise ExceededFramesNumberError

    def get(self):
        return mock_response

    @staticmethod
    def validate_errors(content, type_):
        if type_ == "image":
            error = Classify.get_binary_request_errors(content)
        elif type_ == "binary":
            error = Classify.get_binary_request_errors(content)
        else:
            raise ImageExtensionError("error type_ should be image or binary")

    @staticmethod
    def classify(images):
        model = joblib.load(r"trained_models\svm_model_256.pkl")
        l_detection = LivenessDetectionClassifier(model=model)
        classification = l_detection.classify(images)
        return classification

    def post(self):
        images = []
        request = reqparse.request
        if request.data:
            "only one file"
            images.append(image_io.imread(io.BytesIO(request.data)))
            return mock_response
        if request.files:
            self.get_binary_request_errors(request.files)
            request_images = request.files.getlist(FRAMES_KEY)
            images.extend([image_io.imread(image) for image in request_images])
        if request.form:
            self.get_binary_request_errors(request.form)
            request_images = request.form.getlist(FRAMES_KEY)
            images.extend([image_io.imread(b64decode(image), plugin="imageio") for image in request_images])

        classification = self.classify(images)
        response = self.create_response(**{
            "type": classification[0],
            "responseMessage": "All images processed.",
            "imagesProcessed": len(images)
        })

        return response
