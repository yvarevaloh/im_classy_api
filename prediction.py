from flask_restful import Resource, reqparse
from flask import jsonify
from PIL import Image
import io
from os.path import splitext
from copy import deepcopy
from base64 import b64decode
from constants import FRAMES_KEY, MAX_FRAMES, VALID_EXTENSIONS
from errors import MissingFramesKeyError, ExceededFramesNumberError, ImageExtensionError, ImageExtensionError

from liveness_detection.image import Image

mock_response = {
    "type": "digital",
    "score": 9.0,
    "predictions": {
        "real": 0.56,
        "printed": 0.0,
        "digital": 9.0
    },
    "version": "1.0",
    "responseCode": "200",
    "responseMessage": "Success"
}


class Prediction(Resource):
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
            error = Prediction.get_binary_request_errors(content)
        elif type_ == "binary":
            error = Prediction.get_binary_request_errors(content)
        else:
            raise ImageExtensionError("error type_ should be image or binary")

    def post(self):
        images = []
        request = reqparse.request
        if request.data:
            "only one file"
            images.append(Image.open(io.BytesIO(request.data)))
            return mock_response
        if request.files:
            self.get_binary_request_errors(request.files)
            request_images = request.files.getlist(FRAMES_KEY)
            images.extend([Image.open(image) for image in request_images])
        if request.form:
            self.get_binary_request_errors(request.form)
            request_images = request.form.getlist(FRAMES_KEY)
            images.extend([Image.open(io.BytesIO(b64decode(image))) for image in request_images])

        response = self.create_response(**{
            "responseMessage": "All images processed.",
            "imagesProcessed": len(images)
        })
        return response
