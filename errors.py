from constants import FRAMES_KEY, MAX_FRAMES, VALID_EXTENSIONS

from werkzeug.exceptions import HTTPException


class InternalServerError(HTTPException):
    pass


class MissingFramesKeyError(HTTPException):
    pass


class ExceededFramesNumberError(HTTPException):
    pass


class ImageExtensionError(HTTPException):
    pass


errors = {
    "InternalServerError": {
        "message": "Something went wrong",
        "status": 500
    },
    "MissingFramesKeyError": {
        "message": f"{FRAMES_KEY} must be in the request.",
        "status": 400
    },
    "ExceededFramesNumberError": {
        "message": f"Number of frames exceeded. Max is {MAX_FRAMES}.",
        "status": 400
    },
    "ImageExtensionError": {
        "message": f"Invalid file extension found!, only {VALID_EXTENSIONS} allowed.",
        "status": 400
    },

}
