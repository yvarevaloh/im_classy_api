from flask import Flask
from flask_restful import Api
from flask_cors import CORS
from prediction import Classify
from errors import errors


app = Flask(__name__)
api = Api(app, errors=errors)
cors = CORS(app, resources={r"/classify/*": {"origins": "*"}})


api.add_resource(Classify, '/classify/')


@app.route('/')
def hello_world():
    return 'Hello from Liveness detection!'
