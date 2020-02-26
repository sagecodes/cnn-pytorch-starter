import io
import sys
import torchvision.transforms as transforms
from PIL import Image
# import jsonify
# import requests

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

from torchvision import models

sys.path.insert(0, './helpers')
sys.path.insert(0, './models')
from resnet50 import Resnet50_pretrained
from model_helpers import load_model
from model_helpers import predict


# Make sure to pass `pretrained` as `True` to use the pretrained weights:
model = models.densenet121(pretrained=True)
# Since we are using our model only for inference, switch to `eval` mode:
model.eval()


# Model for prediction
res_model = Resnet50_pretrained(3)
res_model = load_model(res_model, 'trained_models/test_train.pt',True)

# Temp device
device = "cuda"

#######################
# Flask routes
#######################

# Home page
@app.route('/')
def hello():
    return 'Hello World!!!'

# Prediction
@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        # img_bytes = file.read()
        # class_id, class_name = get_prediction(image_bytes=img_bytes)
        prediction = predict(res_model.model, file, device)
        return jsonify({'class_id': prediction})


if __name__ == '__main__':
    app.run()