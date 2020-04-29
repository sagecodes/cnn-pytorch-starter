import io
import sys
import torchvision.transforms as transforms
from PIL import Image

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

from torchvision import models

sys.path.insert(0, './helpers')
sys.path.insert(0, './models')
# from .helpers.model_helpers import 

from resnet50 import Resnet50_pretrained
from model_helpers import load_model
from model_helpers import predict

# Model for prediction
res_model = Resnet50_pretrained(3)
# res_model = load_model(res_model, 'trained_models/test_train.pt',True)
model = res_model.model
 # Load weights, set ready for prediction
model = load_model(model, 'trained_models/test_train.pt',True)

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
@app.route('/predict', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # get file from the request
        file = request.files['file']
        prediction = predict(res_model.model, file, device, True)
        return jsonify({'class_id': prediction})
        
    return '''
        <!doctype html>
        <title>Upload new File</title>
        <h1>Upload new File</h1>
        <form method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
        </form>
    '''
if __name__ == '__main__':
    app.run()