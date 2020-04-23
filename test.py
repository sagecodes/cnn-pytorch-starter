# Set paths for custom modules
import sys
sys.path.insert(0, './helpers')
sys.path.insert(0, './models')
# Model classes
from resnet50 import Resnet50_pretrained
from vgg import vgg16_pretrained
from scratch import Scratch_net

from model_helpers import load_model
from model_helpers import predict
from data_loader import images_from_dir

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt    
import seaborn as sns
import glob
import os
import click

@click.command()
@click.option('--device', default='cpu', help='compute on cpu or cuda')
@click.option('--weights', default=None, help='{Path to Trained model weights')
@click.option('--data_csv', default=None, help='path to CSV dataset')
@click.option('--data_dir', default=None, help='directory where images are \
                                                                contained')
@click.option('--num_classes', default=1, help='number of classes to predict')
@click.option('--model_type', default='resnet50', help='Model Architecture \
                                                            for prediction')
def test_model( device, weights, data_csv, data_dir, num_classes, model_type):
    '''
    See README.md for details on using test.py

    TODO: 
        - better error handling if no model
    '''
    # load model architecture based on mode_type from args
    # see README.md for more details on model
    # see /models folder for model classes
    if model_type == 'resnet50':    
        model = Resnet50_pretrained(num_classes)
        model = model.model
    elif model_type == 'vgg16':
        model = vgg16_pretrained(num_classes)
        model = model.model
    elif model_type == 'scratch':
        model = Scratch_net(num_classes)
    else:
        print('\n!---------------------------------!\n')
        print('please select an existing model type or create you own')
        print('\n!---------------------------------!\n')

    # Load weights, set ready for prediction
    model = load_model(model, weights,True)

    true_labels = []
    preds = []

    # if data labels are to be loaded from a CSV file
    if data_csv:
        # Load test data from csv & log true labels
        test_df = pd.read_csv(data_csv)
        test_data_dir = data_dir
        paths = test_df["FilePath"]
        test_df["Label"] = pd.factorize(test_df["Label"])[0]
        true_labels = test_df["Label"].values.tolist()

        # run prediction on each image path
        # add predicted value to preds
        for path in paths:
            image = os.path.join(test_data_dir, path)
            print(image)
            preds.append(predict(model,image,device))

    # else load data from directory if no CSV labels given
    else:
        # load images with data loader
        images = images_from_dir(data_dir)
        # print image classes
        print(images.class_to_idx)

        # run prediction on each image path
        # add predicted value to preds and true labels to true_labels
        for image, label in images:
            true_labels.append(label)
            preds.append(predict(model,image,device))

    # print predicted values
    print('\n---------------------------------------------')
    print('predicted values:\n')
    print(preds)
    # print actual true values
    print('\n---------------------------------------------')
    print('True Values\n')
    print(true_labels)

    # print confusion matrix
    cm = confusion_matrix(true_labels, preds)
    print('\n---------------------------------------------')
    print('Confusion Matrix:\n')
    print(cm)
    
    # print classification report
    print('\n---------------------------------------------')
    print('Classification Report:\n')
    print(classification_report(true_labels, preds))

if __name__ == '__main__':
    test_model()