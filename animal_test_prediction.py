#%%
# imports
# Set paths for custom modules
sys.path.insert(0, './helpers')
sys.path.insert(0, './models')
from resnet50 import Resnet50_pretrained
from model_helpers import load_model
from model_helpers import predict

import pandas as pd

import glob
import os
#%%
# Load model
# Define architecture
res_model = Resnet50_pretrained(3)
# Load weights, set ready for prediction
res_model = load_model(res_model, 'trained_models/test_train.pt',True)

# run prediction


# %%
# Data
test_df = pd.read_csv('../datasets/test_animals/test_labels.csv')
test_data_dir = '../datasets/test_animals/'
print(test_df.head())

# %%
test_df


# %%
#Prediction 
device = "cuda"

paths = test_df["FilePath"]

test_df["Label"] = pd.factorize(test_df["Label"])[0]
true_labels = test_df["Label"]

preds = []

for path in paths:
    image = os.path.join(test_data_dir, path)
    print(image)
    preds.append(predict(res_model.model,image,device))


# %%
print(preds)
print(true_labels)

# %%
