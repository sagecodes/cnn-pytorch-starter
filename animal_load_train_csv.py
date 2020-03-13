#%%
##########################
# Imports 
##########################

# import system libraries
import sys
import os
import glob

# import Data Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                        

# %matplotlib inline

# Set paths for custom modules
sys.path.insert(0, './helpers')
sys.path.insert(0, './models')

# Data Loader functions
from data_loader import dir_loader_stack
from data_loader import csv_loader_stack
from data_loader import image_plot
from data_loader import val_train_split

# Model classes
from resnet50 import Resnet50_pretrained

# Model helpers
from model_helpers import train
from model_helpers import predict
from model_helpers import plot_train_history

# torch
import torch.nn as nn
import torch.optim as optim
import torch


#%%

# option to show datasets & vis while running in py script
verbose = True

##########################
# Data Paths 
##########################

# Labels from CSV

df_lab = pd.read_csv('../datasets/animals/labels.csv')
test_df = pd.read_csv('../datasets/test_animals/test_labels.csv')

# One hot encoding
df_lab.Label = pd.Categorical(pd.factorize(df_lab.Label)[0])
test_df.Label = pd.Categorical(pd.factorize(test_df.Label)[0])

if verbose:
    print(f"df_lab shape:  {df_lab.shape}")
    print(f"test_df shape:  {test_df.shape}")

# Dataset folder
data_dir = '../datasets/animals/'
test_data_dir = '../datasets/test_animals/'

# Create Train & Validation split
train_df, val_df = val_train_split(df_lab, 0.2)

# %%
##########################
# Data parameters
##########################

img_size = 244
batch_size = 32
num_workers = 0


# %% 
##########################
# Data Loaders
##########################
train_loader = csv_loader_stack(data_dir,train_df, 'FilePath', 'Label',
                        img_size,batch_size,num_workers,True)

val_loader = csv_loader_stack(data_dir,val_df, 'FilePath', 'Label',
                        img_size,batch_size,num_workers,False)

# test_loader = csv_loader_stack(test_data_dir,test_df, 'FilePath', 'Label',
#                         img_size,batch_size,num_workers,False)


loaders = {
    'train':train_loader,
    'valid':val_loader,
#     'test':test_loader,
}

# %%
##########################
# Verify Sample Data from data loaders
##########################

if verbose: 
    # Train Data sample
    image_plot(train_loader)

# %%
if verbose: 
    # validation data sample
    image_plot(val_loader)

# %%
# Test data sample (placeholder)
# image_plot(test_loader)

#%%
##########################
# Create model
##########################

# Number Classes to predict
num_classes = 3

# Compute device (cuda = GPU)
device = 'cuda'

# create model from model class
res_model = Resnet50_pretrained(num_classes)

#%%
##########################
# Train Model
##########################

# parameters
n_epochs = 2
learn_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(res_model.model.fc.parameters(), lr=learn_rate)

device = 'cuda'
save_path = 'trained_models/test_train_tmp'

#%% 
# Train 
H = train(res_model.model, n_epochs, loaders, optimizer,
                    criterion, device, save_path)


#%%

if verbose:
    # Train Log
    plot_train_history(H,n_epochs)
