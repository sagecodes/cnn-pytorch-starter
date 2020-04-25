# import system libraries
import sys
import os
import glob

# import Data Libraries
import click
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                        

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
from vgg import vgg16_pretrained
from scratch import Scratch_net

# Model helpers
from model_helpers import train
from model_helpers import predict
from model_helpers import plot_train_history

# torch
import torch.nn as nn
import torch.optim as optim
import torch

# Command line arguments
@click.command()
@click.option('--verbose', default=False, help='Verbose output')
@click.option('--device', default='cpu', help='compute on cpu or cuda')
@click.option('--num_classes', default=2, help='number of classes to predict')
@click.option('--n_epochs', default=3, help='number of epochs')
@click.option('--learn_rate', default=0.001, help='learning rate')
@click.option('--save_path', default=None, help='save path for model \
                                                        and history')
@click.option('--csv_labels', default=None, help='path to CSV dataset')
@click.option('--img_size', default=244, help='resize images')
@click.option('--batch_size', default=8, help='Batch size for training')
@click.option('--num_workers', default=0, help='num workers for pytorch')
@click.option('--data_dir', default=None, help='directory where images are \
                                                                    contained')
@click.option('--val_data', default=None, help='path to validation data')
@click.option('--model_type', default='resnet50', help='Model Architecture \
                                                                 for training')
def load_train(verbose, device, num_classes, n_epochs, learn_rate, save_path,
                csv_labels, img_size,batch_size, num_workers, data_dir,
                model_type, val_data):
    '''
    See README.md for train.py instructions

    TODO: 
        - run folder check and create
        - better error handling if no model
    '''
    # if data labels are to be loaded from a CSV file
    if csv_labels:
        # CSV to DataFrame
        train_df = pd.read_csv(csv_labels)

        # One hot encoding for csv labels
        train_df.Label = pd.Categorical(pd.factorize(train_df.Label)[0])

        # if validation dataset included use it
        # else perfrom train / val split on training data
        if val_data:
            # CSV to DataFrame
            val_df = pd.read_csv(val_data)
            # One hot encoding for csv labels
            val_df.Label = pd.Categorical(pd.factorize(val_df.Label)[0])
        else:
            # Create Train & Validation split
            train_df, val_df = val_train_split(train_df, 0.2)

        # Create train & validation data loaders from CSV data
        train_loader = csv_loader_stack(data_dir,train_df, 'FilePath', 'Label',
                                img_size,batch_size,num_workers,True)

        val_loader = csv_loader_stack(data_dir,val_df, 'FilePath', 'Label',
                                img_size,batch_size,num_workers,False)
    
    # else load data from directory if no CSV labels given
    else:
        # if validation dataset included use it
        # else perfrom train / val split on training data
        if val_data:
            # Create train & validation data loaders from directory
            train_loader =  dir_loader_stack(data_dir, img_size, batch_size, 
                                                    num_workers, True)
            
            val_loader = dir_loader_stack(val_data, img_size, batch_size, 
                                                        num_workers,False)
        else:
            # Create Train & Validation split from data loader
            loader = dir_loader_stack(data_dir, img_size, batch_size,
                                        num_workers, True)
            train_size = int(0.8 * len(loader.dataset))
            test_size = len(loader.dataset) - train_size
            train_data, val_data = torch.utils.data.random_split(loader.dataset,
                                                            [train_size, 
                                                            test_size])
            # Create train & validation data loaders from directory
            train_loader = torch.utils.data.DataLoader(train_data,
                                                        batch_size=batch_size)
            val_loader = torch.utils.data.DataLoader(val_data, 
                                                        batch_size=batch_size)
    # Create loaders object to pass for training
    loaders = {
        'train':train_loader,
        'valid':val_loader
    }
    # Print train & validation lengths
    print('Training length: ' + str(len(train_loader.dataset)))
    print('Validation length: ' + str(len(val_loader.dataset)))
  
    
    if verbose:
        # Output training and validation images before training
        print("\nTraining images")
        image_plot(train_loader)
        print("\nValidation images")
        image_plot(val_loader)

    # Load model, Set criterion & set optimizer based on mode_type from args
    # see README.md for more details on model
    # see /models folder for model classes
    if model_type == 'resnet50':
        model = Resnet50_pretrained(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.model.fc.parameters(), lr=learn_rate)
        train_model = model.model
    elif model_type == 'vgg16':
        model = vgg16_pretrained(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.model.classifier._modules['6'].parameters(),
                                                                lr=learn_rate)
        train_model = model.model
    elif model_type == 'scratch':
        model = Scratch_net(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.classifier._modules['6'].parameters(),
                                                                lr=learn_rate)
        train_model = model
    else:
        print('\n!---------------------------------!\n')
        print('please select an existing model type or create you own')
        print('\n!---------------------------------!\n')
    
    if verbose:
        # print model architecture to terminal
        print(train_model)

    # Train model based on input args
    H = train(train_model, n_epochs, loaders, optimizer,
                        criterion, device, save_path)
 
    if verbose:
        # Output Training Log (Train & Val: acc & loss per epoch)
        plot_train_history(H,n_epochs)

if __name__ == '__main__':
    load_train()