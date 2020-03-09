from torchvision import datasets
import torchvision.transforms as transforms
import torch

import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt              


class images_from_csv:
    """
    This class loads labeled image data from a CSV file for a pytorch data loader 

    Example use:
         data = image_data_loader(images_from_csv('data/animals/',
                                                    train_df,
                                                    'FilePath',
                                                    'Label',
                                                    image_transforms(img_size)))

    args:
        data_root(str): Root directory where file paths point to
        df (Pandas DataFrame): contaning file paths and labels in two columns
        path_col (str): column containig the file paths
        label_col (str): column containing the labels for each image
        transforms() (function): Contains pytorch image transforms

    returns:
        image and label to dataloader 


    """

    def __init__(self, data_root, df, path_col, label_col, transforms=None):

        self.img_dir = data_root
        # self.txt_path = txt_path
        self.img_names = df[path_col].values
        self.labels = df[label_col].values
        self.transform = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                        self.img_names[index])).convert('RGB')
        
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.labels.shape[0]


def image_transforms(img_size):
    """
    This function defines the transforms to do to images in the
        pytorch data loader

    Example use:
    # part of the data loader stack
    data = image_data_loader(
        images_from_dir('data/animals/',
                        image_transforms(244)),
                        32,
                        0)

    args:
    - img_size (int): what size the image should be scaled to

    returns transforms to apply to images

    TODO: 
    - Multiple transform options to be activated by an arg
    """

    img_transforms = transforms.Compose(
                                [transforms.Resize(size=(img_size,img_size)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225] )])

    return img_transforms


def images_from_dir(data_dir, img_transforms):
    """
    This function returns transformed images from a directory for
     a pytorch data loader

    Example use:
        # part of the data loader stack
        data = image_data_loader(
            images_from_dir('data/animals/',
                            image_transforms(244)),
                            32,
                            0)

    args:
    - data_dir (str): directory containing subfolder of images to load
    - img_transforms (a pytorch transforms.Compose object): defined transforms
      for images being loaded

    returns transformed image data from ImageFolder(data_dir)
    """

    img_data = datasets.ImageFolder(root=data_dir,
                                  transform=img_transforms)
    return img_data





def image_data_loader(data,batch_size,num_workers,shuffle=False):
    """
    This function loads image data in batch sizes to train models

    Example use:
    # part of the data loader stack
        data = image_data_loader(
            images_from_dir('data/animals/',
                            image_transforms(244)),
                            32,
                            0, 
                            True)

    args:
    - data(pytorch dataloader object): Loader for images from directory or csv 
    - batch_size(int): Batch size to load in for model training
    - num_workers(int): Number of workers for multi-process data loading
    - shuffle(bool)=False: Shuffle data or not 

    returns data from torch DataLoader for model traning
    """
    img_loader = torch.utils.data.DataLoader(data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=shuffle)

    return img_loader

def dir_loader_stack(data_dir,img_size,batch_size,num_workers,shuffle=False):
    """
    This function stacks all functions needed to load and preprocess image data
       from a directory to train models in one place.

    Example use:
    train_loader = dir_loader_stack('data/animals/', 244, 32, 0, True)

    args:
    - data_dir(str): path to directory with images in class sub folders
    - img_size (int): what size the image should be scaled to
    - batch_size(int): Batch size to load in for model training
    - num_workers(int): Number of workers for multi-process data loading
    - shuffle=False(bool): Shuffle data or not 

    Returns:
    - DataLoader object for loading images from a directory from the path given
    """
    data = image_data_loader(
        images_from_dir(data_dir,
                        image_transforms(img_size)),
                        batch_size,
                        num_workers,
                        shuffle)
    return data


def csv_loader_stack(data_root,df, path_col, label_col,
                        img_size,batch_size,num_workers, shuffle=False):
    """
    This function stacks all functions needed to load and preprocess image data
       from a CSV file to train models in one place.

    Example use:
    train_loader = csv_loader_stack('/datasets/animals/', train_df, 'FilePath',
     'Label', 244, 32, 0, True)


    args:
    - data_root(str): path to root directory containing image paths from CSV
    - df (Pandas DataFrame): contaning file paths and labels in two columns
    - path_col (str): column containig the file paths
    - label_col (str): column containing the labels for each image
    - img_size (int): what size the image should be scaled to
    - batch_size(int): Batch size to load in for model training
    - num_workers(int): Number of workers for multi-process data loading
    - shuffle=False(bool): Shuffle data or not

    Returns:
    - DataLoader object for loading images from the CSV file given


    """
    data = image_data_loader(
            images_from_csv(data_root,
            df,
            path_col,
            label_col,
            image_transforms(img_size)),
            batch_size,
            num_workers,
            shuffle=shuffle)

    return data


def image_plot(loader):
    """
    
    """
    testX_sanity, testY_sanity = next(iter(loader))

    L = 3
    W = 3

    fig, axes = plt.subplots(L,W,figsize=(12,12))
    axes = axes.ravel()
    norm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                std=[1/0.229, 1/0.224, 1/0.255])
    
    for i in np.arange(0, L*W):
        img_norm = norm(testX_sanity[i])
        axes[i].imshow(img_norm.permute(1, 2, 0))

        axes[i].set_title('{}'.format(testY_sanity[i]))
        axes[i].axis('off')
    plt.subplots_adjust(hspace = 0)
    plt.show()
    plt.close()


def val_train_split(df, split, verbose=False, seed=42, shuffle=True):
    """
    
    """
    
    # set random seed
    np.random.seed(seed)

    if shuffle:
        df = df.sample(frac=1)
        
    val, train = np.split(df, [int(split*len(df))])
    
    val = val.reset_index(drop=True)
    train = train.reset_index(drop=True)
    
    print(f'Train Shape: {train.shape}')
    print(f'Validation Shape: {val.shape}')
    
    if verbose:
          print('Training DataFrame \n')
          print(train)
          print('\n-----------------------\n')
          print('Validation DataFrame \n')
          print(val)
          
    return train, val