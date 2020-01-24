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
    Custom class for loading labeled image data from a CSV file for pytorch
    """

    def __init__(self, data_root, df, path_col, label_col, transforms=None):

        self.img_dir = data_root
        # self.txt_path = txt_path
        self.img_names = df[path_col].values
        self.labels = df[label_col].values
        self.transform = transforms

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                        self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.labels[index]
        return img, label

    def __len__(self):
        return self.labels.shape[0]


def image_transforms(img_size):
    """
    Parmerters:
    img_size

    returns transforms and normalization
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
    Parmerters:
    data_dir, img_transforms

    returns data transformed from ImageFolder(data_dir)
    """

    img_data = datasets.ImageFolder(root=data_dir,
                                  transform=img_transforms)
    return img_data





def image_data_loader(data,batch_size,num_workers,shuffle=False):
    """
    Parmerters:
    data, batch_size, num_workers, shuffle=False

    returns data from torch DataLoader
    """
    img_loader = torch.utils.data.DataLoader(data,
                                           batch_size=batch_size, 
                                           num_workers=num_workers,
                                           shuffle=shuffle)

    return img_loader

def dir_loader_stack(data_dir,img_size,batch_size,num_workers,shuffle=False):
    """
    Parmerters:
    data_dir, img_size, batch_size, num_workers, shuffle=False

    Return:
    DataLoader object of dara from data_dir path 
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
    Parmerters:

    Return:
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