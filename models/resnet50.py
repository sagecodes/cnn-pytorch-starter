import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms


import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

class Resnet50_pretrained:
    
    def __init__(self, num_classes):
        # self.device = device
        self.num_classes = num_classes
        self.model = models.resnet50(pretrained=True)
        self.fc_out = nn.Linear(2048, num_classes, bias=True)
        
    def build(self, verbose=False):

        # freeze model params for features
        for param in self.model.parameters():
            param.requires_grad = False

        # set output layer to num classes
        self.model.fc = self.fc_out

        if verbose:
            print(self.model)

        # self.model.to(self.device)

        return self.model


    def save(self):
        '''Save model'''
        pass

    def load(self, model_path):
        '''load model weights'''

        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        return self.model

    def log(self):
        '''Training & Validation logs '''
        pass