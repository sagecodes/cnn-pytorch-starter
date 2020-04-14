import torchvision.models as models
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.transforms as transforms


import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image


class vgg16_pretrained:
    
    def __init__(self, num_classes):
        # self.device = device
        self.num_classes = num_classes
        self.model = models.vgg16(pretrained=True)
        self.fc_out = nn.Linear(2048, num_classes, bias=True)

        # freeze model params for features
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = self.fc_out
        

    # def forward(self):
    # No forward needed imported model



