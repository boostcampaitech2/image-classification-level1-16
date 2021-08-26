import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from torchvision import models
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop


class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super(MyModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class ResnetModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=self.resnet.fc.in_features,out_features=num_classes)
        )
        self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        x = self.resnet(x)
        #x = self.sigm(x)
        return x
    
class ResnetModel2(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        for p in self.resnet.parameters():
            p.requires_grad = False
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=self.resnet.fc.in_features,out_features=64),
            nn.Dropout(0.3),
            nn.Linear(in_features=64,out_features=18)
        )
        #self.sigm = nn.Sigmoid()
        
    def forward(self, x):
        x = self.resnet(x)
        #x = self.sigm(x)
        return x