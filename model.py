import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)

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