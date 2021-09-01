import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
#from torchsummary import summary as summary_
from efficientnet_pytorch import EfficientNet

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


#MyModel
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0', num_classes)
#         self.feature1 = self.efficient_net.out_features
#         self.lr1 = nn.Linear(self.feature1, num_classes)
        
    def forward(self, x):
        
        return self.efficient_net(x)
    

#Using Res Net
class MyModel1(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.resnet50 = models.resnext50_32x4d(pretrained=True)
        self.feature1 = self.resnet50.fc.out_features
        self.lr1 = nn.Linear(self.feature1, num_classes)
        
    def forward(self, x):
        x = self.resnet50(x)
        return self.lr1(x)

#Using Google Net
class MyModel2(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.google_net = models.googlenet(pretrained=True)
        self.feature1 = self.google_net.out_features
        self.lr1 = nn.Linear(self.feature1, num_classes)
        
    def forward(self, x):
        x = self.google_net(x)
        return self.lr1(x)
    

#Using Dense Net
class MyModel3(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.densenet = models.densenet161(pretrained=True)
        self.feature1 = self.densenet.fc.out_features
        self.lr1 = nn.Linear(self.feature1, num_classes)
        
    def forward(self, x):
        x = self.densenet(x)
        return self.lr1(x)

#Ensemble
class MyEnsemble(nn.Module):

    def __init__(self, model1, model2, model3, num_classes):
        super(MyEnsemble, self).__init__()
        self.modelA = model1
        self.modelB = model2
        self.modelC = model3
        self.fc1 = nn.Linear(num_classes, 18)

    def forward(self, x):
        
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out3 = self.modelC(x)

        out = out1 + out2 + out3

        return self.fc1(out)