import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class MyLoss(nn.Module):
    
    def __init__(self, smoothing=0.05):
        super(MyLoss, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        x = torch.clip(x, min=self.smoothing, max=1-self.smoothing)
        loss = torch.sum(target*(x-self.smoothing)) + torch.sum((1-target)*(1-self.smoothing-x))
        return loss / len(target)