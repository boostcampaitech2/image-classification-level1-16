import numpy as np
from scipy.spatial import distance_matrix

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

class MyModel(nn.Module):
    
    def __init__(self, backbone, hdim=100):
        super(MyModel, self).__init__()
        self.hdim = hdim
        self.eff = EfficientNet.from_pretrained(backbone, num_classes=hdim)
        #self.projection = nn.Linear(hdim, hdim, bias=False)
        self.bounding = nn.Tanh()
        self.keys = None

    def forward(self, img1, img2):
        
        out1 = self.eff(img1)
        out2 = self.eff(img2)
        out = ((out1 - out2)**2).mean(dim=-1)
        #out = torch.clamp(out, max=1)
        out = self.bounding(out)

        return out
    
    def create_mtx(self, dataloader, device):
        self.eval()
        embeddings = []
        with torch.no_grad():
            for img in tqdm(dataloader):
                img = img.to(device)
                embedding = self.eff(img)
                #embedding = self.bounding(embedding)
                embeddings.append(embedding.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        self.mtx = distance_matrix(embeddings, embeddings)
        return self.mtx
    
    def create_keys(self, dataloader, device):
        self.eval()
        embeddings = []
        with torch.no_grad():
            for img in tqdm(dataloader):
                img = img.to(device)
                embedding = self.eff(img)
                #embedding = self.bounding(embedding)
                embeddings.append(embedding.cpu().numpy())
        embeddings = np.concatenate(embeddings, axis=0)
        self.keys = embeddings

    def query(self, queries, n):
        dists = distance_matrix(queries, self.keys)
        idxs = dists.argsort(axis=-1)
        return idxs[:,:n]
 
        
        