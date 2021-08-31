import os
from transform import get_transform, get_age_transform
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from itertools import permutations

class ClusterTrainDataset(Dataset):

    mask = ['normal', 'incorrect', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5']

    def __init__(self, df, iteration = 64*124, p=0.5, transform=None):
        self.df = df
        self.p = p
        self.iteration = iteration
        self.transform = transform
        self.classes = [0, 1]

    def __len__(self):
        return self.iteration

    def __getitem__(self, index):

        img1_idx = np.random.randint(len(self.df))
        f1, f2 = np.random.choice(self.mask, 2)
        img1_path = (self.df.iloc[img1_idx])[f1]

        if np.random.rand() < self.p:
            img2_path = (self.df.iloc[img1_idx])[f2]
            label = 1

        else:
            img2_idx = np.random.randint(len(self.df))
            while img1_idx == img2_idx:
                img2_idx = np.random.randint(len(self.df))
            img2_path = (self.df.iloc[img2_idx])[f2]
            label = 0

        image1 = Image.open(img1_path)
        image2 = Image.open(img2_path)

        image1 = self.transform(image1)
        image2 = self.transform(image2)

        return image1, image2, label


class ClusterTestDataset(Dataset):
    
    def __init__(self, df, transform):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = Image.open(self.df.iloc[idx]['path'])
        img = self.transform(img)
        return img
    
class ClusterEvalDataset(Dataset):
    
    def __init__(self, df, transform, img_path='/opt/ml/input/data/eval/images'):
        self.df = df
        self.transform = transform
        self.img_path = img_path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        path = os.path.join(self.img_path, self.df.iloc[idx]['ImageID'])
        img = Image.open(path)
        img = self.transform(img)
        return img  

    
def get_dataset(df_train, df_valid, iteration = 64*124, p=0.5):
    transform_train, transform_valid = get_transform()
    train_dataset = ClusterTrainDataset(df_train, iteration, p, transform_train)
    valid_dataset = ClusterTestDataset(df_valid, transform_valid)
    return train_dataset, valid_dataset