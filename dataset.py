import os
from transform import get_transform
from torch.utils.data import Dataset
from PIL import Image

class MaskTrainDataset(Dataset):

    def __init__(self, df, transform=None, target='label'):
        self.df = df
        self.transform = transform
        self.target = target
        self.classes = sorted(df[target].unique())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df['path'].iloc[index]
        label = self.df[self.target].iloc[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

class MaskTestDataset(Dataset):
    def __init__(self, test_path, df, transform=None):
        self.test_path = test_path
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = os.path.join(self.test_path, self.df['ImageID'].iloc[index])
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image    

def get_dataset(df_train, df_valid, df_test, target='label'):
    transform_train, transform_valid = get_transform()
    train_dataset = MaskTrainDataset(df_train, transform_train, target)
    valid_dataset = MaskTrainDataset(df_valid, transform_valid, target)
    test_dataset = MaskTestDataset(test_path = '/opt/ml/input/data/eval/images', df=df_test, transform=transform_valid)
    return train_dataset, valid_dataset, test_dataset