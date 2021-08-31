import os
from transform import get_transform, get_age_transform
from torch.utils.data import Dataset, ConcatDataset
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

# def get_dataset(df_train, df_valid, df_test, target='label'):
#     transform_train, transform_valid = get_transform()
#     train_dataset = MaskTrainDataset(df_train, transform_train, target)
#     valid_dataset = MaskTrainDataset(df_valid, transform_valid, target)
#     test_dataset = MaskTestDataset(test_path = '/opt/ml/input/data/eval/images', df=df_test, transform=transform_valid)
#     return train_dataset, valid_dataset, test_dataset

def get_age_dataset(df_ff, df_train, df_valid, target='label'):
    transform_ff, transform_train, transform_valid = get_age_transform()
    train_ff = MaskTrainDataset(df_ff, transform_ff, target)
    train_dataset = MaskTrainDataset(df_train, transform_train, target)
    final_dataset = ConcatDataset([train_ff, train_dataset])
    valid_dataset = MaskTrainDataset(df_valid, transform_valid, target)
    return final_dataset, valid_dataset

def get_test_dataset(df_test):
    transform_train, transform_valid = get_transform()
    test_dataset = MaskTestDataset(test_path = '/opt/ml/input/data/eval/images', df=df_test, transform=transform_valid)
    return test_dataset