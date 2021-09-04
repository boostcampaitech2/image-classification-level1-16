import os
from torch.utils.data import Dataset
from PIL import Image

class MaskTrainDataset(Dataset):

    def __init__(self, df, transform, target):
        self.df = df
        self.transform = transform
        self.target = target

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df['path'].iloc[index]
        label = self.df[self.target].iloc[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image, label

class MaskEvalDataset(Dataset):

    def __init__(self, df, transform):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image_path = self.df['path'].iloc[index]
        image = Image.open(image_path)
        image = self.transform(image)
        return image    


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
    
# def get_dataset(df_train, df_valid, df_test, target='label'):
#     transform_train, transform_valid = get_transform()
#     train_dataset = MaskTrainDataset(df_train, transform_train, target)
#     valid_dataset = MaskTrainDataset(df_valid, transform_valid, target)
#     test_dataset = MaskTestDataset(test_path = '/opt/ml/input/data/eval/images', df=df_test, transform=transform_valid)
#     return train_dataset, valid_dataset, test_dataset

# def get_age_dataset(df_train1, df_train2, df_valid, df_test, target='age'):
#     transform_train1, transform_train2, transform_valid = get_age_transform()
#     train_dataset1 = MaskTrainDataset(df_train1, transform_train1, target)
#     train_dataset2 = MaskTrainDataset(df_train2, transform_train2, target)
#     train_dataset = ConcatDataset([train_dataset1, train_dataset2])
#     valid_dataset = MaskTrainDataset(df_valid, transform_valid, target)
#     test_dataset = MaskTestDataset(test_path = '/opt/ml/input/data/eval/images', df=df_test, transform=transform_valid)
#     return train_dataset, valid_dataset, test_dataset

# def get_test_dataset(df_test):
#     transform_train, transform_valid = get_transform()
#     test_dataset = MaskTestDataset(test_path = '/opt/ml/input/data/eval/images', df=df_test, transform=transform_valid)
#     return test_dataset