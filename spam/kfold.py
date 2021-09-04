import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

class KFold:

    gender_mapping = {'male':0, 'female':1}
    mask_mapping = {'incorrect_mask':1, 'mask1':0, 'mask2':0,
                 'mask3':0, 'mask4':0, 'mask5':0, 'normal':2}

    def __init__(self, n_splits, age_bins=[0, 29, 59, 100],
                 train_csv_path='/opt/ml/input/data/train/train.csv',
                 train_img_path='/opt/ml/input/data/train/images',
                 random_state=42):
        self.train_csv_path = train_csv_path
        self.train_img_path = train_img_path
        self.folds = []
        self._genderate_kfold(n_splits, age_bins, random_state)

    def __len__(self):
        return len(self.folds)

    def __getitem__(self, idx):
        return self.folds[idx]

    def _genderate_kfold(self, n_splits, age_bins, random_state):

        self.folds = []
        df = self._get_preprocessed_df(age_bins)
        stratify_key = df['age'] + 3 * df['gender']

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        for train_idx, test_idx in skf.split(np.zeros(len(df)), stratify_key):
            df_train = self._generate_path_and_mask_field(df.loc[train_idx])
            df_valid = self._generate_path_and_mask_field(df.loc[test_idx])
            self.folds.append((df_train, df_valid))

    def _get_preprocessed_df(self, age_bins):
        
        df = pd.read_csv(self.train_csv_path)
        
        # correct mislabled images
        df = self._correct_mislabled_images(df)
        
        # categorize age column
        df['age'] = pd.cut(df['age'], bins=age_bins, labels=False)
        
        # categorize gender column
        df['gender'] = df['gender'].map(self.gender_mapping)

        return df

    def _correct_mislabled_images(self, df):

        # male -> female
        correction_female = ['001498-1_male_Asian_23', '004432_male_Asian_43', '005223_male_Asian_22']
        df.loc[df['path'].isin(correction_female), 'gender'] = 'female'

        # female -> male
        correction_male = ['001720_female_Asian_18', '006359_female_Asian_18', '006360_female_Asian_18', '006361_female_Asian_18',
                           '006362_female_Asian_18', '006363_female_Asian_18', '006364_female_Asian_18']
        df.loc[df['path'].isin(correction_male), 'gender'] = 'male'
        
        return df

    def _generate_path_and_mask_field(self, df):

        # create mask column
        df['mask'] = [['incorrect_mask', 'mask1', 'mask2', 'mask3', 'mask4', 'mask5', 'normal'] for _ in range(len(df))]
        df = df.explode('mask')
        
        # find corresponding path
        df['path'] = df.apply(lambda row: self._search_image(row['path'], row['mask']), axis=1)
        
        # categorize mask column
        df['mask'] = df['mask'].map(self.mask_mapping)
        
        return df

    def _search_image(self, path, mask):
        
        for f in os.listdir(os.path.join(self.train_img_path, path)):
            if f.startswith(mask):
                return os.path.join(self.train_img_path, path, f)

        raise FileNotFoundError

