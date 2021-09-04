import os
from itertools import chain

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

from spam.transform import get_transform
from spam.dataset import MaskEvalDataset


class Inferencer:
    
    def __init__(self, eval_csv_path, eval_img_path):
        self.eval_csv_path = eval_csv_path
        self.eval_img_path = eval_img_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def inference(self, config):
        
        df_eval = pd.read_csv(self.eval_csv_path)
        dataloader = self._get_dataloader(config)        

        weights = config['weights']
        with torch.no_grad():
            for target in weights:
                
                model = EfficientNet.from_pretrained(config['model'], num_classes=2 if target=='gender' else 3)
                model.to(self.device)
                model.eval()
                
                probs = []
                for weight_path in weights[target]:

                    prob = []
                    model.load_state_dict(torch.load(weight_path))

                    for inputs in tqdm(dataloader):
                        inputs = inputs.to(self.device)
                        outputs = model(inputs)
                        prob_batch =F.softmax(outputs, dim=-1)
                        prob.append(prob_batch.cpu().numpy())

                    probs.append(np.concatenate(prob, axis=0))
                
                ensembled_prob = np.mean(probs, axis=0)
                soft_vote = ensembled_prob.argmax(axis=-1)
                df_eval[target] = soft_vote

        df_submit = self._sum_targets(df_eval)
        df_submit.to_csv('output.csv', index=False)
    
    def get_pseudo_label(self, target, weight_paths, config, model=None):
        
        df_eval = pd.read_csv(self.eval_csv_path)
        dataloader = self._get_dataloader(config)        
        
        if model is None:
            model = EfficientNet.from_pretrained(config['model'], num_classes=2 if target=='gender' else 3)
            model.to(self.device)
        model.eval()

        probs = []

        with torch.no_grad():
            for weight_path in weight_paths:

                prob = []
                model.load_state_dict(torch.load(weight_path))

                for inputs in tqdm(dataloader):
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    prob_batch =F.softmax(outputs, dim=-1)
                    prob.append(prob_batch.cpu().numpy())

                probs.append(np.concatenate(prob, axis=0))
            
        ensembled_prob = np.mean(probs, axis=0)
        print(ensembled_prob)
        soft_vote = ensembled_prob.argmax(axis=-1)
        idx_filtered = np.nonzero(np.any(ensembled_prob >= np.array(config['threshold']), axis=1))[0]

        df_eval[target] = soft_vote
        df_filtered = df_eval.iloc[idx_filtered].copy()
        print('pseudo labeled datas:', len(df_filtered))

        return df_filtered

    def _get_dataloader(self, config):
        
        df_eval = pd.read_csv(self.eval_csv_path)
        df_eval['path'] = df_eval['ImageID'].apply(lambda x: os.path.join(self.eval_img_path, x))
        
        # transform
        input_size = EfficientNet.get_image_size(config['model'])
        transform_eval = get_transform(augment=False, resize=input_size, **config['transform'])

        # dataset
        eval_dataset = MaskEvalDataset(df=df_eval, transform=transform_eval)

        # dataloader
        dataloader = DataLoader(eval_dataset, drop_last=False, shuffle=False, **config['dataloader'])
        
        return dataloader
    
    def _sum_targets(self, df):
        df['ans'] = df['age'] + 3*df['gender'] + 6*df['mask']
        return df[['ImageID', 'ans']]
    
# def create_label(model, test_dataloader, df_submit, device, target='ans', save=False):
   
#     model.eval()
#     ans = []
#     for inputs in tqdm(test_dataloader):
#         inputs = inputs.to(device)

#         with torch.no_grad():
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             ans.append(preds.tolist())

#     ans = list(chain(*ans))
#     df_submit[target] = ans
    
#     if save:
#         df_submit.to_csv('test.csv',index=False)
        
#     print(f'inference {target} complete!')
#     return df_submit


    