import pandas as pd
import torch
from itertools import chain
from tqdm import tqdm

def create_label(model, test_dataloader, df_submit, device, target='ans', save=False):
   
    model.eval()
    ans = []
    for inputs in tqdm(test_dataloader):
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            ans.append(preds.tolist())

    ans = list(chain(*ans))
    df_submit[target] = ans
    
    if save:
        df_submit.to_csv('test.csv',index=False)
        
    print(f'inference {target} complete!')
    return df_submit

def sum_label(df):
    
    def _label(row):
        return row['age'] + 3*row['label']# + 6*row['mask']

    df['ans'] = df.apply(_label, axis=1)
    return df[['ImageID', 'ans']]
    