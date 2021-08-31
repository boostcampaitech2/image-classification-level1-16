import os
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

class Trainer:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, model, dataloaders, df_valid, criterion, optimizer, device, num_epochs=20, scheduler=None, save_name='epoch', sub_dir=None):        

        save_dir = os.path.join(self.save_dir, sub_dir) if sub_dir is not None else self.save_dir
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            ### train_phase ###

            model.train()
            running_loss = 0.0
            dist1, dist0 = [], []
            
            for input1, input2, labels in tqdm(dataloaders['train']):

                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device)
            
                outputs = model(input1, input2)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                with torch.no_grad():
                    d1 = outputs[labels == 1]
                    d0 = outputs[labels == 0]
                    dist1.extend(d1.tolist())
                    dist0.extend(d0.tolist())
                    running_loss += loss.item()
               
            epoch_loss = running_loss / len(dataloaders['train'])
            print('[Train]  Loss: {:.4f} D1: {:.4f} D0: {:.4f}'.format(epoch_loss, sum(dist1)/(len(dist1) + 1e-6),  sum(dist0)/(len(dist0) + 1e-6)))


            ### valid phase ###

            model.eval()
            mtx = model.create_mtx(dataloaders['valid'], device)
            groups_idx = mtx.argsort(axis=1)[:,:7]
            mtx.sort(axis=1)
            #groups_dst = mtx[:,:7]
            
            acc = 0
            for group in groups_idx:
                query = df_valid['id'].iloc[group[0]]
                for key in group[1:]:
                    if query == df_valid['id'].iloc[key]:
                        acc += 1
            
            acc /= 6*len(groups_idx)
            
            margin = 0
            for row in mtx:
                margin += row[7] - row[6]
            margin /= len(mtx)
            
            print('[Valid]  Acc: {:.4f} Margin: {:.4f}'.format(acc, margin))
            torch.save(model.state_dict(), os.path.join(save_dir, f'{save_name}{epoch:0>3}.pt'))

            print()

        print('Training complete!')