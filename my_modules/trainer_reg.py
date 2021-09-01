import os
import torch
from tqdm import tqdm
import numpy as np

class Trainer:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, model, dataloaders, criterion, optimizer, device, num_epochs=20, scheduler=None, save_name='epoch', sub_dir=None):        

        save_dir = os.path.join(self.save_dir, sub_dir) if sub_dir is not None else self.save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        best_score = float('inf')
        best_epoch = 0
        
        for epoch in range(1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_cnt = 0
                
                y_pred, y_true = [], []
                
                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.type(torch.FloatTensor).to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.unsqueeze(1))
                        
                        y_true.extend(labels.tolist())
                        y_pred.extend(outputs.squeeze().tolist())
                        
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                    running_cnt += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    
                epoch_loss = running_loss / running_cnt
                corr = np.corrcoef(y_true, y_pred)[0][1]
                
                print('{} Loss: {:.4f} corr:{:.4f}'.format(phase, epoch_loss, corr))

                if phase == 'valid':
                    if epoch_loss < best_score:
                        best_score = epoch_loss
                        best_epoch = epoch
                    if scheduler is not None:
                        scheduler.step()
                    torch.save(model.state_dict(), os.path.join(save_dir, f'{save_name}{epoch:0>3}.pt'))

            print()

        print('Training complete!')
        print('Best f1 score {:4f} at epoch {}'.format(best_score, epoch))