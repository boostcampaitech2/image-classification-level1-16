import os
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

class Trainer:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def train(self, model, dataloaders, criterion, optimizer, device, num_epochs=20, scheduler=None, save_name='epoch', sub_dir=None):        

        save_dir = os.path.join(self.save_dir, sub_dir) if sub_dir is not None else self.save_dir

        for epoch in range(1, num_epochs+1):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                running_cnt = 0
                y_true, y_pred = [], []

                for inputs, labels in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        y_true.extend(labels.tolist())
                        y_pred.extend(preds.tolist())

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                            if scheduler is not None:
                                scheduler.step()
                    
                    running_cnt += inputs.size(0)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                epoch_loss = running_loss / running_cnt
                epoch_acc = running_corrects.double() / running_cnt
                f1 = f1_score(y_true, y_pred, average='macro')

                print('{} Loss: {:.4f} Acc: {:.4f} F1: {}'.format(phase, epoch_loss, epoch_acc, f1))

                if phase == 'valid':
                    torch.save(model.state_dict(), os.path.join(save_dir, f'{save_name}{epoch:0>3}.pt'))

            print()

        print('Training complete!')
        print('Best val Acc: {:4f}'.format(best_acc))