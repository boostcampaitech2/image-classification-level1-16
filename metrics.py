from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Metrics:
    
    all_metric_fn = {'f1':f1_score, 'acc':accuracy_score}
    
    def __init__(self, metrics, earlystop=None):

        self.metrics = metrics
        self.metric_fn = {m:self.all_metric_fn[m] for m in metrics}
        self.best_metric = {m:-float('inf') for m in metrics}
        self.earlystop = earlystop
        
        
    def __call__(phase, y_true, y_pred):

        []
        f1 = f1_score(y_true, y_pred, average='weighted')
        print('{} Loss: {:.4f} Acc: {:.4f} F1: {}'.format(phase, epoch_loss, epoch_acc, f1))
        