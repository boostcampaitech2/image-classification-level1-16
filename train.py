import argparse
import yaml
from spam.trainer import Trainer
from spam.inference import Inferencer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_train', type=str, help='path of train configuration yaml file')
    parser.add_argument('--config_infer', type=str, help='path of inference configuration yaml file') 
    args = parser.parse_args()

    with open(args.config_train) as f:
        config_train = yaml.load(f, Loader=yaml.FullLoader)

    with open(args.config_infer) as f:
        config_infer = yaml.load(f, Loader=yaml.FullLoader)

    trainer = Trainer(**config_train['trainer'])
    inferencer = Inferencer(**config_infer['inferencer'])
    
    for target in config_train['targets']:
        model, weight_paths = trainer.train_ensemble(target, config_train[f'ensemble_{target}'])
        df_pseudo = inferencer.get_pseudo_label(target, weight_paths, config_infer[f'pseudo_labeling_{target}'], model=model)
        model, weight_paths = trainer.train_ensemble(target, config_train[f'ensemble_with_pseudo_{target}'], df_pseudo=df_pseudo)
        
    print('Training Complete!')