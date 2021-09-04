import argparse
import yaml
from spam.trainer import Trainer
from spam.inference import Inferencer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_infer', type=str, help='path of inference configuration yaml file') 
    args = parser.parse_args()

    with open(args.config_infer) as f:
        config_infer = yaml.load(f, Loader=yaml.FullLoader)

    inferencer = Inferencer(**config_infer['inferencer'])
    inferencer.inference(config_infer['inference'])

    print('Evaluation Complete!')