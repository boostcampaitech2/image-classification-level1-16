# image-classification-level1-16
image-classification-level1-16 created by GitHub Classroom

## Structure

```
image-classification-level1-16
│
├── README.md
├── requirements.txt
├── spam
│   ├── augmentation.py
│   ├── dataset.py
│   ├── inference.py
│   ├── kfold.py
│   ├── loss.py
│   ├── model.py
│   ├── trainer.py
│   ├── transform.py
│   └── utils.py
├── configs
│   ├── inference.yaml
│   └── train.yaml
├── evaluate.py
└── train.py
```

## Train

`python train.py --config_train config/train.yaml --config_infer config/inference.yaml`

## Evaluate

`python evaluate.py --config_infer config/inference.yaml`