from torchvision import transforms
from PIL import Image
import torch


class BaseAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, image):
        return self.transform(image)


class AddGaussianNoise(object):
    """
        transform 에 없는 기능들은 이런식으로 __init__, __call__, __repr__ 부분을
        직접 구현하여 사용할 수 있습니다.
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.CenterCrop((320, 256)),
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])

    def __call__(self, **args):
        return self.transform(image)

class MyAugmentation: #trasform1, transform2, valid
    def __init__(self, resize, mean, std, type=None, **args):
        self.type = type
        self.train_transform1 = transforms.Compose([
            transforms.CenterCrop((320, 256)),
            transforms.Resize(resize, Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            transforms.RandomApply(transforms=[transforms.RandomPerspective(distortion_scale=0.2, p=0.5)],p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            AddGaussianNoise()
        ])
        self.train_transform2 = transforms.Compose([
            transforms.CenterCrop((320, 256)),
            transforms.Resize(resize, Image.BILINEAR),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self.valid_transform = transforms.Compose([
            transforms.Resize(resize, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self,image):
        if self.type == 'train1':
            return self.train_transform1(image)
        elif self.type == 'train2':
            return self.train_transform2(image)
        elif self.type == 'valid': #valid
            return self.valid_transform(image)
        return self.train_transform1(image)