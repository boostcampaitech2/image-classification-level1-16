from torchvision import transforms
from .augmentation import RandAugment, Cutout

def get_transform(augment, crop_size, resize, cutout_size=None):
    
    if augment:
        transform = transforms.Compose([
                    RandAugment(),
                    transforms.CenterCrop(crop_size),
                    transforms.Resize(resize),
                    transforms.RandomHorizontalFlip(),
                    Cutout(size=cutout_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        transform = transforms.Compose([
                    transforms.CenterCrop(crop_size),
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return transform
