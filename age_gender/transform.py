from torchvision import transforms
from augmentation import RandAugment, Cutout

def get_transform():
    transform_train = transforms.Compose([
                RandAugment(),
                transforms.CenterCrop(350),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                Cutout(size=100),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform_valid = transforms.Compose([
                transforms.CenterCrop(350),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform_train, transform_valid

def get_age_transform():
    transform_ff = transforms.Compose([
                RandAugment(),
                transforms.CenterCrop(400),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                Cutout(size=100),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform_train = transforms.Compose([
                RandAugment(),
                transforms.CenterCrop(350),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                Cutout(size=100),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    transform_valid = transforms.Compose([
                transforms.CenterCrop(350),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    return transform_ff, transform_train, transform_valid

# def get_age_transform():
#     transform_train1 = transforms.Compose([
#                 RandAugment(),
#                 transforms.Resize(224),
#                 transforms.CenterCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 Cutout(size=100),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#     transform_train2 = transforms.Compose([
#                 RandAugment(),
#                 transforms.CenterCrop(350),
#                 transforms.Resize(224),
#                 transforms.RandomHorizontalFlip(),
#                 Cutout(size=100),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#     transform_valid = transforms.Compose([
#                 transforms.CenterCrop(350),
#                 transforms.Resize(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
#     return transform_train1, transform_train2, transform_valid
    
    
# def get_transform():
    
#     transform_train = transforms.Compose([
#                 transforms.CenterCrop(350),
#                 transforms.Resize(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#     transform_valid = transforms.Compose([
#                 transforms.CenterCrop(350),
#                 transforms.Resize(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#     return transform_train, transform_valid