from torchvision import transforms
from RandAugment import RandAugment

def get_transform(N=2, M=9):
    transform_train = transforms.Compose([
                transforms.RandomCrop(350, padding=4),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_train.transforms.insert(0, RandAugment(N, M))

    transform_valid = transforms.Compose([
                transforms.CenterCrop(350),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform_train, transform_valid
    
    
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