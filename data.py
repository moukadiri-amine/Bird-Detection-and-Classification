from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms as transforms


data_transforms = {
    'train': transforms.Compose([
        transforms.transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(45),
        transforms.RandomVerticalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}