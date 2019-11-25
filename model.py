import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from PIL import Image
from tqdm import tqdm

import os
import sys
import time
import datetime

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

nclasses = 20

def freeze(model, block=2):
    modules = list(model.children())[:-1]
    for module in modules[:-block]:
        for p in module.parameters() : 
            p.requires_grad = False
    return nn.Sequential(*modules)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.final1, self.final2 = nn.Linear(2048, 512), nn.Linear(2048, 512)
        self.final = nn.Linear(1024, nclasses)
        
        ## ResNet152 features extractor        
        self.res152 = models.resnet152(pretrained=True)
        self.res152 = freeze(self.res152,block=3)

        ## ResNet101 features extractor        
        self.res101 = models.resnet101(pretrained=True)
        self.res101 = freeze(self.res101,block=3)

    def forward(self, x):
        x1, x2 = self.res152(x).view(-1,2048), self.res101(x).view(-1,2048)
        x1, x2 = self.final1(x1), self.final2(x2)
        x = torch.cat([x1, x2], dim=1)
        output = self.final(x)
        
        return output
