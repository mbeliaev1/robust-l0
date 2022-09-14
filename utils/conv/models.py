# torch
import torch
import torch.nn as nn
import torch.functional as F
# other
import numpy as np
# internal
from utils.helpers import *

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name='VGG19'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self._make_classifier()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self):
        layers = []
        layers += [nn.Linear(512,512),
                   nn.BatchNorm1d(512),
                   nn.ReLU(True),
                   nn.Linear(512,10)]            
        return nn.Sequential(*layers)

class VGG_eval(nn.Module):
    '''
    This version is used for the attack
    '''
    def __init__(self, mu, sigma, vgg_name='VGG19'):
        super(VGG_eval, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self._make_classifier()
        self.sigma = torch.tensor(sigma)
        self.mu = torch.tensor(mu)

    def forward(self, x):
        x = (x*self.sigma)+self.mu
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self):
        layers = []
        layers += [nn.Linear(512,512),
                   nn.BatchNorm1d(512),
                   nn.ReLU(True),
                   nn.Linear(512,10)]            
        return nn.Sequential(*layers)

class rob_VGG(nn.Module):
    '''
    Robust version of VGG net above. 
    Inputs:
        k - truncation parameter
    '''
    def __init__(self, k, vgg_name='VGG19'):
        super(rob_VGG, self).__init__()
        self.k = k
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self._make_classifier()

    def forward(self, x):
        # apply truncation first
        x = trunc(x,self.k)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self):
        layers = []
        layers += [nn.Linear(512,512),
                   nn.BatchNorm1d(512),
                   nn.ReLU(True),
                   nn.Linear(512,10)]            
        return nn.Sequential(*layers)

class rob_VGG_eval(nn.Module):
    '''
    This version is used for the attack
    '''
    def __init__(self, k, mu, sigma, vgg_name='VGG19'):
        super(rob_VGG_eval, self).__init__()
        self.k = k
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = self._make_classifier()
        self.sigma = torch.tensor(sigma)
        self.mu = torch.tensor(mu)

    def forward(self, x):
        x = (x*self.sigma)+self.mu
        x = trunc(x,self.k)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self):
        layers = []
        layers += [nn.Linear(512,512),
                   nn.BatchNorm1d(512),
                   nn.ReLU(True),
                   nn.Linear(512,10)]            
        return nn.Sequential(*layers)