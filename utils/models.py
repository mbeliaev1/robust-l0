# torch
import torch
import torch.nn as nn
import torch.functional as F
# other
import numpy as np
# internal
from utils.helpers import *

# LINEAR MODELS #
#----------------------------------------------------------------#
class L_Net(nn.Module):
    '''
    5-layer Linear Network
    Takes input a batch of flattened images from MNIST
    Layers:
        lin 1,2  - increase the dimension of the input
        fc 1,2,3 - decrease the dimension of the input

    NOTE softmax is not used at the output as nn.CrossEntropyLoss()
         takes care of this.
    '''
    def __init__(self):
        super(L_Net, self).__init__()
        self.lin1 = nn.Linear(784, 784*2)
        self.lin2 = nn.Linear(784*2, 3136)
        self.fc1 = nn.Linear(3136, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class r_L_Net(nn.Module):
    '''
    5-layer k-Truncated Linear Network (robust)
    Takes input a batch of flattened images from MNIST

    Inputs:
        k - truncation parameter

    Layers:
        lin 1,2  - increase the dimension of the input
        fc 1,2,3 - decrease the dimension of the input

    NOTE lin1 is the layer that preforms truncation
    
    NOTE softmax is not used at the output as nn.CrossEntropyLoss()
         takes care of this.
    '''
    def __init__(self, k):
        super(r_L_Net, self).__init__()
        self.lin1 = fast_trunc(784,784*2,k)
        self.lin2 = nn.Linear(784*2, 3136)
        self.fc1 = nn.Linear(3136, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class L_Net_eval(nn.Module):
    '''
    Eval Version for L_Net() that transforms
    the original domain using mu and sigma.
    Inputs:
        mu, sigma - transform the original domain
    '''
    def __init__(self, mu, sigma):
        super(L_Net_eval, self).__init__()
        self.lin1 = nn.Linear(784,784*2)
        self.lin2 = nn.Linear(784*2, 3136)
        self.fc1 = nn.Linear(3136, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)
        self.sigma = torch.tensor(sigma)
        self.mu = torch.tensor(mu)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = (x*self.sigma)+self.mu
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __call__(self,x):
        return self.forward(x)

class r_L_Net_eval(nn.Module):
    '''
    Eval Version for r_L_Net() that transforms
    the original domain using mu and sigma.
    Inputs:
        mu, sigma - transform the original domain
        k         - truncation paramater
    '''
    def __init__(self, k, mu, sigma):
        super(r_L_Net_eval, self).__init__()
        self.lin1 = fast_trunc(784,784*2,k)
        self.lin2 = nn.Linear(784*2, 3136)
        self.fc1 = nn.Linear(3136, 500)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, 10)
        self.sigma = torch.tensor(sigma)
        self.mu = torch.tensor(mu)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = (x*self.sigma)+self.mu
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def __call__(self,x):
        return self.forward(x)

# CONVOLUTIONAL MODELS #
#----------------------------------------------------------------#
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