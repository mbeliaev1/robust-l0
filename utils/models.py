# torch
import torch
import torch.nn as nn
import torch.functional as F
# other
import numpy as np
# internal
from utils.helpers import *
from utils.trunc import *

# Model loaded using config
cfg = {
    'mlp': [1568, 3136, 3136],
    'cnn_small': [32, 32, 'M', 64, 64, 'M'],
    'cnn_large': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class Net(nn.Module):
    '''
    General network class that infers structure from cfg name as well as input shape

    Consists of feature network which is created by cfg class
    and classifier network which is inferred by input shape and does not contain a softmax

    this network class is general, and hence can have a robust comonent added

    Inputs
        cfg_name - config list for feature network structure
        k -        truncation parameter, if None no truncating 
        mu/sigma - scales that change the input domain
        embedding- dimensionality of fc layers in the classifier
    '''
    def __init__(self, cfg_name, k, embedding, input_shape):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.features = self._make_layers(cfg[cfg_name])
        if k> 0: self.trunc = self._make_trunc(k)
        self.classifier = self._make_classifier(embedding)
        
        if k != 0:
            raise NotImplemented 

    def forward(self, x):

        # first flatten for trunc
        x = self.features(x)
        # flatten (redundant for mlp networks)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

    def _make_trunc(self, k):
        # creates truncation layer or just fc layer
        layers = []
        features = self.input_shape[1:].numel()
        layers.append(nn.Linear(features,features))
        return nn.Sequential(*layers)

    def _make_layers(self, cfg):
        # this is the feature extractor
        layers = []
        
        # linear networks dont have maxpooling layers 
        if 'M' not in cfg:
            in_shape = self.input_shape[1:].numel()
            layers += [nn.Flatten()]
            for x in cfg:
                layers += [nn.Linear(in_shape, x),
                        nn.BatchNorm1d(x),
                        nn.ReLU(inplace=True)]
                in_shape = x
            return nn.Sequential(*layers)

        # convolution networks
        in_channels = self.input_shape[1]
        for x in cfg:
            # print('in here: ',in_channels,x)
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, embedding):
        # first perform a forward pass to get the shape
        with torch.no_grad():
            temp = self.features(torch.randn(self.input_shape))
            in_size = temp.shape[1:].numel()

        layers = []
        layers += [nn.Linear(in_size,embedding),
                   nn.BatchNorm1d(embedding),
                   nn.ReLU(True),
                   nn.Linear(embedding,10)]            
        return nn.Sequential(*layers)
