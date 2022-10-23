# torch
import torch
import torch.nn as nn
import torch.functional as F
# other
import numpy as np
# internal
from utils.helpers import *
from utils.trunc import linear_trunc_ind, linear_trunc_dep, trunc_clip, linear_trunc_dep_abs, trunc_clip_abs

# Model loaded using config
cfg = {
    # 'mlp': [1568, 3136, 500],
    'mlp': [784, 512],
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
        cfg_name    - config list for feature network structure
        k           - truncation parameter, if 0 no truncating 
        mu/sigma    - scales that change the input domain
        embedding   - dimensionality of fc layers in the classifier
        input_shape - one time pass needed to infer shapes (bs, ch, w ,h)
        trunc_type  - optional, str defining truncation type

    NOTE: 
        - truncation preserves input shape
        - assumed to have 10 clases
    '''
    def __init__(self, cfg_name, embedding, input_shape, k=0, trunc_type='clip'):
        super(Net, self).__init__()
        self.input_shape = input_shape
        self.trunc = self._make_trunc(k, trunc_type) # can be empty
        self.features = self._make_layers(cfg[cfg_name])
        self.classifier = self._make_classifier(embedding)
        self.b_norm = nn.BatchNorm2d(input_shape[1])

    def forward(self, x):
        x = self.b_norm(x)

        x = self.trunc(x)
        x = self.features(x)
        x = x.view(x.size(0), -1) 
        x = self.classifier(x)
        return x

    def _make_trunc(self, k, trunc_type):
        '''
        Truncation layer PRESERVES INPUT SHAPE
        Types of truncation implemented:

        linear     - original trunctation from theory, outputs learnable 
                     layer with weights and no bias (by default)

        clip       - essentially a clipping function, removing k top and 
                      bottom integers inside all channels
        '''
        layers = []
        in_shape = self.input_shape[1:].numel()

        if k == 0:
            pass

        elif trunc_type == 'linear_ind':
            # input is always (bs,ch,w,h)
            layers.append(nn.Flatten())
            layers.append(linear_trunc_ind(in_shape,in_shape,k=k))
            layers.append(nn.Unflatten(1,(self.input_shape[1:])))

        elif trunc_type == 'linear_dep':
            # input is always (bs,ch,w,h)
            layers.append(nn.Flatten())
            layers.append(linear_trunc_dep(in_shape,in_shape,k=k))
            layers.append(nn.Unflatten(1,(self.input_shape[1:])))

        elif trunc_type == 'clip':
            layers.append(nn.Flatten())
            layers.append(trunc_clip(k))
            layers.append(nn.Unflatten(1,(self.input_shape[1:])))

        elif trunc_type == 'linear_dep_abs':
            # input is always (bs,ch,w,h)
            layers.append(nn.Flatten())
            layers.append(linear_trunc_dep_abs(in_shape,in_shape,k=k))
            layers.append(nn.Unflatten(1,(self.input_shape[1:])))

        elif trunc_type == 'clip_abs':
            layers.append(nn.Flatten())
            layers.append(trunc_clip_abs(k))
            layers.append(nn.Unflatten(1,(self.input_shape[1:])))
            

        # if neither condition, k=0 and we just pass empty sequential list
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
                   nn.ReLU(inplace=True),
                   nn.Linear(embedding,10)]            
        return nn.Sequential(*layers)
