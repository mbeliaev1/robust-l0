# torch
import torch
import torch.nn as nn
import torch.functional as F
# other
import numpy as np
# internal
from utils.helpers import *
from utils.trunc import *
import json

def load_net(net_path, device, input_shape):
    config = json.load(open(net_path+'/setup.json','rb'))
        
    net = Net(cfg_name=config['cfg_name'],
            k = config['k'],
            input_shape=input_shape,
            trunc_type=config['trunc_type'])

    net.to(device)
    net.load_state_dict(torch.load(net_path+'/net.pth', map_location=device))
    net.eval()

    return net

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
        input_shape - one time pass needed to infer shapes (bs, ch, w ,h)
        trunc_type  - optional, str defining truncation type

    NOTE: 
        - truncation preserves input shape
        - assumed to have 10 clases
        - only works with channel size one
    '''
    def __init__(self, cfg_name, input_shape, k=0, trunc_type='clip'):
        super(Net, self).__init__()
        self.input_shape = input_shape
        if trunc_type == 'conv':
            self.in_ch = 1
        else:
            self.in_ch = input_shape[1]

        self.trunc = self._make_trunc(k, trunc_type) # can be empty
        self.features = self._make_layers(cfg[cfg_name])
        self.classifier = self._make_classifier()
        self.b_norm = nn.BatchNorm2d(self.in_ch)


    def forward(self, x):
        # print('1',x.shape)
        # print('2',x.shape)
        x = self.trunc(x)
        x = self.b_norm(x)
        # print('3',x.shape)
        x = self.features(x)
        # print('4',x.shape)
        x = x.view(x.size(0), -1)
        # print('5',x.shape) 
        x = self.classifier(x)
        # print('6',x.shape)
        return x

    def _make_trunc(self, k, trunc_type):
        '''
        Truncation layer flattens image to (bs,1,h,w*ch) bnef
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

        elif trunc_type == 'conv':
            layers.append(trunc_conv(image_shape=self.input_shape,
                                     out_ch = self.in_ch,
                                     kernel_size=3,
                                     k=k))
        
        elif trunc_type == 'simple':
            layers.append(nn.Flatten())
            layers.append(trunc_simple(in_shape,k=k))
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
        in_channels = self.in_ch
        for x in cfg:
            # print('in here: ',in_channels,x)
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def _make_classifier(self):
        # first perform a forward pass to get the shape
        with torch.no_grad():
            temp = self.features(self.trunc(torch.randn(self.input_shape)))
            in_size = temp.shape[1:].numel()

        layers = []
        layers += [nn.Linear(in_size,10)]         
        return nn.Sequential(*layers)
