import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import numpy as np
# Function and nn.Modules for creating tuncation layers

class fast_trunc(nn.Module):
    def __init__(self, in_features, out_features, k):
        '''
        Custom truncation layer. Returns output of Linear Layer (FC) 
        with truncation happening at each vector dot product
        Inputs:
            in_features     - Dimension of input vector
            out_features    - Dimension of output vector
            k               - truncation param.
        '''
        super(fast_trunc, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        # Initialize weight matrix and bias vector 
        k0 = torch.sqrt(torch.tensor(1/(in_features)))
        w = -1*((-2*k0)*torch.rand(out_features,in_features)+k0)
        b = -1*((-2*k0)*torch.rand(out_features)+k0)
        self.weight = torch.nn.Parameter(data=w, requires_grad=True)
        self.bias = torch.nn.Parameter(data=b, requires_grad=True)

    def forward(self,x):
        # compute regular linear layer output, but save copy of x
        x_vals = x.clone().detach() 
        # x_vals = [bs, in_features]
        x = torch.matmul(x,self.weight.T)
        # x = [bs, out_features]
        temp = x_vals.view(-1,1,784)*self.weight
        # temp = [bs, out_features, in_features] 
        # for x in bs, out_features element wise prod. of w and x
        val_1, _ = torch.topk(temp,self.k)
        val_2, _ = torch.topk(-1*temp,self.k)
        # val shapes are (bs, out_dim, self.k)
        x -= val_1.sum(axis=-1)
        x += val_2.sum(axis=-1)
        x += self.bias
                
        return x

def trunc(x,k):
    '''
    THIS FUNCTION IS USED FOR ROBUST_VGG (CNN)
    
    Takes input x, and removes the top and bottom k features by
    zeroing them. 
    Inputs:
        x - 4-dim: [bs,in_ch,out_dim,out_dim]
        k - truncation parameter
    Outputs:
        x - truncated version of input
    '''
    x_vals = x.clone().detach()
    out_dim = x.shape[3]
    # Now x is dimension [bs, in_ch, out_dim, out_dim]
    with torch.no_grad():
        for i in range(x.shape[0]):
            temp = x_vals[i,:]
            _, idx_1 = torch.topk(temp.view(-1,out_dim**2),k)
            _, idx_2 = torch.topk(-1*temp.view(-1,out_dim**2), k)
            for ch in range(x.shape[1]):
                x[i,ch,idx_1[ch]//out_dim,idx_1[ch]%out_dim] = 0
                x[i,ch,idx_2[ch]//out_dim,idx_2[ch]%out_dim] = 0
    return x