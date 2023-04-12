import torch
import torch.nn as nn
import math
# import torch.nn.functional as F
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import numpy as np
from utils.helpers import *
# Function and nn.Modules for creating tuncation layers

class linear_trunc_ind(nn.Module):
    def __init__(self, in_features, out_features, k=0, bias=False):
        '''
        Custom truncation layer. Returns output of Linear Layer (FC) 
        with truncation happening at each vector dot product
        Inputs:
            in_features     - Dimension of input vector
            out_features    - Dimension of output vector
            image_size      - 
            k               - truncation param
        '''
        super(linear_trunc_ind, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        # compute regular linear layer output, but save copy of x
        x_vals = x.clone().detach() #(bs, in_features)
        x = torch.matmul(x,self.weight.T) #(bs, out_features)
        temp = x_vals.view(-1,1,self.in_features)*self.weight # (bs, out_features, in_features)

        # for x in bs, out_features element wise prod. of w and x
        val_top, _ = torch.topk(temp,self.k) #(bs, out_dim, self.k)
        val_bot, _ = torch.topk(-1*temp,self.k)
        
        x -= val_top.sum(axis=-1)
        x -= val_bot.sum(axis=-1)
        if self.bias is not None: x += self.bias
                
        return x

class linear_trunc_dep(nn.Module):
    def __init__(self, in_features, out_features, k=0, bias=False):
        '''
        Custom truncation layer. Returns output of Linear Layer (FC) 
        with truncation happening at each vector dot product
        Inputs:
            in_features     - Dimension of input vector
            out_features    - Dimension of output vector
            image_size      - 
            k               - truncation param
        '''
        super(linear_trunc_dep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        # compute regular linear layer output, but save copy of x
        x_vals = x.clone().detach() #(bs, in_features)
        # sum over first axis gives you largest contribution per in_feature
        temp = (x_vals.view(-1,1,self.in_features)*self.weight).sum(axis=1) # (bs, in_features)

        # idx gives you largest contributing pixels
        _, idx_top = torch.topk(temp,self.k) #(bs, in_dim)
        _, idx_bot = torch.topk(-1*temp,self.k)

        z = torch.ones_like(x).to(x.device)
        z[np.arange(x.shape[0]),idx_top.T] = 0
        z[np.arange(x.shape[0]),idx_bot.T] = 0
        x = torch.matmul(x*z,self.weight.T) #(bs, out_features)

        # x[np.arange(x.shape[0]),idx_top.T] = 0
        # x[np.arange(x.shape[0]),idx_bot.T] = 0
        # x = torch.matmul(x,self.weight.T) #(bs, out_features)

        if self.bias is not None: x += self.bias
                
        return x

class trunc_clip(nn.Module):
    def __init__(self, k=0):
        '''
        Takes input x, and removes the top and bottom k features by
        zeroing them. Truncation with identity weights
        '''
        super(trunc_clip, self).__init__()
        self.k = k

    def forward(self,x):
        x_vals = x.clone().detach() 

        _, idx_top = torch.topk(x_vals,self.k) #(bs, out_dim, self.k)
        _, idx_bot = torch.topk(-1*x_vals,self.k)

        # better to create mask instead of inplace 
        z = torch.ones_like(x).to(x.device)
        z[np.arange(x.shape[0]),idx_top.T] = 0
        z[np.arange(x.shape[0]),idx_bot.T] = 0
        x = x*z

        # or not
        # x[np.arange(x.shape[0]),idx_top.T] = 0
        # x[np.arange(x.shape[0]),idx_bot.T] = 0

        return x

class trunc_simple(nn.Module):
    def __init__(self, in_features, k=0, bias=False):
        '''
        Takes input x (assumed flatten), and removes the top and bottom k features by
        by first multiplying by window and then zeroing them
        '''
        super(trunc_simple, self).__init__()
        self.k = k
        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(in_features))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.ones_(self.weight)

        # nn.init.uniform_(self.weight,0,1)
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        #     if fan_in != 0:
        #         bound = 1 / math.sqrt(fan_in)
        #         nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):

        x = x*self.weight
        x_vals = x.clone().detach() 

        _, idx_top = torch.topk(x_vals,self.k) #(bs, out_dim, self.k)
        _, idx_bot = torch.topk(-1*x_vals,self.k)

        # better to create mask instead of inplace 
        z = torch.ones_like(x).to(x.device)
        z[np.arange(x.shape[0]),idx_top.T] = 0
        z[np.arange(x.shape[0]),idx_bot.T] = 0
        x = x*z

        return x
    
class trunc_conv(nn.Module):
    def __init__(self, image_shape, out_ch, kernel_size, k, bias=True):
        '''
        Custom convolution layer which manually performs nn.Conv2d. 
        See torch documentation for reference

        NOTE:
        Assumes stride is 1 and padding is zero
        Assumes kernel is and image are square
        '''
        super(trunc_conv, self).__init__()
        self.in_ch = image_shape[1]
        self.l = image_shape[-1]
        self.out_ch = out_ch
        self.ker_dim = kernel_size
        self.k = k
        self.out_dim = self.l - kernel_size + 1
        
        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(self.out_ch, self.in_ch, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_ch))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        '''
        Performs 2d convolution where x is input
        '''
        # build output tensor
        x_vals = x.clone().detach()
        z = self._find_mask(x_vals,x.device)
        out = nn.functional.conv2d(x*z,self.weight,self.bias,padding=1)

        return out
    
    def _find_mask(self,x,device):
        '''
        Finds the mask used to weight x in convolution
        '''

        # flat kern is the conv in matrix form (out_dim * in_dim)
        flat_kern = toeplitz_mult_ch(self.weight.detach().cpu(), [self.in_ch,self.l,self.l])
        flat_kern = torch.tensor(flat_kern).to(device)
        kern_mask = flat_kern != 0
        assert kern_mask.sum().sum() == (self.out_ch*self.out_dim**2)*(self.in_ch*self.ker_dim**2)
        r_scale = 1/kern_mask.sum(axis=0)

        # compute window averages with forward conv pass
        kern_avg = nn.functional.conv2d(x,self.weight.detach())
        kern_avg = kern_avg.reshape(x.shape[0],self.out_ch*self.out_dim**2)/(self.in_ch*self.ker_dim**2)

        # now we compute difference between pixel*featre and window average 
        r_vals = x.reshape(-1,1,self.in_ch*self.l**2)*flat_kern #feature pixel product
        r_vals -= kern_avg.reshape(-1,self.out_ch*self.out_dim**2,1) # subtract all averages
        r_vals *= kern_mask # remove values where no pixel feature product
        r_vals = r_vals.sum(axis=1)/r_scale # sum average contribution over all windows

        # now find top and bottom k to create mask
        _, idx_top = torch.topk(r_vals,self.k) #(bs, self.k)
        _, idx_bot = torch.topk(-1*r_vals,self.k)

        
        z = torch.ones_like(r_vals).float()
        z[np.arange(r_vals.shape[0]),idx_top.T] = 0
        z[np.arange(r_vals.shape[0]),idx_bot.T] = 0

        return z.view(x.shape)
    
# COPIES OF FUNCTIONS WITH ABSOLUTE VALUES INSTEAD #
class linear_trunc_dep_abs(nn.Module):
    def __init__(self, in_features, out_features, k=0, bias=False):
        '''
        Custom truncation layer. Returns output of Linear Layer (FC) 
        with truncation happening at each vector dot product
        Inputs:
            in_features     - Dimension of input vector
            out_features    - Dimension of output vector
            image_size      - 
            k               - truncation param
        '''
        super(linear_trunc_dep_abs, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(out_features,in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self,x):
        # compute regular linear layer output, but save copy of x
        x_vals = x.clone().detach() #(bs, in_features)
        # sum over second axis gives you largest contribution per in_feature
        temp = (x_vals.view(-1,1,self.in_features)*self.weight).sum(axis=1) # (bs, in_features)

        # idx gives you largest contributing pixels in absolute value
        _, idx_top = torch.topk(torch.abs(temp),self.k) #(bs, in_dim)

        z = torch.ones_like(x).to(x.device)
        z[np.arange(x.shape[0]),idx_top.T] = 0
        x = torch.matmul(x*z,self.weight.T) #(bs, out_features)

        # x[np.arange(x.shape[0]),idx_top.T] = 0
        # x = torch.matmul(x,self.weight.T) #(bs, out_features)

        if self.bias is not None: x += self.bias
                
        return x

class trunc_clip_abs(nn.Module):
    def __init__(self, k=0, bias=False):
        '''
        Takes input x, and removes the top and bottom k features by
        zeroing them. Truncation with identity weights
        '''
        super(trunc_clip_abs, self).__init__()
        self.k = k

    def forward(self,x):
        x_vals = x.clone().detach() 

        _, idx_top = torch.topk(torch.abs(x_vals),self.k) #(bs, out_dim, self.k)

        # better to create mask instead of inplace 
        z = torch.ones_like(x).to(x.device)
        z[np.arange(x.shape[0]),idx_top.T] = 0
        x = x*z

        # or not
        # x[np.arange(x.shape[0]),idx_top.T] = 0
        return x

class trunc_conv_abs(nn.Module):
    def __init__(self, image_size, kernel_size, k, bias=True):
        '''
        Custom convolution layer which manually performs nn.Conv2d. 
        See torch documentation for reference

        NOTE:
        Assumes stride is 1 and padding is zero
        Assumes kernel is and image are square
        Assumed in and out channels are 1
        '''
        super(trunc_conv_abs, self).__init__()
        self.l = image_size
        self.ker_dim = kernel_size
        self.k = k
        self.out_dim = image_size - kernel_size + 1
        
        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(1, 1, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(1))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        '''
        Performs 2d convolution where x is input
        '''
        # build output tensor
        x_vals = x.clone().detach()
        z = self._find_mask(x_vals,x.device)
        out = nn.functional.conv2d(x*z,self.weight,self.bias)

        return out
    
    def _find_mask(self,x,device):
        '''
        Finds the mask used to weight x in convolution
        '''

        # flat kern is the conv in matrix form (out_dim**2, in_dim**2)
        flat_kern = toeplitz_1_ch(self.weight[0,0].detach().cpu(), [self.l,self.l])
        flat_kern = torch.tensor(flat_kern).to(device)
        kern_mask = flat_kern != 0
        assert kern_mask.sum().sum() == (self.out_dim**2)*(self.ker_dim**2)
        r_scale = 1/kern_mask.sum(axis=0)

        # compute window averages with forward conv pass
        kern_avg = nn.functional.conv2d(x,self.weight.detach())
        kern_avg = kern_avg.reshape(x.shape[0],self.out_dim**2)/(self.ker_dim**2)

        # now we compute difference between pixel*featre and window average 
        r_vals = x.reshape(-1,1,self.l**2)*flat_kern #feature pixel product
        r_vals -= kern_avg.reshape(-1,self.out_dim**2,1) # subtract all averages
        r_vals *= kern_mask # remove values where no pixel feature product
        r_vals = r_vals.sum(axis=1)/r_scale # sum average contribution over all windows

        
        # for im in range(r_vals.shape[0]):
        #     # iterate over each kernel, adding contribution to r_vals
        #     for i_kern in range(flat_kern.shape[0]):
        #         # print("out image index: ",i_kern//out_dim, i_kern%out_dim)
        #         kern_avg = np.dot(flat_kern[i_kern],x[im,0].flatten())/(self.ker_dim**2)
        #         # print(kern_mask[i_kern])
        #         temp = (flat_kern[i_kern]*x[im,0].flatten()-kern_avg)*kern_mask[i_kern]
        #         r_vals[im] += temp 
        #     r_vals[im] /= r_scale
        

        # now find top and bottom k to create mask
        _, idx_top = torch.topk(torch.abs(r_vals),self.k) #(bs, out_dim, self.k)

        z = torch.ones_like(r_vals).float()
        z[np.arange(r_vals.shape[0]),idx_top.T] = 0

        return z.view(x.shape)

class trunc_simple_abs(nn.Module):
    def __init__(self, in_features, k=0):
        '''
        Takes input x (assumed flatten), and removes the top and bottom k features by
        by first multiplying by window and then zeroing them
        '''
        super(trunc_simple_abs, self).__init__()
        self.k = k
        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(in_features))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.ones_(self.weight, a=math.sqrt(5))

    def forward(self,x):

        x = x*self.weight
        x_vals = x.clone().detach() 
        
        _, idx_top = torch.topk(torch.abs(x_vals),self.k) #(bs, out_dim, self.k)

        # better to create mask instead of inplace 
        z = torch.ones_like(x).to(x.device)
        z[np.arange(x.shape[0]),idx_top.T] = 0
        x = x*z

        return x
# def trunc(x,k):
#     '''
#     THIS FUNCTION IS USED FOR CNN NETWORKS
    
#     Takes input x, and removes the top and bottom k features by
#     zeroing them. 
#     Inputs:
#         x - 4-dim: [bs,in_ch,out_dim,out_dim]
#         k - truncation parameter
#     Outputs:
#         x - truncated version of input

#     NOTE: Assumes image is square
#     '''
#     x_vals = x.clone().detach()
#     out_dim = x.shape[3]
#     # Now x is dimension [bs, in_ch, out_dim, out_dim]
#     with torch.no_grad():
#         for i in range(x.shape[0]):
#             temp = x_vals[i,:]
#             _, idx_1 = torch.topk(temp.view(-1,out_dim**2),k)
#             _, idx_2 = torch.topk(-1*temp.view(-1,out_dim**2), k)
#             for ch in range(x.shape[1]):
#                 x[i,ch,idx_1[ch]//out_dim,idx_1[ch]%out_dim] = 0
#                 x[i,ch,idx_2[ch]//out_dim,idx_2[ch]%out_dim] = 0
#     return x

class my_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        '''
        Custom convolution layer which manually performs nn.Conv2d. 
        See torch documentation for reference

        NOTE:
        Assumes stride is 1 and padding is zero
        Assumes kernel is square and odd
        '''
        super(my_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # initialize weight and bias 
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Copied directly from torch implementation:
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        '''
        Performs 2d convolution where x is input
        '''
        # build output tensor
        # out = nn.functional.conv2d(x,self.weight,self.bias)
        h = x.shape[-1] - self.kernel_size + 1
        w = x.shape[-2] - self.kernel_size + 1
        bs = x.shape[0]
        out = torch.empty(bs,self.out_channels,h,w).to(self.weight.device)

        # iterate 
        for i_img in range(bs):
            for out_ch in range(self.out_channels):
                for i in range(w):
                    for j in range(h):
                        window = x[i_img,:,i:i+self.kernel_size,j:j+self.kernel_size]
                        out[i_img,out_ch,i,j] = torch.mul(window,self.weight[out_ch]).sum() 
                        out[i_img,out_ch,i,j] += self.bias[out_ch]
        return out

