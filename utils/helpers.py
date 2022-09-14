import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

def prep_CIFAR(root, bs):
    '''
    Preps the CIFAR Dataset from root/datasets/, loading in all
    the classes, using batch size of bs

    Outputs Data without saving
    '''
    # Load the data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root=root+'/datasets/CIFAR/', train=True,
                                        download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                            shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root=root+'/datasets/CIFAR/', train=False,
                                        download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs,
                                            shuffle=False, num_workers=2)

    # Finally compile loaders into Data structure
    Data = {}
    Data['bs']  = bs
    Data['x_test'] = []
    Data['y_test'] = []
    Data['x_train'] = []
    Data['y_train'] = []
   
    # Go through loaders collecting data
    for _, data in enumerate(train_loader, 0):
        Data['x_train'].append(data[0])
        Data['y_train'].append(data[1])

    for _, data in enumerate(test_loader, 0):
        Data['x_test'].append(data[0])
        Data['y_test'].append(data[1])

    Data['x_og'] = Data['x_train'].copy()
    Data['y_og'] = Data['y_train'].copy()

    return Data

def prep_MNIST(root, bs):
    '''
    Preps the MNIST Dataset from root/datasets/, loading in all
    the classes, using batch size of bs

    Outputs Data without saving
    The images are flattened
    '''
    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    trainset = datasets.MNIST(root=root+'/datasets/',train = True,download = True, transform=transform)
    testset = datasets.MNIST(root=root+'/datasets/',train = False,download = True, transform=transform)

    # now the loaders
    bs = 256
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=True)

    # Finally compile loaders into Data structure
    Data = {}
    Data['bs']  = bs
    Data['x_test'] = []
    Data['y_test'] = []
    Data['x_train'] = []
    Data['y_train'] = []
   
    # Go through loaders collecting data
    for _, data in enumerate(train_loader, 0):
        Data['x_train'].append((data[0].reshape(-1,1,1,28*28)))
        Data['y_train'].append(data[1])

    for _, data in enumerate(test_loader, 0):
        Data['x_test'].append((data[0].reshape(-1,1,1,28*28)))
        Data['y_test'].append(data[1])

    Data['x_og'] = Data['x_train'].copy()
    Data['y_og'] = Data['y_train'].copy()

    return Data

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
        x = torch.matmul(x,self.weight.T)
        temp = x_vals.view(-1,1,784)*self.weight
        # temp shape is (bs, out_dim, in_dim)
        val_1, _ = torch.topk(temp,self.k)
        val_2, _ = torch.topk(-1*temp,self.k)
        # val shapes are (bs, out_dim, self.k)
        x -= val_1.sum(axis=-1)
        x += val_2.sum(axis=-1)
        x += self.bias

        ####
        # MORE EFFICIENT IMPLEMENTATION WAS UPDATED AFTER REPORT,
        # IT IS IDENTICAL TO THE BELOW ORIGINAL IMPLEMENTAION, 
        # BUT IF ANY ERROR OCCURS FEEL FREE TO REVERT BACK BY COMMENTING 
        # LINES 121-130, and UNCOMMENTING LINES 140-149. The speedup is roughly 
        # 30%.
        ####

        # OLD IMPLEMENTATION BEGINS
        # x_vals = x.clone().detach()
        # x = torch.matmul(x,self.weight.T)
        # # iterate over the result to apply truncation after
        # for i in range(x.shape[0]):
        #     temp = x_vals[i,:]*self.weight
        #     val_1, _ = torch.topk(temp,self.k)
        #     val_2, _ = torch.topk(-1*temp,self.k)
        #     x[i] -= torch.sum(val_1,dim=1)
        #     x[i] += torch.sum(val_2,dim=1)
        # x += self.bias
        # OLD IMPLEMENTATION ENDS
        
        return x

def trunc(x,k):
    '''
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

def mu_sigma(beta, CIFAR=False):
    '''
    Rescales the original domain given Beta value.
    Inputs:
        beta  - The magnitude by which we scale the domain
        CIFAR - bool to tell if we are using CIFAR or MNIST
    Outputs:
        mu    - New mean for the data
        sigma - New std for the data
    '''
    # min/max pixel values for MNIST/CIFAR datasets
    if CIFAR:
        MIN = -2.429065704345703
        MAX = 2.7537312507629395
    else:
        MIN = -0.42421296
        MAX = 2.8214867
    # transfomration
    mu = MIN - (0.5-(1/(2*beta)))*(MAX-MIN)*beta
    sigma = (MAX-MIN)*beta
    return mu, sigma

class Logger():
    '''
    Logger used within sparse_rs.py to record details on txt file.
    '''
    def __init__(self, log_path):
        self.log_path = log_path

    def log(self, str_to_log):
        with open(self.log_path, 'a') as f:
            f.write(str_to_log + '\n')
            f.flush()

class flatten(object):
    '''
    Flatten into one dimension
    '''
    def __call__(self, sample):
        image = sample[0]
        new_image = torch.flatten(sample[0])
        return (new_image)
