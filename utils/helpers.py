import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from scipy import linalg
def evaluate(net, x, y, device):
    '''
    gives validation accuracy for dataset and model
    Inputs: 
        net      - torch.nn module with network class
        x        - inputs
        y        - targets
        device   - torch device name

    Returns
        acc
    '''
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in zip(x, y):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * (correct / total)
    return acc

def prep_data(root, bs, dataset):
    '''
    Preps the CIFAR or MNIST Dataset from root/datasets/, loading in all
    the classes, using batch size of bs

    Outputs Data without saving
    '''
    # Load the data
    assert dataset in ['MNIST', 'CIFAR']

    if dataset == 'CIFAR':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])
        trainset = datasets.CIFAR10(root=root+'/datasets/CIFAR/', train=True,
                                            download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=root+'/datasets/CIFAR/', train=False,
                                            download=True, transform=transform_test)

    elif dataset == 'MNIST':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        trainset = datasets.MNIST(root=root+'/datasets/',train = True,
                                    download = True, transform=transform_train)
        testset = datasets.MNIST(root=root+'/datasets/',train = False,
                                    download = True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=bs,
                                            shuffle=True, num_workers=2)
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

class flatten(object):
    '''
    Flatten into one dimension
    '''
    def __call__(self, sample):
        image = sample[0]
        new_image = torch.flatten(sample[0])
        return (new_image)

def toeplitz_1_ch(kernel, input_size):
    # shapes
    k_h, k_w = kernel.shape
    i_h, i_w = input_size
    o_h, o_w = i_h-k_h+1, i_w-k_w+1
    
    # construct 1d conv toeplitz matrices for each row of the kernel
    toeplitz = []
    for r in range(k_h):
        toeplitz.append(linalg.toeplitz(c=(kernel[r,0], *np.zeros(i_w-k_w)), r=(*kernel[r], *np.zeros(i_w-k_w))) ) 

    # construct toeplitz matrix of toeplitz matrices (just for padding=0)
    h_blocks, w_blocks = o_h, i_h
    h_block, w_block = toeplitz[0].shape

    W_conv = np.zeros((h_blocks, h_block, w_blocks, w_block))

    for i, B in enumerate(toeplitz):
        for j in range(o_h):
            W_conv[j, :, i+j, :] = B

    W_conv.shape = (h_blocks*h_block, w_blocks*w_block)

    return W_conv