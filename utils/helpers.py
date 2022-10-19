import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np

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

