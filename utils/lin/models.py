import torch
from utils.helpers import *
import torch.nn as nn
import torch.nn.functional as F

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

# EVAL NETWORKS FOR ATTACK SCRIPTS
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

