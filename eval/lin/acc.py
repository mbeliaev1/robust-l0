# torch imports
import torch
# other modules
import argparse
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# internal imports
from utils.lin.sparse_rs import RSAttack
from utils.lin.models import *
from utils.helpers import *



def main():
    # Parse the only input, path
    parser = argparse.ArgumentParser(description='Calculates the clean accuracy of network using sparse rs')
    parser.add_argument("exp_path", help="pass the RELATIVE path of the PARENT directory of your network (net.pth)")
    args = parser.parse_args()
    exp_path = root + '/' + args.exp_path + '/'
    # since we only tested with k=10, we can simply check the name of dir (rob or og)
    if 'og' in exp_path.split('/'):
        k = 0
    else:
        k = 10

    # check cuda
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # load data and network
    Data = prep_MNIST(root, bs=256)
    net_path = exp_path + 'net.pth'
    if k == 0:
        net = L_Net().to(device)
    else:
        net = r_L_Net(k).to(device)
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()
    # test the accuracy 
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in zip(Data['x_test'],Data['y_test']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * (correct / total)

    print("Clean Accuracy: ",acc,'%')

if __name__ == '__main__':
    main()