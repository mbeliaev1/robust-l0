# torch imports
import torch
# other modules
import argparse
import numpy as np 
import foolbox 
from tqdm import trange
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# internal imports
from utils.conv.sparse_rs import RSAttack
from utils.conv.models import *
from utils.helpers import *

def main():
    #---------------------------------------------------------------------------#
    # Parse the only input, path
    parser = argparse.ArgumentParser(description='Runs the pointwise attack on network for both beta=100 and beta=1')
    parser.add_argument("exp_path", help="pass the RELATIVE path of the PARENT directory of your network (net.pth)")
    # optinal arguements
    parser.add_argument("bs", nargs="?", type=int, default=64, help="batch size fed into foolbox")
    parser.add_argument("num_batches", nargs="?", type=int, default=16, help="batch size fed into foolbox")
    parser.add_argument("num_iters", nargs="?", type=int, default=10, help="times to repeat the attack")
    args = parser.parse_args()

    exp_path = root + '/' + args.exp_path + '/'
    bs = args.bs
    num_batches = args.num_batches
    num_iters = args.num_iters
    # since we only tested with k=10, we can simply check the name of dir (rob or og)
    if 'og' in exp_path.split('/'):
        k = 0
    else:
        k = 10
    #---------------------------------------------------------------------------#
    # check cuda
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    # get data loaders
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
    testset = datasets.CIFAR10(root=root+'/datasets/CIFAR/',train = False,download = False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers = 2)
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images = images.numpy()
    # prep the network
    net_path = exp_path + 'net.pth'
    if k == 0:
        net = VGG().to(device)
    else:
        net = rob_VGG(k).to(device)
    net.load_state_dict(torch.load(net_path, map_location=device))
    net.eval()

    # RUN the attack
    for j in range(2):
        if j%2 == 1:
            bounds = (-260,260) # aprox beta=100
        else:
            bounds = (-2.429065704345703,2.7537312507629395)
        # SETUP #
        fmodel = foolbox.models.PyTorchModel(net, bounds=bounds, num_classes=10, channel_axis=1, device=device)
        attack = foolbox.attacks.PointwiseAttack(model=fmodel,distance=foolbox.distances.L0)
        Data = {}
        Data['final_l0'] = []
        # ITERATE #
        print("-----"*8)
        print("running attack with bounds: ",bounds)

        for batch in trange(num_batches):
            images, labels = dataiter.next()
            images = images.numpy()
            # final l0 distances of entire batch
            final_l0 = np.ones(bs)*10000
            # best attacked images 
            final_images = images.copy()

            for _ in range(num_iters):
                # run the attack saving the best adversarial images
                adv_images = attack(images.copy(), labels.numpy().copy())
                out = net(torch.tensor(adv_images).to(device))
                _, pred_adv = torch.max(out.data, 1)
                # calculate L_0 distances for these new images
                perturbed = np.zeros((bs,32,32),dtype=bool)
                for z in range(bs):
                    for ch in range(3):
                        # adds True if pixel at this channel was perturbed
                        perturbed[z] += abs(images[z,ch,:]-adv_images[z,ch,:]>0.0001)
                # this gives list of total perturbed pixels for each image
                total_l0 = perturbed.sum(axis=1).sum(axis=1)
                # we only save on three conditions
                cond_1 = total_l0>0             # there was an actual attack
                cond_2 = total_l0 < final_l0    # the attack was better than the best one
                cond_3 = (pred_adv != labels.to(device)).to('cpu').numpy()  # the attack was succesful
                # save both the images and the best L_0 distances for this iteration
                improved = cond_1*cond_2*cond_3
                final_images[improved] = adv_images[improved]
                final_l0[improved] = np.minimum(total_l0[improved],final_l0[improved])
            # save the best l_0 distances for this batch
            Data['final_l0'].append(final_l0)
        Data['final_l0'] = np.asarray(Data['final_l0'])
        print('Final median l0 distance: ',np.median(Data['final_l0']))

if __name__ == '__main__':
    main()