import torch
import argparse
import json
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# local imports
from utils.lin.adv import adv_trainer


def setup(args):
    '''
    cuda setup, as well as saving of path and variables needed
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # torch.backends.cudnn.benchmark = True

    out_dir = os.join(args.out_dir,args.name)
    if not os.path.isdir(out_dir): os.mkdir(out_dir)
    
    setup_path = os.join(out_dir,'setup.json')
    with open(setup_path, 'w') as f:
        json.dump(vars(args),f)

    return

def main(args):
    # SETUP #
    # check cuda
    setup(args)
    breakpoint()



    k = 10 # if k=0 the network will use the regular FC net
    perturb = 10
    save_dir = '/new_trained/lin/test/'
    os.mkdir(root+save_dir)
    print(device)
    trainer = adv_trainer(
                root = root, 
                k = k,
                perturb = perturb,
                beta = 100,
                seed = 25,
                save_dir = save_dir, 
                bs=256, 
                num_iters=10, 
                num_queries=300, 
                num_epochs=25,
                device=device)
    trainer.run()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Algorithm Name
    parser.add_argument(
        "--arch",
        type=str,
        default='fc',
        help="name of architecture to use, either cnn or fc",
    )

    # experiment name for saving models
    parser.add_argument(
        "--name",
        type=str,
        default='test',
        help="name of the experiment directory to save results to",
    )

    # name of dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default='MNIST',
        help="dataset to use, either MNIST or CIFAR",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default='new_trained',
        help="directory to save results into",
    )

    args = parser.parse_args()

    main(args)







