import torch
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# local imports
from utils.conv.adv import adv_trainer

def main():
    # check cuda
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
        
    k = 10 # if k=0 the network will use the regular VGG net
    perturb = 10
    save_dir = '/new_trained/conv/test/'
    os.mkdir(root+save_dir)
    trainer = adv_trainer(
                root = root, 
                k = k,
                perturb = perturb,
                beta = 100,
                seed = 104,
                save_dir = save_dir, 
                bs=128, 
                num_iters=10, 
                num_queries=300, 
                num_epochs=25,
                device=device)
    trainer.run()

if __name__ == '__main__':
    main()







