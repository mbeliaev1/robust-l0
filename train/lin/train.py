import torch
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# local imports
from utils.lin.adv import adv_trainer

# check cuda
if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

def main():
    k = 10 # if k=0 the network will use the regular VGG net
    perturb = 10
    save_dir = '/new_trained/lin/test/'
    os.mkdir(root+save_dir)
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
    main()







