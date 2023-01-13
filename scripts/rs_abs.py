## this scripts runs sparse rs attack on abs model. 
#NOTE: before running thsi script, move the abs/exp directory, abs/abs_model, 
# and abs/data directories out of abs and put it in root (robust-l0/)

import torch
import torch.nn as nn
import numpy as np


import torch
import argparse
import json
import logging
from tqdm import trange
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)

# from utils.adv_trainer import *
from utils.models import *
from utils.helpers import *
from utils.attack import *

from abs_models import models as mz        # model zoo
from abs_models import utils as u

import torch



def main(args):
    print('Starting Evaluation')
    logging.basicConfig(filename=os.path.join(args.log_dir,'result log%.2f.txt'%args.beta), 
                        level=logging.DEBUG, 
                        format="%(message)s")

    logging.info('Starting evaluation with following configuration:')
    for key in vars(args).keys():
        logging.info('\t%s:%s'%(key,str(vars(args)[key])))
    logging.info('-----------------------------------------------')

    bs = 100
    device = args.device
    model = mz.get_VAE(n_iter=50).to(device)              # ABS do n_iter=1 for speedup (but ess accurate)
    x, y_target = u.get_batch(bs)  
    logits = model(u.n2t(x))             # returns torch.tensor, shape (batch_size, n_channels, nx, ny)

    y = np.argmax(logits.cpu(), axis=1)
    acc = 100*float(np.sum(np.array(y)==y_target)/bs)

    logging.info('clean accuracy: %.2f'%acc)


    beta = args.beta
    queries = args.queries
    restarts = args.restarts

    r_acc, _, _, _ = attack(model, 
                            budget=args.budget, 
                            x=torch.tensor(x),
                            y=torch.tensor(y_target),
                            beta = beta,
                            n_queries=queries,
                            n_restarts=restarts,
                            device=device,
                            log_path=os.path.join(args.log_dir,'rs_log%.2f.txt'%args.beta))

    logging.info('robust accuracy: %.2f'%r_acc)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--log_dir",
        type=str,
        default='new_trained/abs/',
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=12,
        help="l_0 budget of adversary for relevant attacks",
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=10000,
        help="Queries to run sparse-rs attack for. Defaults from the sparse-rs paper user on ABS",
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=5,
        help="Number of random restarts to use in the sparse rs attack. Defaults from the sparse-rs paper user on ABS",
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="name of device to run on",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1,
        help="If input added, overrides experiment beta",
    )


    args = parser.parse_args()
    main(args)
