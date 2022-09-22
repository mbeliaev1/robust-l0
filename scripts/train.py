import torch
import argparse
import json
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# local imports
from utils.adv import adv_trainer


def setup(args):
    '''
    cuda setup, as well as saving of path and variables needed
    '''
    # INIT CUDA #
    #-------------------------------------------------------------------------------------#
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # torch.backends.cudnn.benchmark = True

    # CHECK INPUTS #
    #-------------------------------------------------------------------------------------#
    assert args.arch in ['fc','cnn'], "Unsupported architecture chosen: %s"%args.arch
    # if bs was not set manually, use default
    if args.bs is None:
        args.bs = 128*(args.arch == 'cnn') + 256*(args.arch == 'fc')

    # CREATE DIRS #
    #-------------------------------------------------------------------------------------#
    exp_name = '%s_k%d_p%d_seed%d'%(args.arch,args.k,args.perturb,args.seed)
    exp_dir = os.path.join(args.out_dir,args.exp,exp_name)

    if os.path.isdir(exp_dir):
        raise Exception('%s already exists!!!'%exp_dir)
    else:
        if not os.path.isdir(args.out_dir):
            os.mkdir(args.out_dir)
        if not os.path.isdir(os.path.join(args.out_dir,args.exp)): 
            os.mkdir(os.path.join(args.out_dir,args.exp))
        os.mkdir(exp_dir)
    
    setup_path = os.path.join(exp_dir,'setup.json')
    with open(setup_path, 'w') as f:
        json.dump(vars(args),f,indent=4)

    return device, '/' + exp_dir + '/'

def main(args):
    # SETUP #
    device, save_dir = setup(args)
    print('Running on device: %s\n Results will be saved to %s'%(device,save_dir))
    print('Configuration:')
    for key in vars(args).keys():
        print('\t', key,': ', vars(args)[key])

    trainer = adv_trainer(
                root = root,
                arch = args.arch, 
                k = args.k,
                perturb = args.perturb,
                beta = args.beta,
                seed = args.seed,
                save_dir = save_dir, 
                bs=args.bs, 
                num_iters=args.iters, 
                num_queries=args.queries, 
                num_epochs=args.epochs,
                device=device)

    trainer.run()

    print('Finished Execution')

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
        "--exp",
        type=str,
        default='test',
        help="name of the experiment directory to save results to",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default='new_trained',
        help="directory to save results into",
    )

    # NETWORK PARAMETERS #
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="truncation parameter, use 0 for no truncation",
    )

    parser.add_argument(
        "--perturb",
        type=int,
        default=10,
        help="l_0 budget of adversary to generate adv examples while training",
    )

    parser.add_argument(
        "--beta",
        type=int,
        default=100,
        help="paramater for scaling l_inf norm of adversary attack",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=99,
        help="experiment seed",
    )

    parser.add_argument(
        "--bs",
        type=int,
        default=None,
        help="batchsize of NN, note default is set dependent on net architecture",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="times to repeat the training and attacking cycle",
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=300,
        help="time budget for each rs attack",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=25,
        help="how many epochs per iter",
    )

    args = parser.parse_args()
    main(args)







