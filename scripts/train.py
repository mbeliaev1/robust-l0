# from xmlrpc.client import boolean
import torch
import argparse
import json
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# local imports
from utils.adv_trainer import adv_trainer


def setup(args):
    '''
    cuda setup, as well as saving of path and variables needed
    '''
    # INIT CUDA #
    #-------------------------------------------------------------------------------------#
    if torch.cuda.is_available() & args.device.startswith('cuda'):
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # torch.backends.cudnn.benchmark = True

    # CHECK INPUTS #
    #-------------------------------------------------------------------------------------#
    # assert args.arch in ['fc','cnn'], "Unsupported architecture chosen: %s"%args.arch

    # CREATE DIRS #
    #-------------------------------------------------------------------------------------#
    # directory structure for experiments
    dir_list = ['%s'%args.out_dir, 
                '%s'%args.exp, 
                '%s'%args.dataset,
                '%s'%args.cfg_name,
                '%s'%args.trunc_type,
                'k_%d'%args.k,
                'adv_%d'%args.perturb,
                '%d'%args.seed]

    # create dir structure
    save_dir = ''
    for dirname in dir_list:
        save_dir += '%s/'%dirname
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
    
    setup_path = os.path.join(save_dir,'setup.json')
    with open(setup_path, 'w') as f:
        json.dump(vars(args),f,indent=4)

    return args.device, '/' + save_dir

    # exp_dir = os.path.join(args.out_dir,args.exp,exp_name)
    # if os.path.isdir(exp_dir):
    #     raise Exception('%s already exists!!!'%exp_dir)
    # else:
    #     if not os.path.isdir(args.out_dir):
    #         os.mkdir(args.out_dir)
    #     if not os.path.isdir(os.path.join(args.out_dir,args.exp)): 
    #         os.mkdir(os.path.join(args.out_dir,args.exp))
    #     os.mkdir(exp_dir)
    


def main(args):
    # SETUP #
    device, save_dir = setup(args)
    print('Running on device: %s\n Results will be saved to %s'%(device,save_dir))
    print('Configuration:')
    for key in vars(args).keys():
        print('\t', key,': ', vars(args)[key])


    trainer = adv_trainer(
                root = root,
                cfg_name = args.cfg_name, 
                trunc_type = args.trunc_type,
                dataset = args.dataset,
                k = args.k,
                perturb = args.perturb,
                bs=args.bs, 
                num_iters=args.iters, 
                num_epochs=args.epochs,
                num_queries=args.queries, 
                no_adv = args.no_adv,
                lr = args.lr,
                momentum = args.momentum,
                decay = args.decay,
                embedding = args.embedding,
                seed = args.seed,
                save_dir = save_dir, 
                device=device)


    trainer.run()

    print('Finished Execution')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--cfg_name",
        type=str,
        default='cnn_large',
        help="name of architecture to use, either VGG11, 19, or fc"
    )

    parser.add_argument(
        "--trunc_type",
        type=str,
        default='clip',
        help="name of truncation structure to use"
    )

    parser.add_argument(
        "--no_adv",
        action = 'store_true',
        help="removes the adversarial training component, but still performs robust test with perturb"
    )

    parser.add_argument(
        "--device",
        type=str,
        default='cuda:0',
        help="name of device to run on",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default='MNIST',
        help="name of the dataset",
    )
    # experiment name for saving models
    parser.add_argument(
        "--exp",
        type=str,
        default='test_mnist',
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
        default=12,
        help="l_0 budget of adversary to generate adv examples while training",
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
        default=128,
        help="batchsize of NN, note default is set dependent on net architecture",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="initial learning rate",
    )
    
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="initial momentum",
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=0.5,
        help="decay of learning rate and momentum",
    )

    parser.add_argument(
        "--embedding",
        type=int,
        default=256,
        help="size of embedding in fc network creating classifier",
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=5,
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







