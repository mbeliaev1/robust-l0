# from xmlrpc.client import boolean
import torch
import argparse
import json
import logging
import pickle
from tqdm import trange
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# local imports
from utils.models import load_net
from utils.helpers import *
from utils.attack import attack

def setup_config(cfg_path, device):
    '''
    loads the relevant information using given config path (json)
    
    Output:
        model   - torch.nn module
        data    - data formatted as in helpers.prep_data()
    '''
    # load config file
    config = json.load(open(cfg_path,'rb'))
    logging.info('\n')
    logging.info('-----------------------------------------------')
    logging.info('Evaluating configuration:')
    logging.info('-----------------------------------------------')
    # for key in config.keys(): # could replace with important keys

    for key in ['dataset', 'no_adv', 'trunc_type', 'k']:
        logging.info('\t%s:%s'%(key,str(config[key])))
    

    data = prep_data(root, bs=100, dataset=config['dataset'])

    # load model
    result_path = cfg_path[0:-len('/setup.json')]
    model = load_net(result_path, device, input_shape=data['x_test'][0].shape)
    model.to(device)

    return model, data, config

def setup_device(args):
    '''
    setup cuda for entire experiment
    
    Output:
        model   - torch.nn module
        data    - data formatted as in helpers.prep_data()
        device  - device used for evaluation (model loaded here)
        logger  - output file to log to
    '''

    if torch.cuda.is_available() & args.device.startswith('cuda'):
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.multiprocessing.set_sharing_strategy('file_system')

    return device

def eval_config(args, model, budget, beta, data, device):
    # clean accuracy
    if budget == 0:
        r_acc = evaluate(model, data['x_test'], data['y_test'], device)
        logging.info('\tClean Accuracy: %.3f'%r_acc)
    else:
        # robust accuracy one  batch
        r_acc, _, _, _ = attack(model, 
                                budget=budget, 
                                x=data['x_test'][0:5],
                                y=data['y_test'][0:5],
                                beta = beta,
                                n_queries=args.queries,
                                n_restarts=args.restarts,
                                device=device,
                                log_path='temp.log')

        logging.info('\tbudget=%d: %.3f'%(budget,r_acc))

    return r_acc

def main(args):
    print('Starting Evaluation . . .')
    # setup device
    device = setup_device(args)
    # setup logger
    logging.basicConfig(filename=os.path.join(args.eval_dir,args.log_name), 
                        level=logging.DEBUG, 
                        format="%(message)s")

    logging.info('Starting evaluation with following configuration:')
    for key in vars(args).keys():
        logging.info('\t%s:%s'%(key,str(vars(args)[key])))
    logging.info('-----------------------------------------------')
    
    # create list of config files to evaluate:
    cfg_paths = [os.path.join(root, name)
                 for root, dirs, files in os.walk(args.eval_dir)
                 for name in files
                 if name.endswith((".json"))]
    
    # load model and data (NOTE: loading dataset redundancy)
    for i in trange(len(cfg_paths)):
        # print('Evaluating model %d/%d . . .'%(i,len(cfg_paths)))
        result_arr = []
        model, data, config = setup_config(cfg_paths[i], device)
        # go over all budgets:
        for budget in np.arange(0,101,2):
            r_acc = eval_config(args, model, budget, args.beta, data, device)
            result_arr.append(r_acc)
        result_str = 'result_%s_%d_%d.p'%(config['dataset'],config['k'],args.beta)
        pickle.dump(result_arr,open(os.path.join(args.eval_dir,result_str),'wb'))
        print(result_arr)
    print('Finished Evaluation!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--eval_dir",
        type=str,
        help="Location of directory to evaluate, note it will evaluate all sub directories"
    )

    parser.add_argument(
        "--log_name",
        type=str,
        default = 'final_rs.txt',
        help="Name of log to save to (always saves to cfg_path)"
    )

    parser.add_argument(
        "--queries",
        type=int,
        default=500,
        help="Queries to run sparse-rs attack for. Defaults from the sparse-rs paper user on ABS",
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=1,
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
        type=int,
        default=1,
        help="beta scaling to use for attack",
    )


    args = parser.parse_args()
    main(args)
