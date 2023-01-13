# from xmlrpc.client import boolean
import torch
import foolbox
import argparse
import json
import logging
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

def run_pw_attack(model,data_x,data_y,num_iters,beta,device):
    '''
    Runs pointwise attack on model using x,y
    NOTE: assumes x,y is list of tensors 
    '''

    # calculate bounds
    in_b = np.array([0,1])
    out_b = in_b/(1/beta)
    out_b += ((1-beta)/2)
    logging.critical('Using bounds: %.2f,%.2f'%(out_b[0],out_b[1]))

    # load models
    fmodel = foolbox.models.PyTorchModel(model=model,
                                        bounds=(out_b[0],out_b[1]),
                                        num_classes=10,
                                        channel_axis=1,
                                        device=device)

    attack = foolbox.attacks.PointwiseAttack(model=fmodel,
                                            distance=foolbox.distances.L0)
    # run attack
    advs = []
    advs_dist = []
    # for batch in trange(num_batches):
    for batch in trange(len(data_x)):
        x, y = data_x[batch].numpy(), data_y[batch].numpy()
        bs = x.shape[0]
        # final_l0 = np.ones(bs)*100000
        # final_x = x.copy() # numpy form
        adversarial_dists = []
        adversarials = []
        for _ in range(num_iters):
            adv_x = attack(x, y, unpack=False)
            distance = [x.distance.value for x in adv_x]
            original_class = [x.original_class for x in adv_x]
            output = [x.output for x in adv_x]
            adversarial_dists.append(distance)
            adversarials.append({'distance': distance,'label': original_class, 'scores':  output})

        # find min
        adversarial_dists = np.asarray(adversarial_dists)
        mindex = np.argmin(adversarial_dists,axis=0)
        min_dists = [adversarial_dists[i_mindex, i] for i,i_mindex in enumerate(mindex)]
        min_dists = np.asarray(min_dists)

        min_advs = {'distance': [], 'label': [], 'scores': []}
        for i, i_mindex in enumerate(mindex):
            for key in min_advs.keys():
                min_advs[key].append(adversarials[i_mindex][key][i])
        
        # save
        advs_dist.append(min_dists)
        advs.append(min_advs)

    advs_dist = np.asarray(advs_dist)
    logging.critical('Final median l0 distance: %.3f',np.median(advs_dist))

    return advs, advs_dist

def setup_config(cfg_path, device):
    '''
    loads the relevant information using given config path (json)
    
    Output:
        model   - torch.nn module
        data    - data formatted as in helpers.prep_data()
    '''
    # load config file
    config = json.load(open(cfg_path,'rb'))
    logging.critical('\n')
    logging.critical('-----------------------------------------------')
    logging.critical('Evaluating configuration:')
    logging.critical('-----------------------------------------------')
    # for key in config.keys(): # could replace with important keys


    for key in ['dataset', 'no_adv', 'trunc_type', 'k', 'beta', 'seed']:
        logging.critical('\t%s:%s'%(key,str(config[key])))
    
    data = prep_data(root, bs=args.samples, dataset=config['dataset'])

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

def eval_config(args, model, beta, data, device):
    # clean accuracy
    acc = evaluate(model, data['x_test'], data['y_test'], device)
    logging.critical('Clean Accuracy: %.3f'%acc)

    # robust accuracy one  batch
    advs, advs_dist = run_pw_attack(model=model, 
                                data_x=data['x_test'][0:1],
                                data_y=data['y_test'][0:1],
                                num_iters=args.iters,
                                beta=args.beta,
                                device=device)

    logging.critical('Median values:')
    logging.critical(str(advs_dist))

def main(args):
    print('Starting Evaluation . . .')
    # setup device
    device = setup_device(args)
    
    # setup logger
    logging.basicConfig(filename=os.path.join(args.eval_dir,args.log_name), 
                        level=logging.CRITICAL, 
                        format="%(message)s")

    logging.critical('Starting evaluation with following configuration:')
    for key in vars(args).keys():
        logging.critical('\t%s:%s'%(key,str(vars(args)[key])))
    logging.critical('-----------------------------------------------')
    
    # create list of config files to evaluate:
    cfg_paths = [os.path.join(root, name)
                 for root, dirs, files in os.walk(args.eval_dir)
                 for name in files
                 if name.endswith((".json"))]
    
    # load model and data (NOTE: loading dataset redundancy)
    for i in trange(len(cfg_paths)):
        # print('Evaluating model %d/%d . . .'%(i,len(cfg_paths)))
        model, data, config = setup_config(cfg_paths[i], device)
        eval_config(args, model, args.beta, data, device)

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
        default = 'eval_pw.txt',
        help="Name of log to save to (always saves to cfg_path)"
    )

    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="number of iterations to run pointwise attack",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="total number of images to test",
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
        help="beta value NOTE: for now only 1",
    )


    args = parser.parse_args()
    main(args)
