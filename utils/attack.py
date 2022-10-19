# torch imports
import torch
import numpy as np
from tqdm import trange

# internal
from utils.attacks.sparse_rs import RSAttack
from utils.helpers import evaluate
# attack function, did not want to implement as full class yet


def attack_batch(net, x, y, adversary):
    '''
    Helper function for attack below
    ''' 
    # check original classification
    output = net(x)
    pred = (output.max(1)[1] == y).float()

    # attack images that need to be fooled
    ind_to_fool = (pred == 1).nonzero(as_tuple=False).squeeze()
    _, adv = adversary.perturb(x[ind_to_fool], y[ind_to_fool])

    n_missed = len(x) - len(ind_to_fool) # number of misclassified
    # classify perturbed images and compute robust accuracy
    output = net(adv)
    fooled = (output.max(1)[1] != y[ind_to_fool])
    i_adv = ind_to_fool[fooled]
    r_acc = (1-(len(i_adv) + n_missed)/len(x))*100

    # NOTE: To compute norms run this code:
    # pixels = x[i_adv] != new_adv
    # norms = (pixels.sum(axis=-1).sum(axis=-1))

    return r_acc, adv[fooled], y[i_adv], i_adv

def attack(net, k, x, y, **kwargs):
    '''
    This is our general untargetedattack function which 
    returns a list of perturbed examples given EITHER a batch
    or an entire dataset
    
    Inputs:
        net         - nn.Module that we will be attacking 
        k           - L_0 budget (maximum perturbation)
        x           - input data of either batches in the form of list
                        or tensor of shape (bs, c, w, h)
        y           - labels of input data
        n_queries   - time budget for each attack SPARSE RS
        n_restarts  - number of restarts SPARSE RS
        device      - where to train network
        log_path    - text file to log results to
        
    Outputs:
    EITHER LIST if dataset list is given, or individual elements
        adv         - (list) of MISCLASSIFIED adversary examples, 
                        does not need to be same length as r_acc
        i_adv       - (list) of indexes of adversary examples corresponding 
                        to i_adv from original dataset
        r_acc       - (float) accuracy on the set with misclassified included
        norms       - l0 norm of adversary examples (how many pixels perturbed)
    ''' 
    
    device = kwargs['device']
    
    # determine if input is multiple batches or single
    is_batch = True
    if isinstance(x, list):
        is_batch = False
        assert (len(x[0].shape) == 4), "Improper data format given for x"
    else:
        assert (len(x.shape) == 4), "Improper data format given for x"

    adversary = RSAttack(predict=net,
                    norm='L0',
                    eps=k,
                    n_queries=kwargs['n_queries'],
                    n_restarts=kwargs['n_restarts'],
                    device=device,
                    log_path=kwargs['log_path']
                    )

    if is_batch:
        return attack_batch(net, x, y, adversary)

    else:
        x_adv, y_adv, i_adv = [], [], []
        total = 0
        weighted_acc = 0
        for x_batch, y_batch in zip(x,y):
            # run batch and store in list
            r_acc, b_x_adv, b_y_adv, b_i_adv = attack_batch(net, 
                                                              x_batch.to(device), 
                                                              y_batch.to(device), 
                                                              adversary)
            # some batches might have different weight
            bs = len(y_batch)
            weighted_acc += r_acc*bs
            total += bs
            
            x_adv.append(b_x_adv) 
            y_adv.append(b_y_adv)
            i_adv.append(b_i_adv)
        
        return weighted_acc/total, x_adv, y_adv, i_adv