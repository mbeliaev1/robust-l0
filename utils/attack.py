# torch imports
import torch
import numpy as np
from tqdm import trange

# internal
import randomizedAblation.utils as utils
import randomizedAblation.utils_color as utils_cifar
from utils.attacks.sparse_rs import RSAttack
from utils.helpers import evaluate, beta_transform, beta_revert
# attack function, did not want to implement as full class yet

class SingleChannelModel():
    """ reshapes images to rgb before classification
        i.e. [N, 1, H, W x 3] -> [N, 3, H, W]
    """
    def __init__(self, model):
        if isinstance(model, torch.nn.Module):
            assert not model.training
        self.model = model

    def __call__(self, x):
        # print('in single',x.view(x.shape[0], 3, x.shape[2], x.shape[3] // 3).shape)
        # print('call',x.shape)
        return self.model(x.view(x.shape[0], 3, x.shape[2], x.shape[3] // 3))

class Ablation_wrapper():
    """ wraps the randomized ablation model so it can be used with this attack
    """
    def __init__(self, model):
        if isinstance(model, torch.nn.Module):
            assert not model.training
        self.model = model

    def __call__(self, x):
        # values below are for MNIST
        predicted = utils.predict(x.to('cuda:0'), self.model, 45, 10000, 0.05)

        # values below are for CIFAR 
        # default value is 10000 samples, but we set to 5000 due to memeory capacity
        # predicted = utils_cifar.predict(x.to('cuda:0'), self.model, 75, 2000, 0.05)
        predicted[predicted<0] = torch.tensor(np.random.randint(0,10,(predicted<0).sum().item()))
        logits = torch.nn.functional.one_hot(predicted, 10).float()
        return logits.to('cuda:0')

class BetaModel():
    '''
    Scales input using beta for l_inf bound

    beta_revert shapes input back so that original pixel values are from [0,1]
    and attacked pixel values go beyond that
    '''
    def __init__(self, model, beta):
        if isinstance(model, torch.nn.Module):
            assert not model.training
        self.model = model
        self.beta = beta

    def __call__(self, x):
        x_og = beta_revert(x, self.beta)
        return self.model(x_og)


def attack_batch(net, x, y, adversary):
    '''
    Helper function for attack below
    ''' 
    # reshape x to be 1 ch
    bs, ch, h ,w = x.shape
    x = x.view(bs, 1, h, w*ch)
    # print('in call',x.shape)
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

    return r_acc, adv[fooled].view(-1,ch,h,w), y[i_adv], i_adv

def attack(net, budget, x, y, beta, **kwargs):
    '''
    This is our general untargetedattack function which 
    returns a list of perturbed examples given EITHER a batch
    or an entire dataset
    
    Inputs:
        net         - nn.Module that we will be attacking 
        budget      - L_0 budget (maximum perturbation)
        x           - input data of either batches in the form of list
                        or tensor of shape (bs, c, w, h)
        y           - labels of input data
        beta        - scaling for l_inf
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

    #-----------------------------------------#
    # NOTE: Wrapping ablation model manually
    # net = Ablation_wrapper(net)
    #-----------------------------------------#

    # determine if input is multiple batches or single
    is_batch = True
    if isinstance(x, list):
        is_batch = False
        assert (len(x[0].shape) == 4), "Improper data format given for x"
        if x[0].shape[1] == 3:
            net = SingleChannelModel(net)
            print("1 single channel mode")
    else:
        assert (len(x.shape) == 4), "Improper data format given for x"
        if x.shape[1] == 3:
            net = SingleChannelModel(net)
            print("2 single channel mode")

    
    net = BetaModel(net, beta)
    adversary = RSAttack(predict=net,
                    norm='L0',
                    eps=budget,
                    n_queries=kwargs['n_queries'],
                    n_restarts=kwargs['n_restarts'],
                    device=device,
                    log_path=kwargs['log_path']
                    )

    if is_batch:
        x_beta = beta_transform(x,beta)
        r_acc, x_adv, y_adv, i_adv = attack_batch(net, x_beta.to(device), y.to(device), adversary)
        return r_acc, beta_revert(x_adv, beta), y_adv, i_adv

    else:
        tot_batches = len(x)
        x_adv, y_adv, i_adv = [], [], []
        total = 0
        weighted_acc = 0
        count = 0
        for x_batch, y_batch in zip(x,y):
            print('Running sparse-rs attack: %d/%d batches done'%(count,tot_batches), end='\r')
            count+=1
            x_batch_beta = beta_transform(x_batch,beta)
            # run batch and store in list
            r_acc, b_x_adv, b_y_adv, b_i_adv = attack_batch(net, 
                                                            x_batch_beta.to(device), 
                                                            y_batch.to(device), 
                                                            adversary)
            b_x_adv = beta_revert(b_x_adv, beta)
            # some batches might have different weight
            bs = len(y_batch)
            weighted_acc += r_acc*bs
            total += bs
            
            x_adv.append(b_x_adv) 
            y_adv.append(b_y_adv)
            i_adv.append(b_i_adv)
        
        return weighted_acc/total, x_adv, y_adv, i_adv