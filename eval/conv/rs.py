# torch imports
import torch
# other modules
import argparse
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# internal imports
from utils.conv.sparse_rs import RSAttack
from utils.conv.models import *
from utils.helpers import *



def main():
    # Parse the only input path
    parser = argparse.ArgumentParser(description='Calculates robust accuracy of network')
    parser.add_argument("exp_path", help="pass the RELATIVE path of the PARENT directory of your network (net.pth)")
    # optional arguement
    parser.add_argument("num_queries", nargs="?", type=int, default=300, help="num queries to run the attack for")
    parser.add_argument("beta", nargs="?", type=int, default=100, help="beta value (domain scaling)")
    parser.add_argument("perturb", nargs="?", type=int, default=10, help="adversary budget")
    parser.add_argument("num_batches", nargs="?", type=int, default=39, help="how many batches to evaluate from the MNIST testset (batchsize is 256)")
    args = parser.parse_args()
    exp_path = root + '/' + args.exp_path + '/'
    num_queries = args.num_queries
    num_batches = args.num_batches
    perturb = args.perturb
    beta = args.beta
    # since we only tested with k=10, we can simply check the name of dir (rob or og)
    if 'og' in exp_path.split('/'):
        k = 0
    else:
        k = 10

    # check cuda
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # load data and network
    Data = prep_CIFAR(root, bs=256)
    mu, sigma = mu_sigma(beta, CIFAR=True)
    net_path = exp_path + 'net.pth'
    if k == 0:
        eval_net = VGG_eval(mu,sigma).to(device)
    else:
        eval_net = rob_VGG_eval(k,mu,sigma).to(device)
    eval_net.load_state_dict(torch.load(net_path, map_location=device))
    eval_net.eval()
    # RUN THE ATTACK
    # We use the double the queries as we wanta  more accurate r_test value (one batch)
    adversary = RSAttack(eval_net,
                            norm='L0',
                            eps=perturb,
                            n_queries=num_queries,
                            n_restarts=1,
                            seed=12345,
                            device=device,
                            log_path=exp_path+'log_temp.txt'
                            )

    # First compute the % robust accuracy on test set for only one batch
    all_acc = []
    # for i in trange(len(Data['x_test'])):
    # 39 full batches in the dataset
    for i in range(num_batches):
        x = (Data['x_test'][i].to(device)-mu)/sigma
        y = Data['y_test'][i].to(device)
        with torch.no_grad():
            # find points originally correctly classified
            output = eval_net(x)
            pred = (output.max(1)[1] == y).float().to(device)
            ind_to_fool = (pred == 1).nonzero(as_tuple=False).squeeze()
            # preform the attack on corresponding indeces and save
            _, adv = adversary.perturb(x[ind_to_fool], y[ind_to_fool])
            # analyze the attack
            output = eval_net(adv.to(device))
            r_acc = (output.max(1)[1] == y[ind_to_fool]).float().to(device)
            adversary.logger.log('robust accuracy {:.2%}'.format(r_acc.float().mean()))
            all_acc.append(r_acc.float().mean().cpu().numpy()*100)
    all_acc = np.asarray(all_acc)
    print("Robust Accuracy: ",np.mean(all_acc),'%')

if __name__ == '__main__':
    main()