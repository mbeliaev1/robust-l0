from curses.ascii import BS
import torch
import argparse
import numpy as np 
import foolbox 
from tqdm import trange
import json
import pickle
# PATHING
import os
import sys
root = os.path.abspath(os.curdir)
sys.path.append(root)
# internal imports
from utils.sparse_rs import RSAttack
from utils.models import *
from utils.helpers import *



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

    # LOAD CONFIG #
    #-------------------------------------------------------------------------------------#
    save_dir = os.path.join(args.load_dir,args.exp,args.name) + '/'
    config = json.load(open(save_dir+'setup.json','rb'))
    print('Loading configuration:')
    for key in config.keys():
        print('\t', key,': ', config[key])

    # CHECK INPUTS #
    assert args.mode in ['rs','acc','pw','all'], "Unsupported eval mode chosen: %s"%args.mode
        
    # setup_path = os.path.join(exp_dir,'setup.json')
    # with open(setup_path, 'w') as f:
    #     json.dump(vars(args),f,indent=4)

    return device, config, save_dir

def main(args):
    # SETUP #
    device, config, save_dir = setup(args)
    net_path = save_dir + 'net.pth'
    print('Running on device: %s\n Results will be saved to %s'%(device,save_dir))

    print('Eval Configuration:')
    for key in vars(args).keys():
        print('\t', key,': ', vars(args)[key])

    # load data
    if config['arch'] == 'fc':
        Data = prep_MNIST(root, bs=config['bs'])
    elif config['arch'] == 'cnn':
        Data = prep_CIFAR(root, bs=config['bs'])

    # evaluate based on mode
    if args.mode in ['acc','all']:
        # Clean Accuracy Test # 
        if config['arch'] == 'fc':
            if config['k'] == 0:
                net = L_Net().to(device)
            else:
                net = r_L_Net(config['k']).to(device)
        elif config['arch'] == 'cnn':
            if config['k'] == 0:
                net = VGG().to(device)
            else:
                net = rob_VGG(config['k']).to(device)    
        net.load_state_dict(torch.load(net_path, map_location=device))
        net.eval()
        # test the accuracy 
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in zip(Data['x_test'],Data['y_test']):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * (correct / total)

        print("Clean Accuracy: ",acc,'%')
        pickle.dump(acc, open(save_dir+'acc.p','wb'))


    if args.mode in ['rs','all']:
        # SparseRS l0 attack # 
        if config['arch'] == 'fc':
            # use eval beta
            mu, sigma = mu_sigma(args.beta, CIFAR=False)
            if config['k'] == 0:
                net = L_Net_eval(mu,sigma).to(device)
            else:
                net = r_L_Net_eval(config['k'],mu,sigma).to(device)
        elif config['arch'] == 'cnn':
            mu, sigma = mu_sigma(args.beta, CIFAR=False)
            if config['k'] == 0:
                net = VGG_eval(mu,sigma).to(device)
            else:
                net = rob_VGG_eval(config['k'],mu,sigma).to(device)  
        net.load_state_dict(torch.load(net_path, map_location=device))
        net.eval()
        # test the accuracy 
        adversary = RSAttack(config['arch'],
                            net,
                            norm='L0',
                            eps=args.perturb,
                            n_queries=args.queries,
                            n_restarts=1,
                            seed=args.seed,
                            device=device,
                            log_path=save_dir+'rs_log.txt'
                            )

        # First compute the % robust accuracy on test set for only one batch
        all_acc = []
        # for i in trange(len(Data['x_test'])):
        # 39 full batches in the dataset
        print('Running Sprase RS on test set')
        for i in trange(39):
            x = (Data['x_test'][i].to(device)-mu)/sigma
            y = Data['y_test'][i].to(device)
            with torch.no_grad():
                # find points originally correctly classified
                output = net(x)
                pred = (output.max(1)[1] == y).float().to(device)
                ind_to_fool = (pred == 1).nonzero(as_tuple=False).squeeze()
                # breakpoint()
                # preform the attack on corresponding indeces and save
                _, adv = adversary.perturb(x[ind_to_fool], y[ind_to_fool])
                # analyze the attack
                output = net(adv.to(device))
                r_acc = (output.max(1)[1] == y[ind_to_fool]).float().to(device)
                adversary.logger.log('robust accuracy {:.2%}'.format(r_acc.float().mean()))
                all_acc.append(r_acc.float().sum().cpu().numpy()/len(pred)*100)
        all_acc = np.asarray(all_acc)

        print("Robust Accuracy w/ Sparse RS: ",np.mean(all_acc),'%')
        pickle.dump(all_acc, open(save_dir+'rs.p','wb'))

    if args.mode in ['pw','all']:
        # Pointwise Attack # 
        num_iters = 10
        bs = 64
        num_batches = 16

        if config['arch'] == 'fc':
            transform = transforms.Compose(
                [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)),flatten()])
            testset = datasets.MNIST(root=root+'/datasets/',train = False,download = False, transform=transform)
            if config['k'] == 0:
                net = L_Net().to(device)
            else:
                net = r_L_Net(config['k']).to(device)
                
        elif config['arch'] == 'cnn':
            transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),])
            testset = datasets.CIFAR10(root=root+'/datasets/CIFAR/',train = False,download = False, transform=transform)
            if config['k'] == 0:
                net = VGG().to(device)
            else:
                net = rob_VGG(config['k']).to(device)    
        # get batch of images#
        test_loader = torch.utils.data.DataLoader(testset, batch_size=bs, shuffle=False, num_workers = 2)
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        images = images.numpy()

        # SETUP #
        net.load_state_dict(torch.load(net_path, map_location=device))
        net.eval()
        bounds = (images.min(),images.max())
        fmodel = foolbox.models.PyTorchModel(net, bounds=bounds, num_classes=10, channel_axis=1, device=device)
        attack = foolbox.attacks.PointwiseAttack(model=fmodel,distance=foolbox.distances.L0)
        Data = {}
        Data['final_l0'] = []
        # ITERATE #
        print("-----"*8)
        print("running PW attack with bounds: ",bounds)


        for batch in trange(num_batches):
            images, labels = dataiter.next()
            images = images.numpy()
            # final l0 distances of entire batch (64 bs)
            final_l0 = np.ones(bs)*10000
            # best attacked images 
            final_images = images.copy()

            # num iters 10
            for _ in range(num_iters):
                # run the attack saving the best adversarial images
                adv_images = attack(images.copy(), labels.numpy())
                out = net(torch.tensor(adv_images).to(device))
                _, pred_adv = torch.max(out.data, 1)
                # calculate L_0 distances for these new images
                if config['arch'] == 'fc':
                    perturbed = np.zeros((bs,28*28),dtype=bool)
                    for z in range(bs):
                        perturbed[z] += abs(images[z,:]-adv_images[z,:]>0.0001)
                    total_l0 = perturbed.sum(axis=1)
                elif config['arch']== 'cnn':
                    perturbed = np.zeros((bs,32,32),dtype=bool)
                    for z in range(bs):
                        for ch in range(3):
                            # adds True if pixel at this channel was perturbed
                            perturbed[z] += abs(images[z,ch,:]-adv_images[z,ch,:]>0.0001)
                    total_l0 = perturbed.sum(axis=1).sum(axis=1)
                # this gives list of total perturbed pixels for each image
                # we only save on three conditions
                cond_1 = total_l0>0             # there was an actual attack
                cond_2 = total_l0 < final_l0    # the attack was better than the best one
                cond_3 = (pred_adv != labels.to(device)).to('cpu').numpy()  # the attack was succesful
                # save both the images and the best L_0 distances for this iteration
                improved = cond_1*cond_2*cond_3
                final_images[improved] = adv_images[improved]
                final_l0[improved] = np.minimum(total_l0[improved],final_l0[improved])
            # save the best l_0 distances for this batch
            Data['final_l0'].append(final_l0)

        Data['final_l0'] = np.asarray(Data['final_l0'])
        print('Final median l0 distance: ',np.median(Data['final_l0']))

        pickle.dump(Data['final_l0'], open(save_dir+'pw.p','wb'))


    print('Finished Execution')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    # Algorithm Name

    parser.add_argument(
        "--mode",
        type=str,
        default='acc',
        help="name of the mode/attack model to use for evaluation, supported options: rs, acc, pw, all",
    )

    # experiment name for saving models
    parser.add_argument(
        "--exp",
        type=str,
        default='test',
        help="name of the experiment directory to load results from",
    )

    # experiment name for saving models
    parser.add_argument(
        "--name",
        type=str,
        help="name of the actual run (could use variables later)",
    )

    parser.add_argument(
        "--load_dir",
        type=str,
        default='new_trained',
        help="directory to load results from",
    )

    parser.add_argument(
        "--perturb",
        type=int,
        default=10,
        help="l_0 budget of adversary for attack magnitude (optional)",
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
        "--queries",
        type=int,
        default=300,
        help="experiment seed",
    )

    args = parser.parse_args()
    main(args)

