# torch imports
from torch._C import TensorType
import torch.nn as nn
from torch.serialization import save 
import torch.optim as optim
# other modules
import numpy as np
import time
from tqdm import trange
import pickle
# internal imports
from utils.lin.sparse_rs import RSAttack
from utils.lin.models import *
from utils.helpers import *

class adv_trainer():
    def __init__(self,
                root, 
                k,
                perturb,
                beta,
                seed,
                save_dir, 
                bs, 
                num_iters, 
                num_queries, 
                num_epochs,
                device):
        '''
        Class responsible for retraining with adverarial examples using the
        sparse_rs framework.
        
        Inputs:
            root        - location of parent directory for the library
            k           - truncation param.
            perturb     - l_0 budget for sparse_rs
            beta        - l_inf norm parameter (scales image domain)
            seed        - used within sparse_rs
            save_dir    - location to output log files and models
            bs          - batch size
            num_iters   - times to repeat the training and attacking cycle
            num_queries - time budget for each attack
            num_epochs  - how long to train during each iteration
            device      - where to train network
            
        Outputs:
            Saves these files in save_path
            net.pth       - model state_dict (saved on cpu)
            results.p     - results as a list of strings (see self.run()) 
            f_results.p   - final acc and r_acc when using the full time budget
            log.txt       - log of all attacks while training
            log_final.txt - log of the final attack (used to make figures/eval)
        '''
        super(adv_trainer, self).__init__()
        # init params
        self.root = root
        self.k = k
        self.perturb = perturb
        self.beta = beta
        self.seed = seed
        self.save_path = root+save_dir 
        self.bs = bs 
        self.device = torch.device(device)
        self.num_queries = num_queries 
        self.num_epochs = num_epochs 
        self.num_iters = num_iters
        self.iter = 0
        # extra params derived
        self.mu, self.sigma = mu_sigma(self.beta)
        self.net_path = self.save_path+'net.pth'
        self.results_str = []
        pickle.dump(self.results_str, open(self.save_path+'results.p','wb'))
        
        # prep the network
        if k == 0:
            self.net = L_Net().to(self.device)
            self.eval_net = L_Net_eval(self.mu,self.sigma).to(self.device)
        else:
            self.net = r_L_Net(self.k).to(self.device)
            self.eval_net = r_L_Net_eval(self.k, self.mu,self.sigma).to(self.device)
        torch.save(self.net.state_dict(), self.net_path)
        # prep the training utils
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001, momentum=0.9)
        # prep the data
        self.Data = prep_MNIST(self.root, bs)
    
    def train(self):
        '''
        Trains self.net with self.criterion using self.optimizer.
        Performs self.num_epochs passes of the data, saving the weights after.
        '''
        self.net.train()
        for _ in trange(self.num_epochs):  
            running_loss = 0.0
            for inputs, labels in zip(self.Data['x_train'],self.Data['y_train']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            torch.save(self.net.state_dict(), self.net_path)
    
    def test(self):
        '''
        Preforms a test on self.net using the MNIST test dataset.
        Returns the clean accuracy
        '''
        self.net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in zip(self.Data['x_test'],self.Data['y_test']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * (correct / total)
        return acc

    def r_test(self, train=False, test=False):
        '''
        Preforms an attack on the data using sparse_rs as the adversary.

        By default will run attack on only one batch of testset and 
        return rob. acc. for mid training statistics.

        Inputs:
            train  - If TRUE, attacks ENTIRE TRAINset, and returns adversarial examples
            test   - If TRUE, attacks ENTIRE TESTset (for longer), only returns rob. acc.
        '''
        # load net 
        state_dict = torch.load(self.net_path, map_location=self.device)
        self.eval_net.load_state_dict(state_dict)
        self.eval_net.eval()
        # setup params depending on input
        keys = ['x_test','y_test']
        batches = 1
        num_queries = self.num_queries
        log_path=self.save_path+'log.txt'
        if train:
            adv_xs = []
            adv_ys = []
            keys =  ['x_og','y_og']
            batches = len(self.Data['x_og'])
        elif test:
            all_acc = []
            batches = len(self.Data['x_test'])   
            num_queries = 5000
            log_path=self.save_path+'log_final.txt'
        # load adversary
        adversary = RSAttack(self.eval_net,
                                norm='L0',
                                eps=self.perturb,
                                n_queries=num_queries,
                                n_restarts=1,
                                seed=self.seed,
                                device=self.device,
                                log_path=log_path
                                )
        # attack over defined bathces (1 or all)
        for i in trange(batches):
            x = (self.Data[keys[0]][i].to(self.device)-self.mu)/self.sigma
            y = self.Data[keys[1]][i].to(self.device)
            with torch.no_grad():
                # find points originally correctly classified
                output = self.eval_net(x)
                pred = (output.max(1)[1] == y).float().to(self.device)
                ind_to_fool = (pred == 1).nonzero(as_tuple=False).squeeze()
                # preform the attack on corresponding indeces and save
                _, adv = adversary.perturb(x[ind_to_fool], y[ind_to_fool])
                # analyze the attack
                output = self.eval_net(adv.to(self.device))
                r_acc = (output.max(1)[1] == y[ind_to_fool]).float().to(self.device)
                adversary.logger.log('robust accuracy {:.2%}'.format(r_acc.float().mean()))
                # save if training
                if train:
                    idx_fooled = (output.max(1)[1] != y[ind_to_fool])
                    adv_xs.append(torch.clone(adv[idx_fooled]))
                    adv_ys.append(torch.clone(y[ind_to_fool][idx_fooled]))
                # eval if testing
                elif test:
                    all_acc.append(r_acc.float().mean()*100)
        if train:
            return adv_xs, adv_ys
        elif test:
            return sum(all_acc[0:-1]).item()/(len(all_acc)-1)
        else:
            return r_acc.float().mean()*100
                
    def attack_save(self):
        '''
        Goes through original train set and attacks with sprase_rs for
        num_queries. Saves the corresponding new examples
        to the trainset in Data and returns how many new examples were created
        '''
        # re initialize the dataset
        self.Data['x_train'] = []
        self.Data['y_train'] = []
        # get adversarial examples
        adv_xs,adv_ys = self.r_test(train=True)
        # now we update the Data (on the cpu since its large)
        rem_x = torch.tensor([]).float().to('cpu')
        rem_y = torch.tensor([]).long().to('cpu')

        for x,y,adv_x,adv_y in zip(self.Data['x_og'],self.Data['y_og'],adv_xs,adv_ys):
            adv_x=(adv_x.to('cpu')*self.sigma)+self.mu
            # concatenate and shuffle, storing the remainder
            new_x, new_y = torch.cat((x.to('cpu'),adv_x)), torch.cat((y.to('cpu'),adv_y.to('cpu')))
            shuffle = torch.randperm(new_x.size()[0])
            new_x, new_y = new_x[shuffle], new_y[shuffle]
            rem_x = torch.cat((rem_x,torch.clone(new_x[self.Data['bs']:-1])))
            rem_y = torch.cat((rem_y,torch.clone(new_y[self.Data['bs']:-1])))
            # Now store the data with proper batch size
            self.Data['x_train'].append(torch.clone(new_x[0:self.Data['bs']]))
            self.Data['y_train'].append(torch.clone(new_y[0:self.Data['bs']]))
        # when done we want to add the remainder as well
        for i in range(rem_x.shape[0]//self.Data['bs']):
            self.Data['x_train'].append(rem_x[self.Data['bs']*i : self.Data['bs']*(i+1)])
            self.Data['y_train'].append(rem_y[self.Data['bs']*i : self.Data['bs']*(i+1)])

        return len(self.Data['y_og']), len(self.Data['y_train'])

    def run(self):
        '''
        Runs the retraining loop for num_iters
        '''
        while self.iter<self.num_iters:
            res_str = "Running iter%d"%(self.iter)
            print(res_str)
            self.results_str.append(res_str)
            self.train()
            
            acc = self.test()
            res_str = "Accuracy on testset: %.3f"%acc
            print(res_str)
            self.results_str.append(res_str)

            r_acc = self.r_test()
            res_str = "Robust Accuracy: %.3f"%r_acc
            print(res_str)
            self.results_str.append(res_str)

            if self.num_iters != 1:
                old, new = self.attack_save()
                res_str = "Went from %d batches to %d batches after attack"%(old, new)
                print(res_str)
                self.results_str.append(res_str)
            self.iter += 1

        pickle.dump(self.results_str, open(self.save_path+'results.p','wb'))
        torch.save(self.net.state_dict(), self.net_path)

        # Now compute final accuracy and save
        acc = self.test()
        r_acc = self.r_test(test=True)
        
        print(acc, r_acc)
        pickle.dump((acc,r_acc),open(self.save_path+'f_results.p','wb'))
        self.net.to('cpu')
        torch.save(self.net.state_dict(), self.net_path)
        print("FINISHED TRAINING AND EVALUATING")

