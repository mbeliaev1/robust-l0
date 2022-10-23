# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
# other modules
import numpy as np
from tqdm import trange
import pickle
# internal imports

from utils.models import Net
from utils.helpers import prep_data, evaluate
from utils.attack import attack

class adv_trainer():
    def __init__(self,
                root, 
                cfg_name,
                trunc_type,
                dataset,
                k,
                perturb,
                bs, 
                num_iters, 
                num_epochs,
                num_queries, 
                no_adv,
                lr,
                momentum,
                decay,
                embedding,
                seed,
                save_dir, 
                device):
        '''
        Class responsible for retraining with adverarial examples using the
        sparse_rs framework.
        
        Inputs:
            root        - location of parent directory for the library
            cfg_name    - string determining the architecture to use for the model
            trunc_type  - string determining the truncation method used
            dataset     - dataset name
            k           - truncation param (used as defense)
            perturb     - l_0 budget for sparse_rs (used as attack)
            bs          - batch size
            num_iters   - times to repeat the training and attacking cycle
            num_epochs  - how long to train during each iteration
            num_queries - time budget for each attack
            no_adv      - True if you want skip the adversarial training component
            lr          - inital learning rate
            momentum    - inital momentum value
            decay       - decay for lr and momentum (decay=1 means no decay)
            embedding   - dimensionality of fc layer in classifier
            seed        - used within sparse_rs
            save_dir    - directory relative to root 
            device      - where to train network (all networks saved on cpu)
            
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
        self.perturb = perturb
        self.seed = seed
        self.save_path = root+save_dir 
        # self.bs = bs 
        self.device = torch.device(device)
        self.num_queries = num_queries 
        self.num_epochs = num_epochs 
        self.num_iters = num_iters
        self.no_adv = no_adv

        # prep the network based on k and cfg_name
        self.Data = prep_data(root, bs, dataset)
        self.net = Net(cfg_name = cfg_name, 
                       k = k, 
                       embedding = embedding, 
                       input_shape= self.Data['x_train'][0].shape,
                       trunc_type=trunc_type).to(self.device)

        self.lrs = [lr/pow(1/decay,i) for i in range(self.num_iters)]
        self.momentums = [momentum/pow(1/decay,i) for i in range(self.num_iters)]
        
        # dump empty string
        self.results_str = []
        pickle.dump(self.results_str, open(self.save_path+'results.p','wb'))

        self.net_path = self.save_path+'net.pth'
        torch.save(self.net.state_dict(), self.net_path)

        self.criterion = nn.CrossEntropyLoss()
        self.iter = 0

        # check perturbation budget
        
    def train(self):
        '''
        Trains self.net with self.criterion using self.optimizer.
        Performs self.num_epochs passes of the data, saving the weights after.
        '''
        optimizer = optim.SGD(self.net.parameters(), 
                              self.lrs[self.iter], 
                              momentum=self.momentums[self.iter],
                              weight_decay=5e-4)
        self.net.train()
        # for _ in trange(self.num_epochs):  
        for _ in trange(self.num_epochs):
            running_loss = 0.0
            for inputs, labels in zip(self.Data['x_train'],self.Data['y_train']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # torch.cuda.empty_cache()
            torch.save(self.net.state_dict(), self.net_path)
    
    def test(self, train=False):
        '''
        Preforms a test on self.net using the MNIST test dataset.
        Returns the clean accuracy
        '''
        if train:
            data_string = 'train'
        else:
            data_string = 'test'

        acc = evaluate(net = self.net, 
                       x = self.Data['x_%s'%data_string],
                       y = self.Data['y_%s'%data_string],
                       device = self.device)
        return acc

    def r_test(self, test=True):
        '''
        Preforms an attack on the data using sparse_rs as the adversary.

        By default will run attack on only one batch of testset and 
        return rob. acc. for mid training statistics.

        Inputs:
            test   - If TRUE, attacks ENTIRE TESTSET (for longer 5000 queries), only returns rob. acc.
                     If FALSE, attacks ENTIRE TRAINSET for n_queries and returns examples generated

            if both train and test are false (default), runs short test on testset determined by 
            num_queries
        '''
        # from paper we can use n_queries=10000 and n_restarts=5
        if test:
            n_queries = 5000
            log_path = self.save_path+'rs_log_final.txt'
            data_string = 'test'
        # if not test, we use original dataset for adversarial training
        else:
            n_queries = self.num_queries
            log_path = self.save_path+'rs_log.txt'
            data_string = 'og'

        r_acc, x_advs, y_advs, i_advs = attack(net= self.net,
                                            budget = self.perturb,
                                            x = self.Data['x_%s'%data_string],
                                            y = self.Data['y_%s'%data_string],
                                            n_queries=n_queries,
                                            n_restarts=1,
                                            device=self.device,
                                            log_path=log_path
                                            )

        return r_acc, x_advs, y_advs, i_advs
                
    def attack_save(self):
        '''
        Goes through original train set and attacks with sprase_rs for
        num_queries. Saves the corresponding new examples
        to the trainset in Data and returns how many new examples were created
        '''

        # generate adversarial examples
        _, x_advs, y_advs, _ = self.r_test(test=False)
        # re initialize the dataset
        self.Data['x_train'] = []
        self.Data['y_train'] = []
        # now we update the Data (on the cpu since its large)
        rem_x = torch.tensor([]).float().to('cpu')
        rem_y = torch.tensor([]).long().to('cpu')

        for x,y,x_adv,y_adv in zip(self.Data['x_og'],self.Data['y_og'],x_advs,y_advs):
            # make sure everything is on the same device
            x_adv = x_adv.to('cpu')
            y_adv = y_adv.to('cpu')
            x = x.to('cpu')
            y = y.to('cpu')
            # concatenate and shuffle
            new_x, new_y = torch.cat((x,x_adv)), torch.cat((y,y_adv))
            shuffle = torch.randperm(new_x.size()[0])
            new_x, new_y = new_x[shuffle], new_y[shuffle]
            # store what doesnt fit into one batch as remainder
            rem_x = torch.cat((rem_x,torch.clone(new_x[self.Data['bs']:-1])))
            rem_y = torch.cat((rem_y,torch.clone(new_y[self.Data['bs']:-1])))
            # Now store the data with proper batch size
            self.Data['x_train'].append(torch.clone(new_x[0:self.Data['bs']]))
            self.Data['y_train'].append(torch.clone(new_y[0:self.Data['bs']]))
        # when done we want to add the remainder 
        for i in range(rem_x.shape[0]//self.Data['bs']):
            self.Data['x_train'].append(rem_x[self.Data['bs']*i : self.Data['bs']*(i+1)])
            self.Data['y_train'].append(rem_y[self.Data['bs']*i : self.Data['bs']*(i+1)])

        # return stats regarding new dataset size
        return len(self.Data['y_og']), len(self.Data['y_train'])

    def run(self):
        '''
        Runs the retraining loop for num_iters
        '''
        while self.iter<self.num_iters:
            # train 
            res_str = "Running iter %d"%(self.iter)
            print(res_str)
            self.results_str.append(res_str)
            self.train()

            # test on testset
            acc = self.test()
            res_str = "Accuracy on testset: %.3f"%acc
            print(res_str)
            self.results_str.append(res_str)

            # test on trainset
            acc = self.test(train=True)
            res_str = "Accuracy on trainset: %.3f"%acc
            print(res_str)
            self.results_str.append(res_str)

            #add adversarial examples 
            if (not self.no_adv) and (self.iter<self.num_iters-1):
                old, new = self.attack_save()
                res_str = "Went from %d batches to %d batches after attack"%(old, new)
                print(res_str)
                self.results_str.append(res_str)
            
            self.iter += 1

        # Now compute final accuracy + rob accuracy
        self.net.eval()
        acc = self.test()

        assert self.perturb>0
        r_acc,_,_,_ = self.r_test()
        res_str = "Final testset accuracy: %.3f"%acc
        print(res_str)
        self.results_str.append(res_str)
        res_str = "Robust accuracy: %.3f"%r_acc
        print(res_str)
        self.results_str.append(res_str)

        # save everything
        pickle.dump((acc,r_acc),open(self.save_path+'f_results.p','wb'))
        self.net.to('cpu')
        pickle.dump(self.results_str, open(self.save_path+'results.p','wb'))
        torch.save(self.net.state_dict(), self.net_path)
        print("FINISHED TRAINING AND EVALUATING")
