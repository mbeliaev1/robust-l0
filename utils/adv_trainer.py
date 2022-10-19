# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
# other modules
import numpy as np
from tqdm import trange
import pickle
# internal imports
from utils.sparse_rs import RSAttack
from utils.models import *
from utils.helpers import *

class adv_trainer():
    def __init__(self,
                root, 
                arch,
                k,
                perturb,
                seed,
                save_dir, 
                bs, 
                num_iters, 
                num_queries, 
                num_epochs,
                no_adv,
                lr,
                momentum,
                decay,
                embedding,
                dataset,
                device):
        '''
        Class responsible for retraining with adverarial examples using the
        sparse_rs framework.
        
        Inputs:
            root        - location of parent directory for the library
            arch        - string determining the architecture to use for the model
            k           - truncation param.
            perturb     - l_0 budget for sparse_rs
            seed        - used within sparse_rs
            save_dir    - location to output log files and models
            bs          - batch size
            num_iters   - times to repeat the training and attacking cycle
            num_queries - time budget for each attack
            num_epochs  - how long to train during each iteration
            no_adv      - True if you want skip the adversarial training component
            lr          - inital learning rate
            momentum    - inital momentum value
            decay       - decay for lr and momentum (decay=1 means no decay)
            embedding   - dimensionality of fc layer in classifier
            dataset     - dataset name
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
        self.arch = arch
        self.k = k
        self.perturb = perturb
        self.seed = seed
        self.save_path = root+save_dir 
        self.bs = bs 
        self.device = torch.device(device)
        self.num_queries = num_queries 
        self.num_epochs = num_epochs 
        self.num_iters = num_iters
        self.no_adv = no_adv
        self.lr = lr
        self.momentum = momentum
        self.decay = decay 
        self.embedding = embedding
        self.dataset = dataset

        # prep the network based on k and arch
        self.Data = prep_data(root, bs, dataset)
        self.net = Net(cfg_name = arch, 
                       k = k, 
                       embedding = embedding, 
                       input_shape= self.Data['x_train'][0].shape).to(self.device)

        self.lrs = [lr/pow(1/decay,i) for i in range(self.num_iters)]
        self.momentums = [momentum/pow(1/decay,i) for i in range(self.num_iters)]
        
        # dump empty string
        self.results_str = []
        pickle.dump(self.results_str, open(self.save_path+'results.p','wb'))

        self.net_path = self.save_path+'net.pth'
        torch.save(self.net.state_dict(), self.net_path)

        self.criterion = nn.CrossEntropyLoss()
        self.iter = 0
        
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
        for _ in range(self.num_epochs):
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

    def r_test(self, train=False, test=False):
        '''
        Preforms an attack on the data using sparse_rs as the adversary.

        By default will run attack on only one batch of testset and 
        return rob. acc. for mid training statistics.

        Inputs:
            train  - If TRUE, attacks ENTIRE TRAINset for num_queries, and returns adversarial examples
            test   - If TRUE, attacks ENTIRE TESTset (for longer 5000 queries), only returns rob. acc.

            if both train and test are false (default), runs short test on testset determined by 
            num_queries
        '''
        # load net 
        pass
                
    def attack_save(self):
        '''
        Goes through original train set and attacks with sprase_rs for
        num_queries. Saves the corresponding new examples
        to the trainset in Data and returns how many new examples were created
        '''
        pass

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

            # # robust test using perturb
            # if self.perturb > 0:
            #     r_acc = self.r_test()
            # repeat 
            self.iter += 1

        pickle.dump(self.results_str, open(self.save_path+'results.p','wb'))
        torch.save(self.net.state_dict(), self.net_path)

        # Now compute final accuracy and save
        acc = self.test()
        r_acc = 0

        print(acc, r_acc)
        pickle.dump((acc,r_acc),open(self.save_path+'f_results.p','wb'))
        self.net.to('cpu')
        torch.save(self.net.state_dict(), self.net_path)
        print("FINISHED TRAINING AND EVALUATING")

