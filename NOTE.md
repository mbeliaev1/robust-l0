## To Do

1. Make sure that running CIFAR/MNIST loads the appropriate paramters 

use paper as well as the code below:
```python
k = 0
perturb = 12
seed = 99
save_dir = '/new_trained/test/'
bs = 128 # 128 for cnn models
num_iters = 10
num_queries = 300
num_epochs = 25
no_adv = True
dataset = 'CIFAR'
device = 'cuda:2'
embedding = 512 #512 embedding for cnn arch
momentum = 0.9
decay = 0.5
arch = 'VGG19'
lr = 0.2
```

2. perturb and k need to be working properly for trainers

3. eval scripts