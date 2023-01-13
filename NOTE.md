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

Shell script Descriptions
All shells set up for MNIST test

------cuda:0-----------------------------

shell_1: **DONE** adv training with k=5

shel_6: **DONE** no adv cnn training with k=5

shel_7: adv cnn training with k=5

------cuda:1----------------------------- 

shell_3: **DONE** adv training with k=12

shel_8: **DONE** no adv cnn training with k=12

shel_9: **DONE** adv cnn training with k=12

## Results 

FC Results (Robust Accuracy)

**OLD**

| l-0 budget | OG    | lin-ind | lin-dep | lin-dep abs | clip  | clip abs | 
| -----------| --    | ------- | ------- | ----------- | ----  | -------- |
| 5          | 13.30 | 14.88   | 17.28   | 16.63       | 44.19 | **45.14**    |
| 5 + adv    | 45.74 | 46.89   | 43.98   | 49.07       | **59.64** | 57.39    |
| 12         | 0.03  | 0.07    | 0.05    | 0.10        | **4.80**  | 0.10     |
| 12 + adv   | 11.92 | 12.27   | 10.98   | 13.89       | 28.80 | **34.88**    |

Clean accuracy of CNN 98.5 for all

**OLD**

| l-0 budget | OG    | clip | conv |
| -----------| --    | ---- | ----- |
| 5          | 16.17 | 43.70 | **57.92** |
| 5 + adv    | 78.09 | **88.21** | 83.90 |
| 12         | 0.72  | 3.25  | **8.16**  |
| 12 + adv   | 22.27 | **42.61** | 22.30 |

MNIST RESULTS REDONE WITH CORRECT RANGE

FC Results  (all clean accuracies roughly 91%-93%)

| l-0 budget | OG    | lin-ind | lin-dep | clip     | 
| -----------| --    | ------- | ------- | ----     |
| 12         | 0.01  | 0.01    | 0.00    | **1.24** |
| 12 + adv   | 11.46 | 10.31   | 9.23    | **19.74**|

CNN Results (all clean accuracies roughly 98.5%)

| l-0 budget | OG    | clip      | conv      |
| -----------| --    | ----      | -----     |
| 12         | 0.02  | 3.28      | 3.07      |
| 12 + adv   | 6.09  | **26.93** | 11.76     |

CNN + MNIST Results on CIFAR with new network (simple is weighted clip)

MNIST

All clean accuracies roughly 98%

| l-0 budget | OG    | clip      | simple    |
| -----------| --    | ----      | -----     |
| 12         | 0.14  | 36.28     | 0.02      |
| 12 + adv   | 3.35  | **55.78** | 4.43      |

CIFAR
Clean accuracies slightly different (OG: 87%, clip: 86/77%, simple: 82/81% )
| l-0 budget | OG    | clip      | simple    |
| -----------| --    | ----      | -----     |
| 12         | 0.19  | 12.54     | **17.13** |
| 12 + adv   | 3.24  | 15.14     | **27.43** |

simple does better on CIFAR, while worse on MNIST...


- lets look at distribution of images for CIFAR vs MNIST
- run CIFAR with ch by ch method (maybe even better than simple).
- run solid comparison between the methods...

The above is taken care of in note.ipynb

eval.ipynb checks effect of increasing k for already trained networks

there is no real difference 
**TO DO** 

Implement baselines to compare with