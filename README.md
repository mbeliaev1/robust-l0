# Truncation as a Defense for Sparse Attacks
This directory is supplementary material for our work published in IEEE-JSAIT 2024:

[Efficient and Robust Classification for Sparse Attacks](google.com) Mark Beliaev, Payam Delgosha, Hamed Hassani, Ramtin Pedarsani.

All relevant citations for methods used are found in the paper's list of references.

This README contains 4 sections:

[I. Requirements](#i.-requirements)

List of the requirements needed to run the code, as well as instructions on how to setup the required environemnt.

[II. Contents](#ii.-contents)

Summary of the sub-directories and files found within this project.

[III. Training from scratch](#iii.-training-from-scratch)

Description on how to use the code provided to train our truncated models from scratch.

[IV. Evaluating results](#iv.-evaluating-results)

Description on how to use the code provided to validate our results by evaluating the provided pre-trained models, **or** evaluate results for newly trained models.

## I. Requirements
We recommend using pacakge manager [pip](https://pip.pypa.io/en/stable/) as well as 
[conda](https://www.anaconda.com/products/individual) to install the relative packages:

**conda:**
- python-3.8.5 [python](https://www.python.org/downloads/release/python-385/)
- numpy-1.19.2 [numpy](https://numpy.org/devdocs/release/1.19.2-notes.html)
- pytorch-1.7.1 [pytorch](https://pytorch.org/)

**pip:**
- foolbox-2.4.0 [foolbox](https://foolbox.readthedocs.io/en/v2.4.0/)
- tqdm [tqdm](https://tqdm.github.io/)

```bash
conda create -n robust python==3.8.5
conda activate robust
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
python torch_test.py
conda install jupyter
pip install foolbox==2.4.0 tqdm
```

## II. Contents

**datasets/**

The MNIST and CIFAR datasets will be downloaded and stored here if they are not already present when one runs **scripts/train.py** or **scripts/train.py** for the first time. 

**new_trained/**

Empty folder structure for storing the results of new adversarially trained networks. Structure of results saved is found in **scripts/train.py**. 

**figures/**

Containes the code used to generate the figures from our experiment.

**scripts/**

Scripts for training and evaluating models. Usage found in sections [III. Training from scratch](#iii.-training-from-scratch) and [IV. Evaluating results](#iv.-evaluating-results).

**utils/**

All required code to perform our experiments, including the LICENSE file for sparse-rs.

(1) **adv_trainer.py**: Contains the general adversarial training class used for our experiments.  

(2) **attack.py**: Contains the general attack class used for our experiments.  

(3) **models.py**: Contains all the models used for our experiments.

(4) **attacks/sparse_rs.py**: Contains the sparse-rs class that is used to attack our networks in **adv.py**. This file is different from the [original version](https://github.com/fra31/sparse-rs/blob/master/rs_attacks.py) as we use the MNIST dataset ontop of CIFAR, and change variables based on which experiment is being performed. 

(5) **helpers.py**: Contains various helper methods used in our experiments 

(6) **trunc.py**: Contains various truncation implementations. Note that our experiments only utilized the "simple" truncation mechanism as we found it to have the best overall perforamnce. 

## III. Training from scratch

We will briefly cover the details of the training and evaluation scripts found in **scripts/**, setting the parameters for epochs, queries, and iterations to arbitrary numbers that generate the results quickly. For full evaluation and training as done in the paper experiments, in most cases you should use the default paramters, but we urge you to check the paper for specific configurations.

To train a samll CNN with truncation parameter k=12 on MNIST and save to **/new_trained/test/**:

```bash
python scripts/train.py --cfg_name cnn_small --trunc_type simple --dataset MNIST --exp test --k 12 --perturb 12 --seed 0 --epochs 2 --queries 10 --iters 2 
```

To remove the truncation parameter and use the default CNN network with adversarial training, simply set k to zero while keeping perturb at the desired magnitude:

```bash
python scripts/train.py --cfg_name cnn_small --dataset MNIST --exp test --k 0 =- --seed 0 --epochs 2 --queries 10 --iters 2 
```

To remove teh adversarial componenet completely you need the no_adv flag:

```bash
python scripts/train.py --cfg_name cnn_small --dataset MNIST --exp test --k 0 --no_adv --seed 0 --epochs 2 --queries 10 --iters 2 
```

## IV. Evaluating results

To evaluate a particular network, use one of the 3 eval scripts **scripts/eval_rs.py**,**scripts/eval_pw.py**, or **scripts/multi_rs.py** and set the corresponding arguements. For example, we can evaluate all 4 networks we just trained, measuring their accuracy, robust accuracy with sparseRS, and median adversarial attack magnitude wit the pointwise attack:

```bash
python scripts/eval_rs.py --eval_dir new_trained/test --budget 12 --queries 500 --restarts 1 
python scripts/eval_pw.py --eval_dir new_trained/test --iters 10  --sampels 100 

```

Note that the experiment configuarions are loaded by loading all the json configurations found in the provided directory given by the --eval_dir arguement. For sparse-rs, when evaluating the arguement perturb controls the l0 magnitude of the attack for testing robust accuracy, whereas in **scripts/train.py** it controlls the magnitude of the attack in the adversarial training component. 

## additonal experiments
```bash
python scripts/train.py --cfg_name cnn_small --trunc_type simple --dataset MNIST --exp final_long_noadv --k 12 --no_adv --seed 0 --epochs 100 --queries 500 
python scripts/train.py --cfg_name cnn_small --trunc_type simple --dataset MNIST --exp final_long_noadv --k 50 --no_adv --seed 0 --epochs 100 --queries 500 
python scripts/train.py --cfg_name VGG16 --trunc_type simple --dataset CIFAR --exp final_long_noadv --k 12 --no_adv --seed 0 --epochs 100 --queries 500 --bs 128 --lr 0.1 
python scripts/train.py --cfg_name VGG16 --trunc_type simple --dataset CIFAR --exp final_long_noadv --k 50 --no_adv --seed 0 --epochs 100 --queries 500 --bs 128 --lr 0.1 

python scripts/eval_rs.py --eval_dir new_trained/final_long_noadv/MNIST --beta 1 --device cuda:3
python scripts/eval_pw.py --eval_dir new_trained/final_long_noadv/MNIST --device cuda:3
python scripts/multi_rs.py --eval_dir new_trained/final_long_noadv/MNIST --beta 1 --device cuda:3 --log_name multi_1.txt
python scripts/multi_rs.py --eval_dir new_trained/final_long_noadv/MNIST --beta 100 --device cuda:3 --log_name multi_100.txt

python scripts/eval_rs.py --eval_dir new_trained/final_long_noadv/CIFAR --beta 1 --device cuda:3
python scripts/eval_pw.py --eval_dir new_trained/final_long_noadv/CIFAR --device cuda:3
python scripts/multi_rs.py --eval_dir new_trained/final_long_noadv/CIFAR --beta 1 --device cuda:3 --log_name multi_1.txt
python scripts/multi_rs.py --eval_dir new_trained/final_long_noadv/CIFAR --beta 100 --device cuda:3 --log_name multi_100.txt
```
