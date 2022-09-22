# Truncation as a Defense for Sparse Attacks
This directory is supplementary material for our work presented at "": 

[Paper Title](google.com) Authors, , .s

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
There are two ways to setup the environment, either manually (1) or using the provided .yml file (2). If you plan on using a GPU, we recommend going with (1) as this will assure you setup the correct cuda environment specific to your machine, while (2) might resort to the cpu. 

### (1) Manual
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

The MNIST and CIFAR datasets will be downloaded and stored here if they are not already present when one runs **train/lin/train.py** or **train/conv/train.py** for the first time. 

Each of the three files requires an additional arguement when executing in the terminal: the path of the parent directory for the network we want to test. Details of how to use these files with the provided pre_trained models to validate our results is given in [IV. Evaluating results](#iv.-evaluating-results).

**new_trained/**

Empty folder structure for storing the results of new adversarially trained networks. Structure of results saved is found in **scripts/train.py**. 

**scripts/**

Scripts for training and evaluating models. Usage found in sections [III. Training from scratch](#iii.-training-from-scratch) and [IV. Evaluating results](#iv.-evaluating-results).

**utils/**

All required code to perform our experiments, including the LICENSE file for sparse-rs.

(1) **adv.py**: Contains the general adversarial training class used for our experiments.  

(2) **models.py**: Contains all the models we used for our experiments.

(3) **sparse_rs.py**: Contains the sparse-rs class that is used to attack our networks in **adv.py**. This file is different from the original version as we use the MNIST dataset ontop of CIFAR, and change variables based on which experiment is being performed. The license corresponding to the originally used sparse-rs code is provided in the **utils/** folder.

(4) **helpers.py**: Contains helper functions utilized throughout our code. Specifically this file contains the custom truncation module used for FC networks, and the truncation function used for convolution networks.

## III. Training from scratch

We will briefly cover the details of the training and evaluation scripts found in **scripts/**, setting the parameters for epochs, queries, and iterations to arbitrary numbers that generate the results quickly. For full evaluation and training as done in the paper experiments, in most cases you should use the default paramters, but we urge you to check the paper for specific configurations.

To train the FC network with truncation parameter k=10 and save to **/new_trained/test/**:

```console
python scripts/train.py --arch fc --k 10 --perturb 10 --seed 99 --epochs 2 --queries 10 --iters 2 
```

Each training epoch takes 29-30 it/s on an RTX 3080. Note that the computational bottleneck here is the adversarial component, hence we set number of queries (t) to 10. A corresponding directory will be created at **/new_trained/test/fc_k10_p10_seed99** with the model and sparse rs log. 

To remove the truncation parameter and use the default FC network, simply set k to zero:

```console
python scripts/train.py --arch fc --k 0 --perturb 10 --seed 99 --epochs 2 --queries 10 --iters 2 
```

For the convolution networks with truncation components attached to the VGG architecture:

```console
python scripts/train.py --arch cnn --k 10 --perturb 10 --seed 99 --epochs 2 --queries 10 --iters 2
```

This trains using the setup from Table 2, letting k=10 and t=300. Each training epoch takes 32-33 it/s on an RTX 3080. A corresponding directory will be created at **/new_trained/test/cnn_k10_p10_seed99** with the model and sparse rs log.

Once again, to remove the truncation parameter, simply set k to zero:

```console
python scripts/train.py --arch cnn --k 0 --perturb 10 --seed 99 --epochs 2 --queries 10 --iters 2
```

## IV. Evaluating results

To evaluate a particular network, use the script **scripts/eval.py** and set the corresponding arguements. For example, we can evaluate all 4 networks we just trained, measuring their accuracy, robust accuracy with sparseRS, and median adversarial attack magnitude wit the pointwise attack:

```console
python scripts/eval.py --mode all --exp test --name fc_k10_p10_seed99 --perturb 10 --queries 10
python scripts/eval.py --mode all --exp test --name fc_k0_p10_seed99 --perturb 10 --queries 10
python scripts/eval.py --mode all --exp test --name cnn_k10_p10_seed99 --perturb 10 --queries 10
python scripts/eval.py --mode all --exp test --name cnn_k0_p10_seed99 --perturb 10 --queries 10
```

Note that the experiment configuarion is loaded using the --name arguement using the corresponding json file in that directory. This time, the arguement perturb controls the l0 magnitude of the attack for testing robust accuracy, whereas in **scripts/train.py** it controlled the magnitude of the attack in the adversarial training component. Note that using --all performs all three tests and saves them as **acc.p**, **rs.p** and **pw.p** within the expirement directory.
