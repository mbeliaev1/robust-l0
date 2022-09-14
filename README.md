# Truncation as a Defense for Sparse Attacks
This directory is supplementary material for our work presented at 2022 IEEE International Symposium on Information Theory: 

[Efficient and Robust Classification for Sparse Attacks](https://arxiv.org/abs/2201.09369) Mark Beliaev, Payam Delgosha, Hamed Hassani, Ramtin Pedarsani.

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

### (2) Using .yml file
Download the [anaconda](https://www.anaconda.com/products/individual) distribution package and run:
```console
conda env create -f environment.yml
```

Following this, activate the environemtn with:
```console
conda activate trunc_min
```

## II. Contents

**datasets/**

The MNIST and CIFAR datasets will be downloaded and stored here if they are not already present when one runs **train/lin/train.py** or **train/conv/train.py** for the first time. 

**eval/**

Scripts for evaluating the results of the pretrained/new models. Within the folder there are subdirectories for the experiments regarding FC networks **eval/lin/** and convolution networks **eval/conv/**. Each subdirectory has the same three files: 

(1) **acc.py**: Tests the clean accuracy of the specified network. 

(2) **rs.py**: Tests the robust accuracy using sparse-rs.

(3) **pointwise.py**: Tests median adversarial attack magnitude using the pointwise attack for both beta=100 and beta=1.

Each of the three files requires an additional arguement when executing in the terminal: the path of the parent directory for the network we want to test. Details of how to use these files with the provided pre_trained models to validate our results is given in [IV. Evaluating results](#iv.-evaluating-results).

**new_trained/**

Empty folder structure for storing the results of new adversarially trained networks. Contains seperate directories for linear and convolution models.

**pre_trained/**

Same structure as **new_trained/**, but already contains the results for the networks used in our experiments. For details on the contents of the results, see the corresponding **adv.py** files found in **utils/conv/** **utils/lin/**. Due to the restriction in space, we did not add the model weights for the convolution networks, although the code to retrain them is provided, and the results are still there. We plan to upload these weights publically after the paper has been reviewed.

**train/**

Scripts for training models from scratch, as of now they are set up to replicate the models from our experiments.

**utils/**

All required code to perform our experiments, including the LICENSE file for sparse-rs. Within the folder there are subdirectories for the experiments regarding FC networks **utils/lin/** and convolution networks **utils/conv/**. Each subdirectory has the same three files: 

(1) **adv.py**: Contains the general adversarial training class used for our experiments.  

(2) **models.py**: Contains all the models we used for our experiments.

(3) **sparse_rs.py**: Contains the sparse-rs class that is used to attack our networks in **adv.py**. This file is specific for convolution and linear models because our MNIST image data does not have channels like CIFAR and hence the original sprase-rs code was slightly edited (see line 325 for comments regarding this). The license corresponding to the originally used sparse-rs code is provided in the **utils/** folder.

In addition to this, the **helpers.py** file contains helper functions utilized throughout our code. Specifically this file contains the custom truncation module used for FC networks, and the truncation function used for convolution networks.
## III. Training from scratch
### K-truncated Fully Connected Networks
To retrain the network and save to **/new_trained/lin/test/** run:
```console
python train/lin/train.py 
```

This trains using the setup from Table 1, letting k=10 and t=300. Each training epoch takes 29-30 it/s on an RTX 3080. You can change the settings accordingly in the given file as documentation is provided in the **/utils/lin/** folder and helpers folder.

To compare with the regular FC network with k=0 change 'k=10' to 'k=0' on **line 17** of **/train/lin/train.py**. Each training epoch takes 0.5 it/s on an RTX 3080. This will result in an arbitrary robust accuracy of 0%, as the attack is able to fool the classifier on the entire testset regardless of adversarial training. 

### K-truncated Convolution Networks (VGG-19)
For the convolution networks, the results will be saved to  **/new_trained/conv/test/**. Perform the experiments by running:
```console
python train/conv/train.py 
```
This trains using the setup from Table 2, letting k=10 and t=300. Each training epoch takes 32-33 it/s on an RTX 3080. You can change the settings accordingly in the given file as basic documentation and comments are provided in the **/utils/conv/** folder and the **/utils/helpers.py** file.

To compare with the regular FC network with k=0 change 'k=10' to 'k=0' on **line 17** of **/train/conv/train.py**. Each training epoch takes 13-14 it/s on an RTX 3080. 

## IV. Evaluating results
We have provided pretrained models corresponding to the main results from Tables 1-3. The FC nets are found in **/pre_trained/lin/** where **og** and **rob** are the parent directory names for $F^{(0)}$ and $F^{(10)}$ respectively. You can read about the contents of these folders from the adversarial training class found in **/utils/lin/adv.py**. Here we will show how to validate the results for sparse-rs and the pointwise attack using just the model weights of the final networks stored in **net.pth**. 

The same can be done for the convolution models $VGG^{(0)}$ and $VGG^{(10)}$ attacks by simply replacing **lin** with **conv** for all the following instructions. Note though that we left the model weights out as each file is roughly 80MB, making it hard to fit within the 100MB limit of the supplementary material. We leave the instructions here as we plan to release the weights publically once the anonymous review process is finished. 

### Clean accuracy
To test the clean accuracy of the non truncated FC network $F^{(0)}$ run:
```console
python eval/lin/acc.py pre_trained/lin/og/
```
for $F^{(10)}$ you just change the path accordingly:
```console
python eval/lin/acc.py pre_trained/lin/rob/
```
and for the conv models, you change the folder path AND the file path:
```console
python eval/conv/acc.py pre_trained/conv/og/
```

We keep this sort of structure for the other tests as well, and hence will mostly show how to perform them on $F^{(0)}$. 

### Robust accuracy (sparse-rs)
To run the sparse-rs attack on $F^{(0)}$ run:
```console
python eval/lin/rs.py pre_trained/lin/og/ [a] [b] [c] [d] 
```
[a] - number of queries, or time budget of the attack (default: 300)  

[b] - beta value (default: 100)

[c] - l_0 budget (default: 10)

[d] - how many batches of MNIST to use (MAX: 39) (default: 39)


Hence if one wanted to check the results with a longer attack of **time budget 10,000**, **beta value of 10**, **l_0 budget of 5** but only on **one batch** of 256 images from the MNIST test set they would run:
```console
python eval/lin/rs.py pre_trained/lin/og/ 10000 10 5 1
```
The final accuracy will be pritned, and a temporary log file **log_temp.txt** will be created in **pre_trained/lin/og/** describing the attack. 
If then one wanted to confirm this on the k-truncated network $F^{(10)}$, the command would be:
```console
python eval/lin/rs.py pre_trained/lin/rob/ 10000 10 5 1
```

Note all the options can be left blank and the default reults would be calculated.

### Median adversary attack magnitude (Pointwise Attack)
To run the Pointwise Attack on $F^{(0)}$ for BOTH beta=100 and beta=1 as in Table 2 run:
```console
python eval/lin/pointwise.py pre_trained/lin/og/ [a] [b] [c] 
```
[a] - batch size (default: 64)

[b] - num batches to calculate on (default: 16)

[c] - number of iterations to re run the attack for (default: 10)

The output would first show the result for Beta=1, where the adversarial images generated are only allowed to be within the original domain, and then for Beta=100. 

One can do as before by chaging the corresponding file and directory paths to check the results for the VGG networks, as well as the robust FC network.

For example, to perform a quick attack on **1 batch** of **4 images** using **10 iterations** on the robust convolution network $VGG^{(10)}$
```console
python eval/conv/pointwise.py pre_trained/conv/rob/ 4 1 10
```
# trunc
