# About this repo

This repo includes the training and evaluation scripts of the paper : **Neural Network Compression Using Higher-Order Statistics and Auxiliary Reconstruction Losses** (to appear in CVPR workshop: Joint Workshop on Efficient Deep Learning in Computer Vision). 

# Installation

..*.The code has been testing with the following setup:

1. Pytorch 1.3
2. Python 3.5
3. CUDA 10.2 

# Model Architecture

![model architecture image](https://github.com/chatzikon/DNN-COMPRESSION/blob/master/asset/method.png)



The cifar folder contains the files required to make the experiments using the cifar10 and cifar100 dataset and the folder ilscvr the files required to make the experiments using the ILSCVR-12 dataset. There are pretrained models at the respective folder. 

#**CIFAR**: 

*Step1*: In order to prune using P1,P2,P3 pruning strategy, use *_prune_normality files, *_global files are for P4 and *prune are for P5. The Tvalue calculation is employed to extract Tmin and Tmax of each network (eq 9)

*Step2*: The *compression_conv_fc* is used to apply the HOOI to models and the load utils to choose the desired network to decompose

*Step3*: The main_finedune_mse_loss function is used to fine-tune a model using only the final MSE loss and the main_finedune_mult_losses function to fine-tune using multiple auxiliary MSE losses.

*Step4*: Further fine-tune without auxiliary MSE losses using either main_finetune function for a pruned model (compression approach A1) or main_finetune_model_decomposed for a decomposed model (A2) or a pruned+decomposed model (A3).

**ILSCVR**:

*Step1*: P1 strategy is employed for pruning, using the *_normality_imagenet file.

*Step2*: The *compression_conv_fc* is used to apply the HOOI to models and the load utils to choose the desired network to decompose

*Step3*: The main_finedune_mse_loss function is used to fine-tune a model using only the final MSE loss (due to resource constraints,  multiple auxiliary MSE losses were not used during the ILSCVR fine-tuning

*Step4*: Further fine-tune without auxiliary MSE losses using either main_finetune function for a pruned model (compression approach A1) or main_finetune_model_decomposed for a decomposed model (A2) or a pruned+decomposed model (A3).

