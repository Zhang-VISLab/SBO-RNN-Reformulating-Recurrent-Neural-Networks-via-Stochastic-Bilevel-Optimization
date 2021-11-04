# SBO-RNN-Reformulating-Recurrent-Neural-Networks-via-Stochastic-Bilevel-Optimization

## Overview
Code for paper SBO-RNN: Reformulating Recurrent Neural Network via Stochastic Bilevel Optimization.

In this paper, we proposed a family of recurrent neural networks(RNNs), namely SBO-RNN, to improve the training stability of RNNs. With the help of Stochastic gradient descent(SGD), we managed to convert the stochasitic bilevel optimization(SBO) problem into an RNN where the feedforward and backpropagation solve the lower and upper-level optimization for learning hidden states and their hyperparameters, respectively.Empirically we demonstrate our approach with superior performance on several benchmark datasets, with fewer parameters, less training data, and much faster convergence.

![Illustration of SBO-RNN architectures using the optimizers](links)



## Installation
1. Clone repo
```
git clone --recursive https://github.com/Zhang-VISLab/SBO-RNN-Reformulating-Recurrent-Neural-Networks-via-Stochastic-Bilevel-Optimization.git
```

2. Working Environment
```
conda create -n sbornn pytorch
conda activate sbornn
pip install -r requirements.txt
```

3. Setting
```
# Checking the settings <params> in the main function before runing the code
# define the path according your need and the result will be saved under <path>
```

4. Training
```
cd src
CUDA_VISIBLE_DEVICES=0 python train.py --cuda
```

5. Results
```
cd <path>
# dfhistory.cvs contains all training and test log
# *.bin files are model parameters
```

## Visualize the test results

