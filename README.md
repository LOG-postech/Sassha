# SASSHA: SHARPNESS-AWARE ADAPTIVE SECOND-ORDER OPTIMIZATION WITH STABLE HESSIAN APPROXIMATION

This repository contains Pytorch source code for arXiv paper [SASSHA: SHARPNESS-AWARE ADAPTIVE SECOND-ORDER OPTIMIZATION WITH STABLE HESSIAN APPROXIMATION](https://arxiv.org/abs/.....) by Dahun Shin<sup>&ast;</sup>, [Dongyeop Lee](https://edong6768.github.io/)<sup>&ast;</sup>, Jinseok Chung, and [Namhoon Lee](https://namhoonlee.github.io/).

## Introduction

Sassha is a novel second-order method designed to enhance generalization by explicitly reducing sharpness of the solution, while stabilizing the computation of approximate Hessians along the optimization trajectory.

This Pyotorch implementation supports various tasks, including image_classification, finetuning, and label noise experiments.

For a detailed explanation of the Sassha algorithm, please refer to [this paper](https://arxiv.org/pdf/2006.00719.pdf)


## Getting started

First clone the Sassha repository to your local system:
```
git clone https://github.com/LOG-postech/Sassha.git
```

We recommend using Anaconda to set up the environment and install all necessary dependencies:

```
conda env create -f sassha.yaml
```

Ensure you are using Python 3.9 or later.

Next, activate the newly created Conda environment:

```
conda activate sassha
```

Navigate to the example folder of your choice. For instance, to run an image classification experiment:

```
cd image_classification
```

Now, train the model with the following command:
```
python train.py --workers 4 --dataset imagenet -a resnet50 --epochs 90 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 30 60 --lr-decay 0.1 \
--optimizer sassha \
--lr 0.3 --wd 1e-4 --rho 0.2 --lazy_hessian 10 --seed 0 \
--project_name sassha \
imagenet-folder with train and val folders
```

### Distributed Training (Single node, multiple GPUs)
Sassha is fully compatible with multi-GPU environments for distributed training. Use the following command to train a model across multiple GPUs on a single node:
```
python train.py --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
--workers 4 --dataset imagenet -a vit_b_32 --epochs 90 -b 1024 \
--LRScheduler cosine --warmup_epochs 8 \
--optimizer sassha \
--lr 0.6 --wd 2e-4 --rho 0.25 --lazy_hessian 10 --eps 1e-6 --seed 0 \
--project_name sassha \
imagenet-folder with train and val folders
```
Ensure that NCCL is properly configured on your system and that your GPUs are available before running the script.

### Reproducing Paper Results
Note that you can reproduce the results reported in the paper by using the provided script arguments available in the configuration file of each example folder.

### Environments
- cuda 11.6.2
- python 3.10

## General Usage

SASSHA can be imported and used as follows:

```python
from optimizers import SASSHA

...

# Initialize your model and optimizer
model = YourModel()
optimizer = SASSHA(model.parameters(), ...)

...

# training loop
for input, output in data:

  # first forward-backward pass
  loss = loss_function(output, model(input))
  loss.backward()
  optimizer.perturb_weights(zero_grad=True)
  
  # second forward-backward pass
  loss_function(output, model(input)).backward(create_graph=True)  
  optimizer.unperturb()
  optimizer.step()
  optimizer.zero_grad()

...
```
## Citation
```
@article{shin2025sassha,
  title={SASSHA: SHARPNESS-AWARE ADAPTIVE SECOND-ORDER OPTIMIZATION WITH STABLE HESSIAN APPROXIMATION},
  author={Shin, Dahun and Lee, Dongyeop and Chung, Jinseok and Lee, Namhoon},
  journal={arXiv preprint arXiv:2502.18153},
  year={2025}
}
```
