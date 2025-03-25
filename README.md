# SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation

This repository contains Pytorch source code for arXiv paper [SASSHA: Sharpness-aware Adaptive Second-order Optimization With Stable Hessian Approximation](https://arxiv.org/abs/2502.18153) by Dahun Shin<sup>&ast;</sup>, [Dongyeop Lee](https://edong6768.github.io/)<sup>&ast;</sup>, Jinseok Chung, and [Namhoon Lee](https://namhoonlee.github.io/).

## Introduction

SASSHA is a novel second-order method designed to enhance generalization by explicitly reducing sharpness of the solution, while stabilizing the computation of approximate Hessians along the optimization trajectory.

This Pytorch implementation supports various tasks, including image classification, finetuning, and label noise experiments.

For a detailed explanation of the SASSHA algorithm, please refer to [our paper](https://arxiv.org/pdf/2502.18153).


## Getting Started

First, clone our repository to your local system:
```bash
git clone https://github.com/LOG-postech/Sassha.git
cd Sassha
```

We recommend using Anaconda to set up the environment and install all necessary dependencies:

```bash
conda create -n "sassha" python=3.9
conda activate sassha
pip install -r requirements.txt
```

Ensure you are using Python 3.9 or later.

Navigate to the example folder of your choice. For instance, to run an image classification experiment:

```bash
cd image_classification
```

Now, train the model with the following command:
```bash
python train.py --workers 4 --dataset imagenet -a resnet50 --epochs 90 -b 256 \
--LRScheduler multi_step --lr-decay-epoch 30 60 --lr-decay 0.1 \
--optimizer sassha \
--lr 0.3 --wd 1e-4 --rho 0.2 --lazy_hessian 10 --seed 0 \
--project_name sassha \
{enter/your/imagenet-folder/with/train_and_val_data}
```

Here, enter the path to imagenet datasets in `{enter/your/imagenet-folder/with/train_and_val_data}`.

### Distributed Training (Single node, multiple GPUs)
SASSHA is fully compatible with multi-GPU environments for distributed training. Use the following command to train a model across multiple GPUs on a single node:
```bash
python train.py --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 \
--workers 4 --dataset imagenet -a vit_b_32 --epochs 90 -b 1024 \
--LRScheduler cosine --warmup_epochs 8 \
--optimizer sassha \
--lr 0.6 --wd 2e-4 --rho 0.25 --lazy_hessian 10 --eps 1e-6 --seed 0 \
--project_name sassha \
{enter/your/imagenet-folder/with/train_and_val_data}
```
Ensure that NCCL is properly configured on your system and that your GPUs are available before running the script.

### Reproducing Paper Results
Configurations used in [our paper](https://arxiv.org/pdf/2502.18153) are provided as shell scrips in each example folder.

### Environments
- cuda 11.6.2
- python 3.9

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
```bibtex
@article{shin2025sassha,
  title={SASSHA: Sharpness-aware Adaptive Second-order Optimization With Stable Hessian Approximation},
  author={Shin, Dahun and Lee, Dongyeop and Chung, Jinseok and Lee, Namhoon},
  journal={arXiv preprint arXiv:2502.18153},
  year={2025}
}
```