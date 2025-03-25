from __future__ import print_function
import logging
import os
import sys
import random

import numpy as np
import argparse
from tqdm import tqdm, trange

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import datasets, transforms
from torch.autograd import Variable

from utils import *
from models.resnet import *
from bypass_bn import enable_running_stats, disable_running_stats

# load optimizers
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('optimizers'))))
from optimizers import get_optimizer

# load hessain power scheduler
from optimizers.hessian_scheduler import ConstantScheduler, ProportionScheduler, LinearScheduler, CosineScheduler

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--batch-size', type=int, default=256, metavar='B',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='TB',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='E',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.15, metavar='LR',
                    help='learning rate (default: 0.15)')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='learning rate ratio')
parser.add_argument('--lr-decay-epoch', type=int, nargs='+', default=[80, 120],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--depth', type=int, default=20,
                    help='choose the depth of resnet')
parser.add_argument('--optimizer', type=str, default='sassha',
                    help='choose optim')
parser.add_argument('--data', type=str, default='cifar100',
                    help='choose dataset cifar10/100')
parser.add_argument('--noise_level', type=float, default=0.2,
                    help='noise_level')
parser.add_argument('--min_lr', type=float, default=0.0, help="the minimum value of learning rate")

# Second-order optimization settings
parser.add_argument("--n_samples", default=1, type=int, help="the number of sampling")
parser.add_argument('--betas', type=float, nargs='*', default=[0.9, 0.999], help='betas')
parser.add_argument("--eps", default=1e-4, type=float, help="add a small number for stability")
parser.add_argument("--lazy_hessian", default=10, type=int, help="Delayed hessian update.")
parser.add_argument("--clip_threshold", default=0.05, type=float, help="Clipping threshold.")

# Hessian power scheduler
parser.add_argument('--hessian_power_scheduler', type=str, default='constant', help="choose LRScheduler 1. 'constant', 2. 'proportion', 3. 'linear', 4. 'cosine'")
parser.add_argument('--max_hessian_power', type=float, default=1)
parser.add_argument('--min_hessian_power', type=float, default=0.5)

# Sharpness minimization settings
parser.add_argument("--rho", default=0.05, type=float, help="Rho parameter for SAM.")
parser.add_argument("--adaptive", default=False, type=bool, help="True if you want to use the Adaptive SAM.")
parser.add_argument('--project_name', type=str, default='project_name', help="project_name")

args = parser.parse_args()

# wandb logging
wandb_log = True
wandb_project = args.project_name
wandb_run_name = f'{args.optimizer}_lr_{args.lr}_wd_{args.weight_decay}_rho_{args.rho}'

num_classes = ''.join([c for c in args.data if c.isdigit()])
num_classes = int(num_classes)

# set random seed to reproduce the work
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

for arg in vars(args):
    print(arg, getattr(args, arg))
if not os.path.isdir('checkpoint/'):
    os.makedirs('checkpoint/')

# get a dataset (e.g,. cifar10, cifar100)
train_loader, test_loader = getNoisyData(name=args.data, 
                                        train_bs=args.batch_size,
                                        test_bs=args.test_batch_size,
                                        noise_level=args.noise_level)

# get a model
model = resnet(num_classes=num_classes, depth=args.depth).cuda()
print(model)

model = torch.nn.DataParallel(model)
print('    Total params: %.2fM' % (sum(p.numel()
                                       for p in model.parameters()) / 1000000.0))
# define a loss
criterion = nn.CrossEntropyLoss()

# get an optimizer
optimizer, create_graph, two_steps = get_optimizer(model, args)

# learning rate schedule
scheduler = lr_scheduler.MultiStepLR(
    optimizer,
    args.lr_decay_epoch,
    gamma=args.lr_decay,
    last_epoch=-1)

# select a hessian power scheduler
if args.hessian_power_scheduler == 'constant':
    hessian_power_scheduler = ConstantScheduler(
        T_max=args.epochs*len(train_loader), 
        max_value=0.5,
        min_value=0.5)

elif args.hessian_power_scheduler == 'proportion':
    hessian_power_scheduler = ProportionScheduler(
        pytorch_lr_scheduler=scheduler,
        max_lr=args.lr,
        min_lr=args.min_lr,
        max_value=args.max_hessian_power,
        min_value=args.min_hessian_power)

elif args.hessian_power_scheduler == 'linear':
    hessian_power_scheduler = LinearScheduler(
        T_max=args.epochs*len(train_loader), 
        max_value=args.max_hessian_power,
        min_value=args.min_hessian_power)

elif args.hessian_power_scheduler == 'cosine':
    hessian_power_scheduler = CosineScheduler(
        T_max=args.epochs*len(train_loader), 
        max_value=args.max_hessian_power,
        min_value=args.min_hessian_power)

optimizer.hessian_power_scheduler = hessian_power_scheduler

# import and init wandb
if wandb_log:
    import wandb
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.init(project=wandb_project, name=wandb_run_name)
    wandb.config.update(args)

best_acc = 0.0
iter_num = 0
# training loop
for epoch in range(1, args.epochs + 1):
    print('Current Epoch: ', epoch)
    train_loss = 0.
    total_num = 0
    correct = 0

    scheduler.step()
    model.train()

    if args.optimizer == 'msassha':
        optimizer.move_up_to_momentumAscent()

    with tqdm(total=len(train_loader.dataset)) as progressbar:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            if two_steps:
                enable_running_stats(model)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                
                if args.optimizer == 'sassha':
                    optimizer.perturb_weights(zero_grad=True)
                
                elif args.optimizer in ['samsgd', 'samadamw']:
                    optimizer.first_step(zero_grad=True)
                
                disable_running_stats(model)
                criterion(model(data), target).backward(create_graph=create_graph)

                if args.optimizer == 'sassha':
                    optimizer.unperturb()
                    optimizer.step()
                    optimizer.zero_grad()
                
                elif args.optimizer in ['samsgd', 'samadamw']:
                    optimizer.second_step(zero_grad=True)

            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward(create_graph=create_graph)
                optimizer.step()
                optimizer.zero_grad()

            # for records
            train_loss += loss.item() * target.size()[0]
            total_num += target.size()[0]
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
            progressbar.update(target.size(0))
            iter_num += 1

    if args.optimizer == 'msassha':
        optimizer.move_back_from_momentumAscent()

    acc, val_loss = test(model, test_loader, criterion)

    train_loss /= total_num
    train_acc = correct / total_num * 100

    if acc > best_acc:
        best_acc = acc
    
    if wandb_log:
        wandb.log({
            "iter": iter_num,
            "train/loss": train_loss,
            "train/acc": train_acc,
            "val/acc": acc*100,
            "val/loss": val_loss,
            "lr": scheduler.get_last_lr()[-1],
            'best_accuracy': best_acc,
            "hessian_power": optimizer.hessian_power_t if args.optimizer == 'sassha' else 0},
            step=epoch)
    
    print(f"Training Loss of Epoch {epoch}: {np.around(train_loss, 2)}")
    print(f"Testing of Epoch {epoch}: {np.around(acc * 100, 2)} \n")

print(f'Best Acc: {np.around(best_acc * 100, 2)}')
