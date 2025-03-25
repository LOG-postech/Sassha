import os
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


def getData(
        name='cifar10',
        path='/home/dahunshin/imagenet',
        train_bs=256,
        test_bs=1000,
        num_workers=1,
        distributed=False):

    if name == 'mnist':

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_bs, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train=False,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.1307,
                             ),
                            (0.3081,
                             ))])),
            batch_size=test_bs,
            shuffle=False)


    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        trainset = datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)
        
        testset = datasets.CIFAR10(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True, drop_last=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            test_sampler = None

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False,
            num_workers=num_workers, pin_memory=True, sampler=test_sampler)
        

    if name == 'cifar100':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        
        trainset = datasets.CIFAR100(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)
        
        testset = datasets.CIFAR100(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True, drop_last=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(testset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            test_sampler = None

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False,
            num_workers=num_workers, pin_memory=True, sampler=test_sampler)


    if name == 'imagenet':
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, drop_last=False)
            test_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=True)
        else:
            train_sampler = None
            test_sampler = None

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_bs, shuffle=(train_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=train_sampler)

        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=test_bs, shuffle=False,
            num_workers=num_workers, pin_memory=True, sampler=test_sampler)

    
    return train_loader, test_loader, train_sampler, test_sampler
    