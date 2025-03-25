import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

def introduce_label_noise(labels, num_classes, noise_level=0.1):
    n_samples = len(labels)
    n_noisy = int(n_samples * noise_level)
    noisy_indices = torch.randperm(n_samples)[:n_noisy]
    # Generate new random labels for the selected indices
    new_labels = torch.randint(0, num_classes, (n_noisy,))
    labels[noisy_indices] = new_labels
    return labels

def getNoisyData(
        name='cifar10',
        train_bs=128,
        test_bs=1000,
        noise_level=0.1,
        ):

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
    
        trainset = datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)
        
        trainset.targets = introduce_label_noise(torch.tensor(trainset.targets),
                                                    num_classes=10, 
                                                    noise_level=noise_level)

        train_loader = torch.utils.data.DataLoader(
                trainset, batch_size=train_bs, shuffle=True, drop_last=False)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        testset = datasets.CIFAR10(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)
        
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False)

    if name == 'cifar100':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        trainset = datasets.CIFAR100(
            root='../data',
            train=True,
            download=True,
            transform=transform_train)

        trainset.targets = introduce_label_noise(torch.tensor(trainset.targets),
                                                    num_classes=100,
                                                    noise_level=noise_level)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=train_bs, shuffle=True, drop_last=False)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        testset = datasets.CIFAR100(
            root='../data',
            train=False,
            download=False,
            transform=transform_test)

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=test_bs, shuffle=False)

    return train_loader, test_loader


def test(model, test_loader, criterion):
    # print('Testing')
    model.eval()
    correct = 0
    total_num = 0
    val_loss = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            val_loss += loss.item() * target.size()[0]
            total_num += len(data)
    # print('testing_correct: ', correct / total_num, '\n')
    return (correct / total_num, val_loss / total_num)


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads
