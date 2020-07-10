""" training a constructive learning model on CIFAR-10 """

import argparse
import os, shutil, sys
import yaml
import numpy as np
import random
import ast, bisect
import time, datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torchnet as tnt

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

import models.cifar as cifarmodels

from ContrastiveTeamO.custom_transforms import get_color_distortion, GaussianBlur

parser = argparse.ArgumentParser('constructive learning training on CIFAR-10')
parser.add_argument('--data-dir', type=str, default='/home/campus/oberman-lab/data/',metavar='DIR', 
        help='Directory where CIFAR-10 data is saved')
parser.add_argument('--seed', type=int, default=0, metavar='S',
        help='random seed (default: 0)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
        help='number of epochs to train (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
        help='input batch size for testing (default: 1000)')
#parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#        help='how many batches to wait before logging training status (default: 100)')
#parser.add_argument('--logdir', type=str, default=None,metavar='DIR',
#        help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/)')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet34',
        help='Model architecture (default: ResNet34)')
group1.add_argument('--dropout',type=float, default=0, metavar='P',
        help = 'Dropout probability, if model supports dropout (default: 0)')
group1.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm")
group1.add_argument('--no-bn',action='store_false', dest='bn',
       help = "Don't use batch norm")
group1.set_defaults(bn=True)
group1.add_argument('--last-layer-nonlinear', 
        action='store_true', default=False)
group1.add_argument('--bias',action='store_true', dest='bias',
        help = "Use model biases")
group1.add_argument('--no-bias',action='store_false', dest='bias',
       help = "Don't use biases")
group1.set_defaults(bias=False)
group1.add_argument('--kernel-size',type=int, default=3, metavar='K',
        help='convolution kernel size (default: 3)')
group1.add_argument('--model-args',type=str, 
        default="{}",metavar='ARGS',
        help='A dictionary of extra arguments passed to the model.'
        ' (default: "{}")')

group0 = parser.add_argument_group('Optimizer hyperparameters')
group0.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='Input batch size for training. (default: 128)')
group0.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='Initial step size. (default: 0.1)')
group0.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
        default='[[0,1],[60,0.2],[120,0.04],[160,0.008]]', help='List of epochs and multiplier '
        'for changing the learning rate (default: [[0,1],[60,0.2],[120,0.04],[160,0.008]]). ')
group0.add_argument('--momentum', type=float, default=0.9, metavar='M',
       help='SGD momentum parameter (default: 0.9)')

group2 = parser.add_argument_group('Regularizers')
group2.add_argument('--decay',type=float, default=5e-4, metavar='L',
        help='Lagrange multiplier for weight decay (sum '
        'parameters squared) (default: 5e-4)')

args = parser.parse_args()

# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Print args
print('Contrastive Learning on Cifar-10')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

# Set and create logging directory
#if args.logdir is None:
#    args.logdir = os.path.join('./logs/',args.dataset,args.model,
#            '{0:%Y-%m-%dT%H%M%S}'.format(datetime.datetime.now()))
#os.makedirs(args.logdir, exist_ok=True)

# Get Train and Test Loaders
# Do 3 deparate train loaders, one with each data augmentation
root = os.path.join(args.data_dir,'cifar10')

ds_train1 = CIFAR10(root, download=True, train=True, transform=transforms.Compose([transforms.RandomResizedCrop(size=32), transforms.ToTensor()]))
train_loader1 = torch.utils.data.DataLoader(
                    ds_train1,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
ds_train2 = CIFAR10(root, download=True, train=True, transform=transforms.Compose([get_color_distortion(), transforms.ToTensor()]))
train_loader2 = torch.utils.data.DataLoader(
                    ds_train2,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)
ds_train3 = CIFAR10(root, download=True, train=True, transform=transforms.Compose([GaussianBlur(kernel_size=3), transforms.ToTensor()]))
train_loader3 = torch.utils.data.DataLoader(
                    ds_train3,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

ds_test = CIFAR10(root, download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(
                    ds_test,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=True)

# initialize model and move it the GPU (if available)
classes = 10
model_args = ast.literal_eval(args.model_args)
in_channels = 3
model_args.update(bn=args.bn, classes=classes, bias=args.bias,
                  kernel_size=args.kernel_size, 
                  in_channels=in_channels,
                  softmax=False,last_layer_nonlinear=args.last_layer_nonlinear,
                  dropout=args.dropout)
model = getattr(cifarmodels, args.model)(**model_args)

if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

# Set Optimizer and learning rate schedule
bparams=[]
oparams=[]
for name, p in model.named_parameters():
    if 'bias' in name:
        bparams.append(p)
    else:
        oparams.append(p)

# Only layer weight matrices should have weight decay, not layer biases
optimizer = optim.SGD([{'params':oparams,'weight_decay':args.decay},
                       {'params':bparams,'weight_decay':0.}],
                  lr = args.lr,
                  momentum = args.momentum,
                  nesterov = False)

def scheduler(optimizer,args):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(args.lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:,0], e)-1,0),1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler
schedule = scheduler(optimizer,args)

# training code

def train(epoch):
    model.train()

    for i, (data1, data2, data3) in enumerate(zip(train_loader1, train_loader2, train_loader3)):
        # So we just loaded in 3 copies of the same batch of images, each with a different transformation

        y, x1, x2, x3 = data1[1], data1[0], data2[0], data3[0]

        if has_cuda:
            y = y.cuda()
            x1, x2, x3 = x1.cuda(), x2.cuda(), x3.cuda()

        exit()

def main():
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()

if __name__=="__main__":
    main()
