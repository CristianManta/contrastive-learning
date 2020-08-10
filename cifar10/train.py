""" training a contrastive learning model on CIFAR-10.
 Optimizer-related arguments not used yet as parameters:
 --> lr-schedule
 --> momentum"""

import argparse
import os, shutil, sys, pathlib

_pth = str(pathlib.Path(__file__).absolute())
for i in range(2):
    (_pth, _) = os.path.split(_pth)
sys.path.insert(0, _pth)  # I just made sure that the root of the project (ContrastiveTeamO) is in the path where Python
# looks for packages in order to import from files that require going several levels up from the directory where this
# script is. Unfortunately, by default Python doesn't allow imports from above the current file directory.


import yaml
import numpy as np
import random
import ast, bisect
from statistics import mean

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torchnet as tnt

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import models.cifar as cifarmodels
from loss.nt_xent import NTXentLoss

from transformations.custom_transforms import get_color_distortion, GaussianBlur

parser = argparse.ArgumentParser('constructive learning training on CIFAR-10')
parser.add_argument('--data-dir', type=str, default='/home/campus/oberman-lab/data/', metavar='DIR',
                    help='Directory where CIFAR-10 data is saved')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#        help='how many batches to wait before logging training status (default: 100)')
# parser.add_argument('--logdir', type=str, default=None,metavar='DIR',
#        help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/)')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet50',
                    help='Model architecture (default: ResNet50)')
group1.add_argument('--dropout', type=float, default=0, metavar='P',
                    help='Dropout probability, if model supports dropout (default: 0)')
group1.add_argument('--bn', action='store_true', dest='bn',
                    help="Use batch norm")
group1.add_argument('--no-bn', action='store_false', dest='bn',
                    help="Don't use batch norm")
group1.set_defaults(bn=True)
group1.add_argument('--last-layer-nonlinear',
                    action='store_true', default=False)
group1.add_argument('--bias', action='store_true', dest='bias',
                    help="Use model biases")
group1.add_argument('--no-bias', action='store_false', dest='bias',
                    help="Don't use biases")
group1.set_defaults(bias=False)
group1.add_argument('--kernel-size', type=int, default=3, metavar='K',
                    help='convolution kernel size (default: 3)')
group1.add_argument('--model-args', type=str,
                    default="{}", metavar='ARGS',
                    help='A dictionary of extra arguments passed to the model.'
                         ' (default: "{}")')

group0 = parser.add_argument_group('Optimizer hyperparameters')
group0.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training. (default: 128)')
group0.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                    help='Initial step size. (default: 0.15)')
group0.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
                    default='[[0,1],[30,0.2],[60,0.04],[80,0.008]]', help='List of epochs and multiplier '
                                                                          'for changing the learning rate (default: [[0,1],[30,0.2],[60,0.04],[80,0.008]]). ')
group0.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum parameter (default: 0.9)')

group2 = parser.add_argument_group('Regularizers')
group2.add_argument('--decay', type=float, default=1e-5, metavar='L',
                    help='Lagrange multiplier for weight decay (sum '
                         'parameters squared) (default: 1e-5)')

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
    print('  ', p[0] + ': ', p[1])
print('\n')

# Set and create logging directory
# if args.logdir is None:
#    args.logdir = os.path.join('./logs/',args.dataset,args.model,
#            '{0:%Y-%m-%dT%H%M%S}'.format(datetime.datetime.now()))
# os.makedirs(args.logdir, exist_ok=True)

# Get Train and Test Loaders
# Do 3 deparate train loaders, one with each data augmentation
root = os.path.join(args.data_dir, 'cifar10')


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                      transforms.RandomHorizontalFlip(),
                                      get_color_distortion(s=1.0),
                                      transforms.ToTensor()])

data_augment = data_transforms
ds_train = CIFAR10(root, download=True, train=True, transform=SimCLRDataTransform(data_augment))

num_train = len(ds_train)
indices = list(range(num_train))
np.random.shuffle(indices)

valid_size = 0.05
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=train_sampler,
                          num_workers=4, drop_last=True, shuffle=False)
valid_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=valid_sampler,
                          num_workers=4, drop_last=True)

# initialize model and move it the GPU (if available)
classes = 10
model_args = ast.literal_eval(args.model_args)
in_channels = 3
model_args.update(bn=args.bn, classes=classes, bias=args.bias,
                  kernel_size=args.kernel_size,
                  in_channels=in_channels,
                  softmax=False, last_layer_nonlinear=args.last_layer_nonlinear,
                  dropout=args.dropout)
model = getattr(cifarmodels, args.model)(**model_args)

if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

# print(model)

# Set Optimizer and learning rate schedule
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                       last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)

# define loss
nt_xent_criterion = NTXentLoss(device=torch.cuda.current_device(), batch_size=args.batch_size, temperature=0.5,
                               use_cosine_similarity=True)


# training code

def train(epoch):
    model.train()
    batch_ix = 0

    print("Current LR: {}".format(scheduler.get_lr()[0]))
    for (xis, xjs), y in train_loader:

        if has_cuda:
            xis, xjs = xis.cuda(), xjs.cuda()

        optimizer.zero_grad()

        his, zis = model(xis)
        hjs, zjs = model(xjs)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = nt_xent_criterion(zis, zjs)

        loss.backward()
        optimizer.step()

        if batch_ix % 100 == 0:
            print('[Epoch %2d, batch %3d] training loss: %.3g' %
                  (epoch, batch_ix, loss.data.item()))

        batch_ix += 1

    # warmup for the first 10 epochs
    if epoch >= 10:
        scheduler.step()


def test():
    model.eval()

    loss_vals = []

    with torch.no_grad():

        for (xis, xjs), y in valid_loader:

            if has_cuda:
                xis, xjs = xis.cuda(), xjs.cuda()

            his, zis = model(xis)
            hjs, zjs = model(xjs)

            zis = F.normalize(zis, dim=1)
            zjs = F.normalize(zjs, dim=1)

            loss = nt_xent_criterion(zis, zjs)
            loss_vals.append(loss.data.item())

    loss_val = mean(loss_vals)
    print('Avg. test loss: %.3g \n' % (loss_val))

    return loss_val


def main():
    best_loss = 10000.0

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss = test()

        torch.save({'state_dict': model.state_dict()}, './runs/encoder_checkpoint.pth.tar')

        if test_loss < best_loss:
            shutil.copyfile('./runs/encoder_checkpoint.pth.tar', './runs/encoder_best.pth.tar')
            best_loss = test_loss


if __name__ == "__main__":
    main()
