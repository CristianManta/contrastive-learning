import os, sys, pathlib, shutil

_pth = str(pathlib.Path(__file__).absolute())
for i in range(2):
    (_pth, _) = os.path.split(_pth)
sys.path.insert(0, _pth)  # I just made sure that the root of the project (ContrastiveTeamO) is in the path where Python
# looks for packages in order to import from files that require going several levels up from the directory where this
# script is. Unfortunately, by default Python doesn't allow imports from above the current file directory.

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim

import torchvision.models as models
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from statistics import mean
import numpy as np
import ast

import baseline_models.cifar as cifarmodels

parser = argparse.ArgumentParser('contrastive learning training on CIFAR-10')
parser.add_argument('--data-dir', type=str, default='/home/math/oberman-lab/data/', metavar='DIR',
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
                    help='Initial step size. (default: 3e-4)')
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

if not os.path.exists('./runs_baseline'):
    os.makedirs('./runs_baseline')

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

# Get data
root = '/home/math/oberman-lab/data/cifar10'

ds_train = CIFAR10(root, download=True, train=True, transform=transforms.ToTensor())
ds_test = CIFAR10(root, download=True, train=False, transform=transforms.ToTensor())

num_train = len(ds_train)
indices = list(range(num_train))
np.random.shuffle(indices)

valid_size = 0.05
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(ds_train, batch_size=128, num_workers=4, drop_last=True, sampler=train_sampler)
valid_loader = DataLoader(ds_train, batch_size=128, num_workers=4, drop_last=True, sampler=valid_sampler)
test_loader = DataLoader(ds_test, batch_size=128, num_workers=4, drop_last=True)

# initialize model and move it the GPU (if available)
classes = 10
model_args = ast.literal_eval(args.model_args)
in_channels = 3
model_args.update(bn=args.bn, classes=classes, bias=args.bias,
                  kernel_size=args.kernel_size,
                  in_channels=in_channels,
                  softmax=False, last_layer_nonlinear=args.last_layer_nonlinear,
                  dropout=args.dropout)
model = getattr(cifarmodels, 'ResNet50')(**model_args)

if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

# Set Optimizer and learning rate schedule
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                       last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)

# define loss
criterion = nn.CrossEntropyLoss()


# training code

def train(epoch):
    model.train()
    batch_ix = 0

    print("Current LR: {}".format(scheduler.get_lr()[0]))
    for (x, y) in train_loader:

        if has_cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        if batch_ix % 100 == 0:
            print('[Epoch %2d, batch %3d] training loss: %.3g' %
                  (epoch, batch_ix, loss.data.item()))

        batch_ix += 1

    # warmup for the first 10 epochs
    if epoch >= 10:
        scheduler.step()


def validate():
    model.eval()

    loss_vals = []

    correct = 0
    total = 0

    with torch.no_grad():

        for (x, y) in valid_loader:

            if has_cuda:
                x, y = x.cuda(), y.cuda()

            Nb = x.shape[0]

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)
            total += Nb
            correct += (predicted == y).sum().item()

            loss = criterion(outputs, y)
            loss_vals.append(loss.data.item())

    loss_val = mean(loss_vals)
    acc = 100 * correct / total
    print('Avg. test loss: %.3g \n' % loss_val)
    print('Accuracy on the validation set: %.3g \n' % acc)

    return loss_val


def test():
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for (x, y) in test_loader:

            if has_cuda:
                x, y = x.cuda(), y.cuda()

            Nb = x.shape[0]

            outputs = model(x)
            predicted = torch.argmax(outputs, dim=1)
            total += Nb
            correct += (predicted == y).sum().item()

    acc = 100 * correct / total

    print('Final accuracy on the test set: %.3g \n' % acc)


def main():
    best_loss = 10000.0

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        val_loss = validate()

        torch.save({'state_dict': model.state_dict()}, './runs_baseline/checkpoint.pth.tar')

        if val_loss < best_loss:
            shutil.copyfile('./runs_baseline/checkpoint.pth.tar', './runs_baseline/best.pth.tar')
            best_loss = val_loss

    test()


if __name__ == "__main__":
    main()
