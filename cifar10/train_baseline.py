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

from torch.optim.lr_scheduler import LambdaLR

import torchvision.models as models
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset

from statistics import mean
import numpy as np
import ast, bisect

import baseline_models.cifar as cifarmodels

parser = argparse.ArgumentParser('Training template for DNN computer vision research in PyTorch')

parser.add_argument('--data-dir', type=str, default='/home/math/oberman-lab/data/', metavar='DIR',
                    help='Directory where CIFAR-10 data is saved')

parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--logdir', type=str, default=None, metavar='DIR',
                    help='directory for outputting log files. (default: ./logs/DATASET/MODEL/TIMESTAMP/)')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0 )')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')

parser.add_argument('--num-train-images', type=int, default=50000, metavar='NI',
                    help='number of images to use in training (default=50000)')
parser.add_argument('--num-test-images', type=int, default=10000, metavar='NI',
                    help='number of test images to classify (default=10000)')
parser.add_argument('--random-subset', action='store_true',
                    default=False, help='use random subset of train and test images (default: False)')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet50',
                    help='Model architecture (default: ResNet50)')
group1.add_argument('--dropout', type=float, default=0, metavar='P',
                    help='Dropout probability, if model supports dropout (default: 0)')
group1.add_argument('--cutout', type=int, default=0, metavar='N',
                    help='Cutout size, if data loader supports cutout (default: 0)')
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
group1.add_argument('--greyscale', action='store_true', dest='greyscale',
                    help="Make images greyscale")
group1.set_defaults(greyscale=False)

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
group2.add_argument('--decay', type=float, default=5e-4, metavar='L',
                    help='Lagrange multiplier for weight decay (sum '
                         'parameters squared) (default: 5e-4)')
args = parser.parse_args()

if not os.path.exists('./runs_baseline_improved'):
    os.makedirs('./runs_baseline_improved')

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
if args.num_train_images > 50000:
    args.num_train_images = 50000
if args.random_subset:
    Ix = np.random.choice(50000, size=args.num_train_images, replace=False)
    Ix = torch.from_numpy(Ix)
else:
    Ix = torch.arange(args.num_train_images)  # Use the first N images of train set
subset = Subset(ds_train, Ix)
num_train = args.num_train_images
train_loader = torch.utils.data.DataLoader(
    subset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True, drop_last=True)

ds_test = CIFAR10(root, download=True, train=False, transform=transforms.ToTensor())
if args.num_test_images > 10000:
    args.num_test_images = 10000
if args.random_subset:
    Ix = np.random.choice(10000, size=args.num_test_images, replace=False)
    Ix = torch.from_numpy(Ix)
else:
    Ix = torch.arange(args.num_test_images)  # Use the first N images of test set
subset = Subset(ds_test, Ix)
num_test = args.num_test_images
test_loader = torch.utils.data.DataLoader(
    subset,
    batch_size=args.test_batch_size, shuffle=False,
    num_workers=4, pin_memory=True, drop_last=True)

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

bparams = []
oparams = []
for name, p in model.named_parameters():
    if 'bias' in name:
        bparams.append(p)
    else:
        oparams.append(p)

# Only layer weight matrices should have weight decay, not layer biases
optimizer = optim.SGD([{'params': oparams, 'weight_decay': args.decay},
                       {'params': bparams, 'weight_decay': 0.}],
                      lr=args.lr,
                      momentum=args.momentum,
                      nesterov=False)


def _scheduler(optimizer, args):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(args.lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:, 0], e) - 1, 0), 1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler


scheduler = _scheduler(optimizer, args)

# optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
#                                                        last_epoch=-1)
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

    scheduler.step()


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
    print('Accuracy on the test set: %.3g \n' % acc)

    return acc


# def test():
#     model.eval()
#
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#
#         for (x, y) in test_loader:
#
#             if has_cuda:
#                 x, y = x.cuda(), y.cuda()
#
#             Nb = x.shape[0]
#
#             outputs = model(x)
#             predicted = torch.argmax(outputs, dim=1)
#             total += Nb
#             correct += (predicted == y).sum().item()
#
#     acc = 100 * correct / total
#
#     print('Final accuracy on the test set: %.3g \n' % acc)


def main():
    best_acc = 0

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        acc = test()

        torch.save({'state_dict': model.state_dict()}, './runs_baseline_improved/checkpoint.pth.tar')

        if acc > best_acc:
            shutil.copyfile('./runs_baseline_improved/checkpoint.pth.tar', './runs_baseline_improved/best.pth.tar')
            best_acc = acc

    print('Best accuracy on the test set: %.3g \n' % best_acc)


if __name__ == "__main__":
    main()
