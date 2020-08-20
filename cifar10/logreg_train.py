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
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler
# from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import models.cifar as cifarmodels
from loss.nt_xent import NTXentLoss

from transformations.custom_transforms import get_color_distortion, GaussianBlur

parser = argparse.ArgumentParser('Use logistic regression to classify a contrastive learning model')
parser.add_argument('--data-dir', type=str, default='/home/math/oberman-lab/data/', metavar='DIR',
                    help='Directory where CIFAR-10 data is saved')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training. (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--num-train-images', type=int, default=50000, metavar='NI',
                    help='number of images to use in training (default=50000)')
parser.add_argument('--num-test-images', type=int, default=10000, metavar='NI',
                    help='number of test images to classify (default=10000)')
parser.add_argument('--random-subset', action='store_true',
                    default=False, help='use random subset of train and test images (default: False)')
parser.add_argument('--fine-tune', action='store_true', dest='fine_tune',
                    help="Whether to fine-tune the whole encoder + LR or not (default: False)")

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

parser.add_argument('--lr', type=float, default=0.8,
                    help='Starting learning rate of the classifier (default: 0.8)')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of training epochs (default: 100)')
parser.add_argument('--decay', type=float, default=0.0,
                    help='Weight decay (default: 0)')

args = parser.parse_args()

# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True
# has_cuda = False

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Print args
print('Contrastive Learning Classification on Cifar-10')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

# make dataloaders
root = os.path.join(args.data_dir, 'cifar10')

ds_test = CIFAR10(root, download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
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
    num_workers=4, pin_memory=True)

ds_train = CIFAR10(root, download=True, train=True, transform=transforms.Compose([transforms.ToTensor()]))
if args.num_train_images > 50000:
    args.num_train_images = 50000
if args.random_subset:
    Ix = np.random.choice(50000, size=args.num_train_images, replace=False)
    Ix = torch.from_numpy(Ix)
else:
    Ix = torch.arange(args.num_train_images)  # Use the first N images of test set
subset = Subset(ds_train, Ix)
num_train = args.num_train_images
train_loader = torch.utils.data.DataLoader(
    subset,
    batch_size=args.batch_size, shuffle=True,
    num_workers=4, pin_memory=True)

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

savedict = torch.load('./runs/encoder_best.pth.tar', map_location='cpu')
model.load_state_dict(savedict['state_dict'])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# Get model output dim
for i, (x, _) in enumerate(train_loader):
    if has_cuda:
        x = x.cuda()
    out = model(x)[0]
    break
num_features = out.shape[1]


# define the Logistic Regression classifier
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_softmax=False):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.use_softmax = use_softmax

    def forward(self, x):
        outputs = self.linear(x)
        if self.use_softmax:
            outputs = outputs.softmax(dim=-1)
        return outputs


clf = LogisticRegression(input_dim=num_features, output_dim=classes)
if has_cuda:
    clf = clf.cuda()
    if torch.cuda.device_count() > 1:
        clf = nn.DataParallel(clf)

# the loss function
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

# define optimizer
optimizer = torch.optim.SGD(clf.parameters(), lr=args.lr, weight_decay=args.decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1.5, last_epoch=-1)


def train(epoch):
    model.eval()
    clf.train()
    print("Current LR: {}".format(scheduler.get_lr()[0]))

    for batch_ix, (x, y) in enumerate(train_loader):
        if has_cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()

        features, _ = model(x)
        features = F.normalize(features, dim=1)  # Added this to try to improve accuracy

        features = features.detach()
        features.requires_grad = True

        output = clf(features)

        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

        if batch_ix % 100 == 0:
            print('[Epoch %2d, batch %3d] training loss: %.3g' %
                  (epoch, batch_ix, loss.data.item()))

    #    if epoch >= 10:
    scheduler.step()


def test():
    model.eval()
    clf.eval()

    test_loss = tnt.meter.AverageValueMeter()

    # top1 = tnt.meter.ClassErrorMeter(accuracy=True)

    # Note: I replaced the top1 meter by a manual computation of the test accuracy after a series of
    # inconsistent outputs by tnt.meter combined with poor documentation on its parameters

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_ix, (x, y) in enumerate(test_loader):
            if has_cuda:
                x, y = x.cuda(), y.cuda()

            Nb = x.shape[0]

            features, _ = model(x)
            features = F.normalize(features, dim=1)  # Added this to try to improve accuracy
            output = clf(features)

            predicted = torch.argmax(output, dim=1)
            total += Nb
            correct += (predicted == y).sum().item()

            loss = loss_fn(output, y)

            # top1.add(output, y)
            test_loss.add(loss.item())

    loss_val = test_loss.value()[0]
    # acc_val = top1.value()[0]
    acc_val = 100 * correct / total

    print('        test loss: %.3g' % loss_val)
    print('        test acc : %.3f' % acc_val)

    return loss_val, acc_val


def main():
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test_loss, test_acc = test()

        torch.save({'state_dict': clf.state_dict()}, './runs/classifier_checkpoint.pth.tar')

        if test_acc > best_acc:
            shutil.copyfile('./runs/classifier_checkpoint.pth.tar', './runs/classifier_best.pth.tar')
            best_acc = test_acc


if __name__ == "__main__":
    main()
