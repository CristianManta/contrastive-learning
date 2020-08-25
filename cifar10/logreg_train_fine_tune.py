from datetime import datetime

now_unformatted = datetime.now()
time_string = now_unformatted.strftime("%b-%d-%Y_%H-%M-%S")

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

parser.add_argument('--logdir', type=str, default=None, metavar='DIR',
                    help='directory for outputting log files. (default: ./logs/TIMESTAMP/)')
parser.add_argument('--weights', type=str, default=None, metavar='DIR',
                    help='path to pre-trained encoder. (default: ./runs/encoder_best.pth.tar')

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
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum parameter (default: 0.9)')
parser.add_argument('--lr-schedule', type=str, metavar='[[epoch,ratio]]',
                    default='[[0,1],[60,0.2],[120,0.04],[160,0.008]]', help='List of epochs and multiplier '
                                                                            'for changing the learning rate (default: [[0,1],[60,0.2],[120,0.04],[160,0.008]]). ')


args = parser.parse_args()

# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True
# has_cuda = False

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Set and create logging directory
if args.logdir is None:
    args.logdir = os.path.join('./logs/', time_string)

os.makedirs(args.logdir, exist_ok=True)

if args.weights is None:
    args.weights = './runs/encoder_best.pth.tar'


# Print args
print('Contrastive Learning Classification on Cifar-10')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

# We want to be able to reproduce the experiments easily
args_file_path = os.path.join(args.logdir, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

shutil.copyfile('./logreg_train_fine_tune.py', os.path.join(args.logdir, 'logreg_train_fine_tune.py'))


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


# define the Logistic Regression classifier
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, use_softmax=False):
        super(LogisticRegression, self).__init__()
        linear_layer1 = nn.Linear(input_dim, input_dim)
        linear_layer2 = nn.Linear(input_dim, output_dim)
        # linear_layer1.weight.data.fill_(0)
        # linear_layer2.weight.data.fill_(0)
        self.l1 = linear_layer1
        self.l2 = linear_layer2
        self.use_softmax = use_softmax

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        outputs = self.l2(x)
        if self.use_softmax:
            outputs = outputs.softmax(dim=-1)
        return outputs


# Define the composite model
class CompositeModel(nn.Module):
    def __init__(self, base, linear_clf):
        super(CompositeModel, self).__init__()
        self.base = base
        self.linear_clf = linear_clf

    def forward(self, x):
        features, _ = self.base(x)
        features = F.normalize(features, dim=1)
        features = features.detach()
        # features.requires_grad = True

        output = self.linear_clf(features)
        return output


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

savedict = torch.load(args.weights, map_location='cpu')
model.load_state_dict(savedict['state_dict'])

# Get model output dim
for i, (x, _) in enumerate(train_loader):
    if has_cuda:
        x = x.cuda()
    out = model(x)[0]
    break
num_features = out.shape[1]

clf = LogisticRegression(input_dim=num_features, output_dim=classes)
if has_cuda:
    clf = clf.cuda()
    if torch.cuda.device_count() > 1:
        clf = nn.DataParallel(clf)

composite_model = CompositeModel(model, clf)

# the loss function
loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

# define optimizer
# optimizer = torch.optim.SGD(composite_model.parameters(), lr=args.lr, weight_decay=args.decay)


bparams = []
oparams = []
for name, p in composite_model.named_parameters():
    if 'bias' in name:
        bparams.append(p)
    else:
        oparams.append(p)


optimizer = optim.SGD([{'params': oparams, 'weight_decay': args.decay},
                       {'params': bparams, 'weight_decay': 0.}],
                      lr=args.lr,
                      momentum=args.momentum,
                      nesterov=False)


# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=1.5, last_epoch=-1)
def _scheduler(optimizer, args):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(args.lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:, 0], e) - 1, 0), 1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler


scheduler = _scheduler(optimizer, args)


def train(epoch):
    composite_model.train()
    print("Current LR: {}".format(scheduler.get_lr()[0]))

    for batch_ix, (x, y) in enumerate(train_loader):
        if has_cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = composite_model(x)
        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

        if batch_ix % 100 == 0:
            print('[Epoch %2d, batch %3d] training loss: %.3g' %
                  (epoch, batch_ix, loss.data.item()))

    #    if epoch >= 10:
    scheduler.step()


def test():
    composite_model.eval()

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

            output = composite_model(x)

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
        checkpoint = os.path.join(args.logdir, 'classifier_checkpoint.pth.tar')
        best = os.path.join(args.logdir, 'classifier_best.pth.tar')

        torch.save({'state_dict': composite_model.state_dict()}, checkpoint)

        if test_acc > best_acc:
            shutil.copyfile(checkpoint, best)
            best_acc = test_acc

    print(f"Best test accuracy: {best_acc}%")
    accuracy_log = os.path.join(args.logdir, 'accuracy.txt')
    with open(accuracy_log, 'w') as f:
        msg = "Best test accuracy: " + str(best_acc) + "%"
        f.write(msg)


if __name__ == "__main__":
    main()
