import argparse
import os, shutil, sys, pathlib

_pth = str(pathlib.Path(__file__).absolute())
for i in range(2):
    (_pth, _) = os.path.split(_pth)
sys.path.insert(0, _pth)  # I just made sure that the root of the project (ContrastiveTeamO) is in the path where Python
# looks for packages in order to import from files that require going several levels up from the directory where this
# script is. Unfortunately, by default Python doesn't allow imports from above the current file directory.

import time
from datetime import datetime

now_unformatted = datetime.now()
time_string = now_unformatted.strftime("%b-%d-%Y_%H-%M-%S")

import random
import ast
import numpy as np
from statistics import mean

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.backends.cudnn as cudnn
from torch.autograd import grad

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import torchvision
import transformations.transforms as transforms
from torchvision.datasets import CIFAR10

import models.cifar as cifarmodels
from loss.nt_xent import NTXentLoss

parser = argparse.ArgumentParser('contrastive learning training on CIFAR-10')
parser.add_argument('--data-dir', type=str, default='/home/math/oberman-lab/data/', metavar='DIR',
                    help='Directory where CIFAR-10 data is saved')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status (default: 50)')

parser.add_argument('--logdir', type=str, default=None, metavar='DIR',
                    help='directory for outputting log files. (default: ./logs/TIMESTAMP/)')


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
group2.add_argument('--penalty', type=float, default=0.1, metavar='L',
                    help='Tikhonov regularization parameter (default: 0.1)')
group2.add_argument('--norm', type=str, choices=['L1', 'L2', 'Linf'], default='L2',
                    help='norm for gradient penalty, wrt model inputs. (default: L2)'
                         ' Note that this should be dual to the norm measuring adversarial perturbations')
group2.add_argument('--h', type=float, default=1e-2, metavar='H',
                    help='finite difference step size (default: 1e-2)')
group2.add_argument('--fd-order', type=str, choices=['O1', 'O2'], default='O1',
                    help='accuracy of finite differences (default: O1)')

args = parser.parse_args()

# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True

# Set random seed
if args.seed is None:
    args.seed = int(time.time())
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Create logging directory
if args.logdir is None:
    args.logdir = os.path.join('./logs/', time_string)
os.makedirs(args.logdir, exist_ok=True)

# Print args
print('Contrastive Learning on Cifar-10')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

args_file_path = os.path.join(args.logdir, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

shutil.copyfile('./classic_tikhonov_exact.py', os.path.join(args.logdir, 'classic_tikhonov_exact.py'))


# Get Train and Test Loaders
root = os.path.join(args.data_dir, 'cifar10')


class SimCLRDataTransform:
    """Produces the 2 data augmentations for each image. To be called on a batch of images (Tensor of shape BxCxWxH
    or BxCxHxW)."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        xis = x.clone()
        xjs = x.clone()
        for i in range(x.shape[0]):
            xis[i] = self.transform(xis[i])
            xjs[i] = self.transform(xjs[i])
        return xis, xjs


class RandomGrayscale:
    """In-house random grayscale. Converts ONE tensor image to grayscale with probability p
    and outputs a Tensor image. The PyTorch RandomGrayscale doesn't take Tensor type arguments."""

    def __init__(self, p=0.1):
        self.p = p

    def to_grayscale(self, img):
        new_img = torch.mean(img, dim=0, keepdim=True)
        new_img = torch.cat((new_img, new_img, new_img), dim=0)
        return new_img

    def __call__(self, img):
        if random.random() < self.p:
            gray_img = self.to_grayscale(img)
            return gray_img
        return img


def get_color_distortion(s=1.0):
    """Transforms a Tensor image. s is the strength of color distortion."""
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([
        rnd_color_jitter,
        rnd_gray])
    return color_distort


data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=32),
                                      transforms.RandomHorizontalFlip(),
                                      get_color_distortion(s=1.0)])

data_augment = SimCLRDataTransform(data_transforms)

ds_train = CIFAR10(root, download=True, train=True, transform=transforms.ToTensor())

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

# Set Optimizer and learning rate schedule
optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                       last_epoch=-1)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.1, last_epoch=-1)

# define loss
nt_xent_criterion = NTXentLoss(device=torch.cuda.current_device(), batch_size=args.batch_size, temperature=0.5,
                               use_cosine_similarity=True)
# TODO: Check again. Chris defined 2 criterions: one for training and another one for test. I think this is fine here.

# --------
# Training
# --------


ix = 0  # count of gradient steps
tik = args.penalty
regularizing = tik > 0
h = args.h  # finite difference step size


def train(epoch):
    global ix

    # Put the model in train mode (unfreeze batch norm parameters)
    model.train()
    print("Current LR: {}".format(scheduler.get_lr()[0]))

    # Run through the training data
    if has_cuda:
        torch.cuda.synchronize()

    for batch_ix, (x, target) in enumerate(train_loader):

        optimizer.zero_grad()

        xis, xjs = data_augment(x)  # NOTE: x isn't on CUDA. I reserve cuda for xis and xjs

        if has_cuda:
            xis = xis.cuda()
            xjs = xjs.cuda()

        if regularizing:
            xis.requires_grad_(True)
            xjs.requires_grad_(True)

        his, zis = model(xis)
        hjs, zjs = model(xjs)

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = nt_xent_criterion(zis, zjs)

        # Compute finite difference approximation of directional derivative of grad loss wrt inputs
        if regularizing:

            dx_xis = grad(loss, xis, retain_graph=True, create_graph=True)[0]
            sh = dx_xis.shape
            xis.requires_grad_(False)

            dx_xjs = grad(loss, xjs, retain_graph=True, create_graph=True)[0]
            xjs.requires_grad_(False)

            # v is the finite difference direction.
            # For example, if norm=='L2', v is the gradient of the loss wrt inputs
            v_xis = dx_xis.view(sh[0], -1)
            v_xjs = dx_xjs.view(sh[0], -1)
            Nb, Nd = v_xis.shape

            if args.norm == 'L2':
                nv_xis = v_xis.norm(2, dim=-1)
                nv_xjs = v_xjs.norm(2, dim=-1)
            else:
                raise ValueError("Only L2 norm is supported for now.")

        tik_penalty = torch.tensor(np.nan)
        dlmean = torch.tensor(np.nan)
        dlmax = torch.tensor(np.nan)
        if regularizing:
            tik_penalty_xis = nv_xis.pow(2).sum() / 2
            tik_penalty_xjs = nv_xjs.pow(2).sum() / 2
            tik_penalty = 0.5 * (tik_penalty_xis + tik_penalty_xjs)
            # print(f"loss before: {loss}")
            loss = loss + tik * tik_penalty
            # print(f"loss after: {loss}")
            # print(f"tik_penalty = {tik_penalty}")
            # print("\n")

        loss.backward()
        optimizer.step()

        if np.isnan(loss.data.item()):
            raise ValueError('model returned nan during training')

        if batch_ix % args.log_interval == 0 and batch_ix > 0:
            print('[epoch %2d, batch %3d] penalized training loss: %.3g' %
                  (epoch, batch_ix, loss.data.item()))
        ix += 1

    if has_cuda:
        torch.cuda.synchronize()


# ------------------
# Evaluate on validation set
# ------------------

def test():
    model.eval()

    loss_vals = []

    with torch.no_grad():

        for (x, target) in valid_loader:

            xis, xjs = data_augment(x)

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

    save_model_path = os.path.join(args.logdir, 'encoder_checkpoint.pth.tar')
    best_model_path = os.path.join(args.logdir, 'encoder_best.pth.tar')

    for epoch in range(1, args.epochs + 1):
        train(epoch)
        scheduler.step()
        test_loss = test()

        torch.save({'state_dict': model.state_dict()}, save_model_path)

        if test_loss < best_loss:
            shutil.copyfile(save_model_path, best_model_path)
            best_loss = test_loss


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt; exiting')
