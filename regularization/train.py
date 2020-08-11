import argparse
import os, shutil, sys, pathlib

_pth = str(pathlib.Path(__file__).absolute())
for i in range(2):
    (_pth, _) = os.path.split(_pth)
sys.path.insert(0, _pth)  # I just made sure that the root of the project (ContrastiveTeamO) is in the path where Python
# looks for packages in order to import from files that require going several levels up from the directory where this
# script is. Unfortunately, by default Python doesn't allow imports from above the current file directory.


import cv2
import random
import time, datetime
import yaml
import ast, bisect
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import torchvision
import transformations.transforms as transforms
from torchvision.datasets import CIFAR10

import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import grad
import torchnet as tnt

# from transformations.custom_transforms import get_color_distortion
import models.cifar as cifarmodels
from loss.nt_xent import NTXentLoss

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
                         'parameters squared) (default: 1e-6)')
group2.add_argument('--penalty', type=float, default=0, metavar='L',
                    help='Tikhonov regularization parameter (squared norm gradient wrt input)')
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
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

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


class SimCLRDataTransform:
    """Produces the 2 data augmentations for each image. To be called on a batch of images (Tensor of shape BxCxWxH)"""

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
    """Converts ONE tensor image to grayscale with probability p and outputs a Tensor image"""

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
    # transforms a Tensor image
    # s is the strength of color distortion.
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

# def one_image_at_a_time_transform(x, transform):
#     for i in range(x.shape[0]):
#         x[i] = transform(x[i])
#     return x

for (x, y) in train_loader:
    x = x.cuda()
    x.requires_grad_(True)
    xis, xjs = data_augment(x)

print("No bug\n")

exit(0)

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

# --------
# Training
# --------


# trainlog = os.path.join(args.logdir, 'training.csv')
# traincolumns = ['index', 'time', 'loss', 'regularizer']
# with open(trainlog, 'w') as f:
#     logger = csv.DictWriter(f, traincolumns)
#     logger.writeheader()

ix = 0  # count of gradient steps

tik = args.penalty

regularizing = tik > 0

h = args.h  # finite difference step size


def train(epoch, ttot):
    global ix

    # Put the model in train mode (unfreeze batch norm parameters)
    model.train()

    # Run through the training data
    if has_cuda:
        torch.cuda.synchronize()
    tepoch = time.perf_counter()

    for batch_ix, (x, target) in enumerate(train_loader):

        if has_cuda:
            x = x.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        if regularizing:
            x.requires_grad_(True)

        xis, xjs = data_augment(x)  # Not sure if this block should be placed before the x.cuda() line
        # xjs = data_augment(x)

        his, zis = model(xis)
        hjs, zjs = model(xjs)

        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        loss = nt_xent_criterion(zis, zjs)

        # Compute finite difference approximation of directional derivative of grad loss wrt inputs
        if regularizing:

            dx = grad(loss, x, retain_graph=True)[0]
            sh = dx.shape
            x.requires_grad_(False)

            # v is the finite difference direction.
            # For example, if norm=='L2', v is the gradient of the loss wrt inputs
            v = dx.view(sh[0], -1)
            Nb, Nd = v.shape

            if args.norm == 'L2':
                nv = v.norm(2, dim=-1,
                            keepdim=True)  # TODO: Question: Why isn't tik_penalty set to (nv.pow(2)).mean()/2?
                nz = nv.view(-1) > 0
                v[nz] = v[nz].div(nv[nz])  # Normalizing the gradient
            if args.norm == 'L1':
                v = v.sign()
                v = v / np.sqrt(Nd)
            elif args.norm == 'Linf':
                vmax, Jmax = v.abs().max(dim=-1)
                sg = v.sign()
                I = torch.arange(Nb, device=v.device)
                sg = sg[I, Jmax]

                v = torch.zeros_like(v)
                I = I * Nd
                Ix = Jmax + I
                v.put_(Ix, sg)

            v = v.view(sh)
            xf = x + h * v
            print(xf.is_cuda)
            exit(0)

            mf = model(xf)
            lf = train_criterion(mf, target)
            if args.fd_order == 'O2':
                xb = x - h * v
                mb = model(xb)
                lb = train_criterion(mb, target)
                H = 2 * h
            else:
                H = h
                lb = lx
            dl = (lf - lb) / H  # This is the finite difference approximation
            # of the directional derivative of the loss

        tik_penalty = torch.tensor(np.nan)
        dlmean = torch.tensor(np.nan)
        dlmax = torch.tensor(np.nan)
        if tik > 0:
            dl2 = dl.pow(2)
            tik_penalty = dl2.mean() / 2
            loss = loss + tik * tik_penalty

        loss.backward()  # TODO: It seems that there is already a double backprop going on (one for weights and one for x)

        optimizer.step()

        if np.isnan(loss.data.item()):
            raise ValueError('model returned nan during training')

        t = ttot + time.perf_counter() - tepoch
        fmt = '{:.4f}'
        logger.writerow({'index': ix,
                         'time': fmt.format(t),
                         'loss': fmt.format(loss.item()),
                         'regularizer': fmt.format(tik_penalty)})

        if (batch_ix % args.log_interval == 0 and batch_ix > 0):
            print('[%2d, %3d] penalized training loss: %.3g' %
                  (epoch, batch_ix, loss.data.item()))
        ix += 1

    if has_cuda:
        torch.cuda.synchronize()

    return ttot + time.perf_counter() - tepoch


# ------------------
# Evaluate test data
# ------------------
# testlog = os.path.join(args.logdir, 'test.csv')
# testcolumns = ['epoch', 'time', 'fval', 'pct_err', 'train_fval', 'train_pct_err']
# with open(testlog, 'w') as f:
#     logger = csv.DictWriter(f, testcolumns)
#     logger.writeheader()


def test(epoch, ttot):
    model.eval()

    with torch.no_grad():

        # Get the true training loss and error
        top1_train = tnt.meter.ClassErrorMeter()
        train_loss = tnt.meter.AverageValueMeter()
        for data, target in train_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = model(data)

            top1_train.add(output.data, target.data)
            loss = criterion(output, target)
            train_loss.add(loss.data.item())

        t1t = top1_train.value()[0]
        lt = train_loss.value()[0]

        # Evaluate test data
        test_loss = tnt.meter.AverageValueMeter()
        top1 = tnt.meter.ClassErrorMeter()
        for data, target in test_loader:
            if has_cuda:
                target = target.cuda(0)
                data = data.cuda(0)

            output = model(data)

            loss = criterion(output, target)

            top1.add(output, target)
            test_loss.add(loss.item())

        t1 = top1.value()[0]
        l = test_loss.value()[0]

    # Report results
    with open(testlog, 'a') as f:
        logger = csv.DictWriter(f, testcolumns)
        fmt = '{:.4f}'
        logger.writerow({'epoch': epoch,
                         'fval': fmt.format(l),
                         'pct_err': fmt.format(t1),
                         'train_fval': fmt.format(lt),
                         'train_pct_err': fmt.format(t1t),
                         'time': fmt.format(ttot)})

    print('[Epoch %2d] Average test loss: %.3f, error: %.2f%%'
          % (epoch, l, t1))
    print('%28s: %.3f, error: %.2f%%\n'
          % ('training loss', lt, t1t))

    return test_loss.value()[0], top1.value()[0]


def main():
    # save_model_path = os.path.join(args.logdir, 'checkpoint.pth.tar')
    # best_model_path = os.path.join(args.logdir, 'best.pth.tar')

    pct_max = 90.
    fail_count = fail_max = 5
    time = 0.
    pct0 = 100.
    for e in range(args.epochs):

        # Update the learning rate
        # schedule.step()

        time = train(e, time)

        loss, pct_err = test(e, time)
        if pct_err >= pct_max:
            fail_count -= 1

        torch.save({'ix': ix,
                    'epoch': e + 1,
                    'model': args.model,
                    'state_dict': model.state_dict(),
                    'pct_err': pct_err,
                    'loss': loss
                    }, save_model_path)
        if pct_err < pct0:
            shutil.copyfile(save_model_path, best_model_path)
            pct0 = pct_err

        if fail_count < 1:
            raise ValueError('Percent error has not decreased in %d epochs' % fail_max)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt; exiting')
