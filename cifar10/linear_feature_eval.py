import argparse
import os, shutil, sys
import yaml
import numpy as np
import random
import ast, bisect
import time, datetime
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
#from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader

import models.cifar as cifarmodels
from ContrastiveTeamO.loss.nt_xent import NTXentLoss

from ContrastiveTeamO.custom_transforms import get_color_distortion, GaussianBlur

from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import importlib.util

parser = argparse.ArgumentParser('Use logistic regression to classify a contrastive learning model')
parser.add_argument('--data-dir', type=str, default='/home/campus/oberman-lab/data/',metavar='DIR',
        help='Directory where CIFAR-10 data is saved')
parser.add_argument('--seed', type=int, default=0, metavar='S',
        help='random seed (default: 0)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
        help='Input batch size for training. (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
        help='input batch size for testing (default: 100)')
parser.add_argument('--num-train-images', type=int, default=10000, metavar='NI',
        help='number of test images to classify (default=10000)')
parser.add_argument('--num-test-images', type=int, default=10000, metavar='NI',
        help='number of test images to classify (default=10000)')
parser.add_argument('--random-subset', action='store_true',
        default=False, help='use random subset of test images (default: False)')

group1 = parser.add_argument_group('Model hyperparameters')
group1.add_argument('--model', type=str, default='ResNet50',
        help='Model architecture (default: ResNet50)')
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

args = parser.parse_args()

# CUDA info
has_cuda = torch.cuda.is_available()
cudnn.benchmark = True
#has_cuda = False

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Print args
print('Contrastive Learning Classification on Cifar-10')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

# make dataloaders
root = os.path.join(args.data_dir,'cifar10')

ds_test = CIFAR10(root, download=True, train=False, transform=transforms.Compose([transforms.ToTensor()]))
if args.num_test_images > 10000:
    args.num_test_images = 10000
if args.random_subset:
    Ix = np.random.choice(10000, size=args.num_test_images, replace=False)
    Ix = torch.from_numpy(Ix)
else:
    Ix = torch.arange(args.num_test_images) # Use the first N images of test set
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
    Ix = torch.arange(args.num_train_images) # Use the first N images of test set
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
                  softmax=False,last_layer_nonlinear=args.last_layer_nonlinear,
                  dropout=args.dropout)
model = getattr(cifarmodels, args.model)(**model_args)
if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)

savedict = torch.load('./runs/encoder_best.pth.tar', map_location='cpu')
model.load_state_dict(savedict['state_dict'])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
#if has_cuda:
#    model = model.cuda()
#    if torch.cuda.device_count()>1:
#        model = nn.DataParallel(model)

# Get model output dim
for i, (x,_) in enumerate(train_loader):
    if has_cuda:
        x = x.cuda()
    out = model(x)[0]
    break
out_dim = out.shape[1]

# Get training and test features using our trained encoder model
train_features = torch.zeros(num_train,out_dim)
y_train = torch.zeros(num_train)
K = 0
print("Gathering training set features.")
for i, (x,y) in enumerate(train_loader):
    sys.stdout.write('    Batch %2d/%d:\r'%(i+1,len(train_loader)))
    if has_cuda:
        x = x.cuda()
    Nb = x.shape[0]
    out = model(x)[0]
    train_features[K:(K+Nb)] = out.cpu()
    y_train[K:(K+Nb)] = y
    K += Nb

test_features = torch.zeros(num_test,out_dim)
y_test = torch.zeros(num_test)
if has_cuda:
    test_features = test_features.cuda()
K = 0
print("Gathering test set features.")
for i, (x,y) in enumerate(test_loader):
    sys.stdout.write('    Batch %2d/%d:\r'%(i+1,len(test_loader)))
    if has_cuda:
        x = x.cuda()
    Nb = x.shape[0]
    out = model(x)[0]
    test_features[K:(K+Nb)] = out.cpu()
    y_test[K:(K+Nb)] = y
    K += Nb

# convert to np arrays
train_features, test_features = train_features.cpu().numpy(), test_features.cpu().numpy()
y_train, y_test = y_train.cpu().numpy(), y_test.cpu().numpy()

# perform fit
print("Fitting the logistic regression classifier...")
scaler = preprocessing.StandardScaler()
scaler.fit(train_features)
clf = LogisticRegression(random_state=0, max_iter=1200, solver='lbfgs', C=1.0)
clf.fit(scaler.transform(train_features), y_train)

# Test the classifier...
print("Logistic Regression feature eval")
print("Train score:", clf.score(scaler.transform(train_features), y_train))
print("Test score:", clf.score(scaler.transform(test_features), y_test))

# free up space on the CPU
del train_features, y_train, test_features, y_test
