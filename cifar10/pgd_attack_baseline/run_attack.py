""" Attack a ResNet18 model with PGD on CIFAR10 """

import argparse, yaml
import os, sys, pathlib
import ast

_pth = str(pathlib.Path(__file__).absolute())
for i in range(3):
    (_pth, _) = os.path.split(_pth)
sys.path.insert(0, _pth)  # I just made sure that the root of the project (ContrastiveTeamO) is in the path where Python
# looks for packages in order to import from files that require going several levels up from the directory where this
# script is. Unfortunately, by default Python doesn't allow imports from above the current file directory.


import numpy as np
import torch
from torch import nn

from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from attack_utils import pgd_attack, kl_div_loss, cw_loss

import cifar10.baseline_models.cifar as cifarmodels

# define the arguments
parser = argparse.ArgumentParser('Attack an example CIFAR10 example with L2PGD')

parser.add_argument('--data-dir', type=str, default='/home/math/oberman-lab/data/cifar10',
        metavar='DIR', help='Directory where data is saved')
parser.add_argument('--model-path', type=str,
                    default='/home/math/dragos.manta/contrastive/ContrastiveTeamO/cifar10/runs_baseline/runs/best'
                            '.pth.tar',
                    metavar='PATH',
                    help='path to the .pth.tar trained model file')
parser.add_argument('--dropout',type=float, default=0, metavar='P',
        help = 'Dropout probability, if model supports dropout (default: 0)')
parser.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm")
parser.add_argument('--no-bn',action='store_false', dest='bn',
        help = "Don't use batch norm")
parser.set_defaults(bn=True)
parser.add_argument('--last-layer-nonlinear',
        action='store_true', default=False)
parser.add_argument('--bias',action='store_true', dest='bias',
        help = "Use model biases")
parser.add_argument('--no-bias',action='store_false', dest='bias',
        help = "Don't use biases")
parser.set_defaults(bias=False)
parser.add_argument('--model-args',type=str,
        default="{}",metavar='ARGS',
        help='A dictionary of extra arguments passed to the model.'
        ' (default: "{}")')

parser.add_argument('--criterion', type=str, default='top1',
        help='given a model and x, how to we estimate y?')
parser.add_argument('--loss-function', type=str, default='KL',
        help='the loss function we will use in PGD (Choices = [KL,CW])')

parser.add_argument('--num-images', type=int, default=1000,metavar='N',
        help='total number of images to attack (default: 1000)')
parser.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time (default: 100) ')
parser.add_argument('--norm', type=str, default='L2',metavar='NORM',
        choices=['L2','Linf','L0','L1'],
        help='The dissimilarity metrics between images. (default: "L2")')

parser.add_argument('--seed', type=int, default=0,
        help='seed for RNG (default: 0)')
parser.add_argument('--random-subset', action='store_true',
        default=False, help='use random subset of test images (default: False)')

parser.add_argument('--eps', type=float, default=0.5,
        help='max. allowed perturbation')
parser.add_argument('--alpha', type=float, default=0.1,
        help='PGD step-size')
parser.add_argument('--iters', type=int, default=20,
        help='max. number of PGD iterations')

args = parser.parse_args()

torch.manual_seed(args.seed)

print('Arguments:')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

has_cuda = torch.cuda.is_available()

# make dataloaders
transform = transforms.Compose([transforms.ToTensor()])
ds = CIFAR10(args.data_dir, download=True, train=False, transform=transform)
if args.random_subset:
    Ix = np.random.choice(10000, size=args.num_images, replace=False)
    Ix = torch.from_numpy(Ix)
else:
    Ix = torch.arange(args.num_images) # Use the first N images of test set
subset = Subset(ds, Ix)
loader = torch.utils.data.DataLoader(
                    subset,
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=4, pin_memory=has_cuda)

# retrieve pretrained model
classes = 10
model_args = ast.literal_eval(args.model_args)
in_channels = 3
model_args.update(bn=args.bn, classes=classes, bias=args.bias,
                  kernel_size=3,
                  in_channels=in_channels,
                  softmax=False,last_layer_nonlinear=args.last_layer_nonlinear,
                  dropout=args.dropout)
model = getattr(cifarmodels, 'ResNet50')(**model_args)
if has_cuda:
    model = model.cuda()
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
savedict = torch.load(args.model_path, map_location='cpu')
model.load_state_dict(savedict['state_dict'])
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# define loss function
if args.loss_function == 'KL':
    loss_fn = kl_div_loss
elif args.loss_function == 'CW':
    loss_fn = cw_loss
else:
    raise ValueError("select a loss function")

# Now run the attack!
d0 = torch.full((args.num_images,),np.inf)
d2 = torch.full((args.num_images,),np.inf)
dinf = torch.full((args.num_images,),np.inf)
d1 = torch.full((args.num_images,),np.inf)

if has_cuda:
    d0 = d0.cuda()
    d2 = d2.cuda()
    dinf = dinf.cuda()
    d1 = d1.cuda()

K=0
for i, (x, y) in enumerate(loader):
    sys.stdout.write('Batch %2d/%d:\r'%(i+1,len(loader)))

    Nb = x.shape[0]

    if has_cuda:
        x = x.cuda()
        y = y.cuda()

    # perform the attack on the batch "x"
    diff = pgd_attack(model, x, y, loss_fn=loss_fn, eps=args.eps, iters=args.iters, alpha=args.alpha)

    l0 = diff.view(Nb, -1).norm(p=0, dim=-1)
    l2 = diff.view(Nb, -1).norm(p=2, dim=-1)
    linf = diff.view(Nb, -1).norm(p=np.inf, dim=-1)
    l1 = diff.view(Nb,-1).norm(p=1,dim=-1)

    ix = torch.arange(K,K+Nb, device=x.device)

    d0[ix] = l0
    d2[ix] = l2
    dinf[ix] = linf
    d1[ix] = l1

    K+=Nb

md = d2.median()
me = d2.mean()
mx = d2.max()

print('\nDone. Statistics:')
print('  Median adversarial distance: %.3g'%md)
print('  Mean adversarial distance:   %.3g'%me)
print('  Max adversarial distance:    %.3g'%mx)

i = 0
while os.path.exists('attack%s'%i):
    i +=1
pth = os.path.join('./','attack%s/'%i)
os.makedirs(pth, exist_ok=True)

args_file_path = os.path.join(pth, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

st = 'pgd_dists'
dists = {'index':Ix.cpu().numpy(),
         'l0':d0.cpu().numpy(),
         'l2':d2.cpu().numpy(),
         'linf':dinf.cpu().numpy(),
         'l1': d1.cpu().numpy()}

with open(os.path.join(pth, st+'.npz'), 'wb') as f:
    np.savez(f, **dists)

