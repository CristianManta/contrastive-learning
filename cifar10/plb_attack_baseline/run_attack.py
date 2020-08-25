import argparse, yaml
import os, sys, pathlib
import ast, bisect

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

from InitMethods import GaussianInitialize, UniformInitialize, SafetyInitialize
from proxlogbarrier_Top1 import Attack, LogRegCriterion

import cifar10.baseline_models.cifar as cifarmodels

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn import preprocessing
#import importlib.util

parser = argparse.ArgumentParser('Attack an example CIFAR-10 encoder model with the ProxLogBarrier attack.'
                                  'Writes adversarial distances (and optionally images) to a npz file.')

groups0 = parser.add_argument_group('Required arguments')
groups0.add_argument('--data-dir', type=str, default='/home/math/oberman-lab/data/cifar10',
        help='Directory where CIFAR10 data is saved')

parser.add_argument('--model-path', type=str,
                    default='/home/math/dragos.manta/contrastive/ContrastiveTeamO/cifar10/runs_baseline_tik/runs/best'
                            '.pth.tar',
                    metavar='PATH',
                    help='path to the .pth.tar trained model file')

groups0.add_argument('--parallel', action='store_true', dest='parallel',
        help='only allow exact matches to model keys during loading')
groups0.add_argument('--strict', action='store_true', dest='strict',
        help='only allow exact matches to model keys during loading')
groups0.add_argument('--criterion', type=str, default='logistic',
        help='given a model and x, how to we estimate y?, choices=[top1,logistic]')
groups0.add_argument('--dropout',type=float, default=0, metavar='P',
        help = 'Dropout probability, if model supports dropout (default: 0)')
groups0.add_argument('--bn',action='store_true', dest='bn',
        help = "Use batch norm")
groups0.add_argument('--no-bn',action='store_false', dest='bn',
       help = "Don't use batch norm")
groups0.set_defaults(bn=True)
groups0.add_argument('--last-layer-nonlinear',
        action='store_true', default=False)
groups0.add_argument('--bias',action='store_true', dest='bias',
        help = "Use model biases")
groups0.add_argument('--no-bias',action='store_false', dest='bias',
       help = "Don't use biases")
groups0.set_defaults(bias=False)
groups0.add_argument('--model-args',type=str,
        default="{}",metavar='ARGS',
        help='A dictionary of extra arguments passed to the model.'
        ' (default: "{}")')

groups2 = parser.add_argument_group('Optional attack arguments')
groups2.add_argument('--num-images', type=int, default=1000,metavar='N',
        help='total number of images to attack (default: 1000)')
#groups2.add_argument('--num-train-images', type=int, default=5000,metavar='N',
#        help='total number of images to train the Logistic Regression classifier (default: 5000)')
groups2.add_argument('--batch-size', type=int, default=100,metavar='N',
        help='number of images to attack at a time (default: 100) ')
groups2.add_argument('--save-images', action='store_true', default=False,
        help='save perturbed images to a npy file (default: False)')
groups2.add_argument('--norm', type=str, default='L2',metavar='NORM',
        choices=['L2','Linf','L0','L1'],
        help='The dissimilarity metrics between images. (default: "L2")')
groups2.add_argument('--init-type',type=str,default='gaussian',
        choices=['gaussian','uniform'])
groups2.add_argument('--seed', type=int, default=0,
        help='seed for RNG (default: 0)')
groups2.add_argument('--random-subset', action='store_true',
        default=False, help='use random subset of test images (default: False)')

group1 = parser.add_argument_group('Attack hyperparameters')
group1.add_argument('--dt', type=float, default=0.1, help='step size (default: 0.1)')
group1.add_argument('--alpha', type=float, default=0.1,
        help='initial Lagrange multiplier of log barrier penalty (default: 0.1)')
group1.add_argument('--beta', type=float, default=0.75,
        help='shrink parameter of Lagrange multiplier after each inner loop (default: 0.75)')
group1.add_argument('--gamma', type=float, default=0.05,
        help='back track parameter (default: 0.05)')
group1.add_argument('--max-outer', type=int, default=30,
        help='maximum number of outer loops (default: 30)')
group1.add_argument('--max-inner', type=int, default=30,
        help='max inner loop iterations (default: 30)')
group1.add_argument('--T', type=float, default=1, help='prox parameter')

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Print args
print('Contrastive Learning Classification on Cifar-10')
for p in vars(args).items():
    print('  ',p[0]+': ',p[1])
print('\n')

has_cuda = torch.cuda.is_available()

# Data loading code
transform = transforms.Compose([transforms.ToTensor()])
ds = CIFAR10(os.path.join(args.data_dir), download=True, train=False, transform=transform)

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
loaderOneByOne = torch.utils.data.DataLoader(
                    subset, batch_size=1,shuffle=False,
                    num_workers=4, pin_memory=has_cuda)

############################

# Retrieve pre trained Contrastive loss model and the logistic regression classifier
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

############################

#change criterion for ImageNet-1k and CIFAR100 to Top5Criterion
#if args.criterion=='cohen':
#    criterion = lambda x, y: CohenCriterion(x,y,model,std=0.05)
#elif args.criterion=='top1':
#    criterion = lambda x, y: Top1Criterion(x,y,model)
#elif args.criterion=='logistic':
criterion = lambda x, y: LogRegCriterion(x,y,model=model)
#else:
#    raise ValueError('Select a valid predection criterion')

if args.norm=='L2':
    norm = 2
elif args.norm=='Linf':
    norm = np.inf
elif args.norm == 'L0':
    norm = 0
elif args.norm == 'L1':
    norm = 1

if has_cuda:
    model = model.cuda()

params = {'bounds':(0,1),
          'dt':args.dt,
          'alpha':args.alpha,
          'beta':args.beta,
          'gamma':args.gamma,
          'max_outer':args.max_outer,
          'max_inner':args.max_inner,
          'T': args.T}

attack = Attack(model=model, norm=norm, criterion=criterion, **params)

d0 = torch.full((args.num_images,),np.inf)
d2 = torch.full((args.num_images,),np.inf)
dinf = torch.full((args.num_images,),np.inf)
d1 = torch.full((args.num_images,),np.inf)

if has_cuda:
    d0 = d0.cuda()
    d2 = d2.cuda()
    dinf = dinf.cuda()
    d1 = d1.cuda()

if args.save_images:
    chan, height, width = 1,28,28 #modify for other datasets
    PerturbedImages = torch.full((args.num_images, chan,height,width), np.nan)
    labels = torch.full((args.num_images,),-1, dtype=torch.long)
    if has_cuda:
        PerturbedImages = PerturbedImages.cuda()
        labels = labels.cuda()

if args.init_type == 'uniform':
    init_attack = UniformInitialize(model=model)
elif args.init_type == 'gaussian':
    init_attack = GaussianInitialize(model=model)

K = 0
## make safety list of images
ImsList = torch.zeros(10,3,32,32).cuda() ##change based on img size
ysList = []

for i, (x,y) in enumerate(loaderOneByOne):
    nclasses = 0
    x,y = x.cuda(), y.cuda()
    evalcrit=criterion(x,y)
    if evalcrit:
        if y in ysList:
            continue
        else:
            ysList.append(y)
            ImsList[y.cpu().item()] = x
            nclasses += 1

    if nclasses == 10:
        break

#print(model(ImsList))
safety = SafetyInitialize(model=model,TrainingIms=ImsList)

for i, (x, y) in enumerate(loader):
    print('Batch %2d/%d:'%(i+1,len(loader)))

    Nb = len(y)
    if has_cuda:
        x, y = x.cuda(), y.cuda()

    xstart = init_attack(x,y,criterion,max_iters=5000)
    xstartNew = safety(xstart,x,y)

    xpert = attack(x,xstartNew,y)

    diff = x - xpert.detach()
    l0 = diff.view(Nb, -1).norm(p=0, dim=-1)
    l2 = diff.view(Nb, -1).norm(p=2, dim=-1)
    linf = diff.view(Nb, -1).norm(p=np.inf, dim=-1)
    l1 = diff.view(Nb,-1).norm(p=1,dim=-1)

    ix = torch.arange(K,K+Nb, device=x.device)

    if args.save_images:
        PerturbedImages[ix] = xpert
        labels[ix] = y
    d0[ix] = l0
    d2[ix] = l2
    dinf[ix] = linf
    d1[ix] = l1

    K+=Nb

if args.norm=='L2':
    md = d2.median()
    mx = d2.max()
elif args.norm=='Linf':
    md = dinf.median()
    mx = dinf.max()
elif args.norm == 'L0':
    md = d0.median()
    mx = d0.max()
elif args.norm == 'L1':
    md = d1.median()
    mx = d1.max()

print('\nDone. Statistics in %s norm:'%args.norm)
print('  Median adversarial distance: %.3g'%md)
print('  Max adversarial distance:    %.3g'%mx)

st = 'proxlogbarrier-'+args.norm

dists = {'index':Ix.cpu().numpy(),
         'l0':d0.cpu().numpy(),
         'l2':d2.cpu().numpy(),
         'linf':dinf.cpu().numpy(),
         'l1': d1.cpu().numpy()}

i = 0
while os.path.exists('attack%s'%i):
    i +=1
pth = os.path.join('./','attack%s/'%i)
os.makedirs(pth, exist_ok=True)

args_file_path = os.path.join(pth, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

if args.save_images:
    dists['perturbed'] = PerturbedImages.cpu().numpy()
    dists['labels'] = labels.cpu().numpy()

with open(os.path.join(pth, st+'.npz'), 'wb') as f:
    np.savez(f, **dists)

