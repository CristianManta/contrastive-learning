from datetime import datetime

now_unformatted = datetime.now()
time_string = now_unformatted.strftime("%b-%d-%Y_%H-%M-%S")

import random
import time
import os, shutil, sys, pathlib
import yaml
import ast, bisect
import csv

import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
import torchnet as tnt

import argparse

# put root of project in sys.path
_pth = str(pathlib.Path(__file__).absolute())
for i in range(2):
    (_pth, _) = os.path.split(_pth)
sys.path.insert(0, _pth)

import dataloader
from dataloader import cutout
import baseline_models as models
import baseline_models.cifar as cifarmodels

# -------------
# Initial setup
# -------------

# Parse command line arguments
parser = argparse.ArgumentParser('Training template for DNN computer vision research in PyTorch')
parser.add_argument('--datadir', type=str, default='/home/math/oberman-lab/data/cifar10', metavar='DIR',
                    help='data storage directory')
parser.add_argument('--dataset', type=str, help='dataset (default: "cifar10")',
                    default='cifar10', metavar='DS',
                    choices=['cifar10', 'cifar100', 'TinyImageNet', 'Fashion', 'mnist'])
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--logdir', type=str, default=None, metavar='DIR',
                    help='directory for outputting log files. (default: ./logs/TIMESTAMP/)')
parser.add_argument('--seed', type=int, default=None, metavar='S',
                    help='random seed (default: int(time.time()) )')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='input batch size for testing (default: 100)')

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
random.seed(args.seed)
np.random.seed(args.seed)

# Set and create logging directory
if args.logdir is None:
    args.logdir = os.path.join('./logs/', time_string)

os.makedirs(args.logdir, exist_ok=True)

if args.dataset in ['mnist', 'Fashion']:
    args.greyscale = True

# Print arguments to std out
# and save argument values to yaml file,
# so we know exactly how this experiment ran,
# and so we can re-load the model later
print('Arguments:')
for p in vars(args).items():
    print('  ', p[0] + ': ', p[1])
print('\n')

args_file_path = os.path.join(args.logdir, 'args.yaml')
with open(args_file_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)

shutil.copyfile('./train_baseline_exact.py', os.path.join(args.logdir, 'train_baseline_exact.py'))

# Data loaders
workers = 4

if args.num_test_images > 10000:
    args.num_test_images = 10000
if args.random_subset:
    Ix = np.random.choice(10000, size=args.num_test_images, replace=False)
else:
    Ix = np.arange(args.num_test_images)  # Use the first N images of test set

test_loader = getattr(dataloader, args.dataset)(args.datadir,
                                                mode='test', transform=False,
                                                batch_size=args.test_batch_size,
                                                greyscale=args.greyscale,
                                                num_workers=workers,
                                                shuffle=False,
                                                pin_memory=has_cuda,
                                                subset=Ix)

image_shape = test_loader.image_shape
transforms = [cutout(args.cutout, channels=image_shape[0])]

if args.num_train_images > 50000:
    args.num_train_images = 50000
if args.random_subset:
    Ix = np.random.choice(50000, size=args.num_train_images, replace=False)
else:
    Ix = np.arange(args.num_train_images)  # Use the first N images of train set

train_loader = getattr(dataloader, args.dataset)(args.datadir,
                                                 mode='train', transform=True,
                                                 greyscale=args.greyscale,
                                                 batch_size=args.batch_size,
                                                 training_transforms=transforms,
                                                 num_workers=workers,
                                                 shuffle=True,
                                                 pin_memory=has_cuda,
                                                 drop_last=True,
                                                 subset=Ix)

# Initialize model
classes = train_loader.classes
model_args = ast.literal_eval(args.model_args)
in_channels = 3 if not args.greyscale else 1
model_args.update(bn=args.bn, classes=classes, bias=args.bias,
                  kernel_size=args.kernel_size,
                  in_channels=in_channels,
                  softmax=False, last_layer_nonlinear=args.last_layer_nonlinear,
                  dropout=args.dropout)
if args.dataset in ['cifar10', 'cifar100', 'Fashion']:
    model = getattr(cifarmodels, args.model)(**model_args)
elif args.dataset == 'TinyImageNet':
    model = getattr(models.tinyimagenet, args.model)(**model_args)
elif args.dataset == 'mnist':
    model = getattr(models.mnist, args.model)(**model_args)

# Loss function and regularizers
criterion = nn.CrossEntropyLoss()
train_criterion = nn.CrossEntropyLoss(reduction='none')

# Move to GPU if available
if has_cuda:
    criterion = criterion.cuda(0)
    train_criterion = train_criterion.cuda(0)
    model = model.cuda(0)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

# -------- If fine tuning --------
# svdict = torch.load('./logs/cifar10/ResNet50/Aug-19-2020_13-48-11/best.pth.tar', map_location='cpu')
# model.load_state_dict(svdict['state_dict'])

# --------------------------------


# ------------------------------------
# Optimizer and learning rate schedule
# ------------------------------------
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


def scheduler(optimizer, args):
    """Return a hyperparmeter scheduler for the optimizer"""
    lS = np.array(ast.literal_eval(args.lr_schedule))
    llam = lambda e: float(lS[max(bisect.bisect_right(lS[:, 0], e) - 1, 0), 1])
    lscheduler = LambdaLR(optimizer, llam)

    return lscheduler


schedule = scheduler(optimizer, args)

# --------
# Training
# --------
decay = args.decay  # penalize by the sum of parameters squared

trainlog = os.path.join(args.logdir, 'training.csv')
traincolumns = ['index', 'time', 'loss']
with open(trainlog, 'w') as f:
    logger = csv.DictWriter(f, traincolumns)
    logger.writeheader()

ix = 0  # count of gradient steps

tik = args.penalty
regularizing = tik > 0
h = args.h


def train(epoch, ttot):
    global ix

    # Put the model in train mode (turn on dropout, unfreeze
    # batch norm parameters)
    model.train()

    # Run through the training data
    if has_cuda:
        torch.cuda.synchronize()
    tepoch = time.perf_counter()

    with open(trainlog, 'a') as f:
        logger = csv.DictWriter(f, traincolumns)

        print("Current LR: {}".format(schedule.get_lr()[0]))

        for batch_ix, (data, target) in enumerate(train_loader):

            if has_cuda:
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()

            if regularizing:
                data.requires_grad_(True)

            output = model(data)
            lx = train_criterion(output, target)
            loss = lx.mean()

            # Compute finite difference approximation of directional derivative of grad loss wrt inputs
            if regularizing:

                dx = grad(loss, data, retain_graph=True, create_graph=True)[0]
                sh = dx.shape
                data.requires_grad_(False)

                # v is the finite difference direction.
                # For example, if norm=='L2', v is the gradient of the loss wrt inputs
                v = dx.view(sh[0], -1)
                Nb, Nd = v.shape

                if args.norm == 'L2':
                    nv = v.norm(2, dim=-1)
                else:
                    raise ValueError("Norms other than L2 are not yet implemented yet.")

            tik_penalty = torch.tensor(np.nan)
            dlmean = torch.tensor(np.nan)
            dlmax = torch.tensor(np.nan)
            if regularizing:
                nv2 = nv.pow(2)
                tik_penalty = nv2.mean() / 2
                # print(f"loss before = {loss}")
                loss = loss + tik * tik_penalty
                # print(f"loss after = {loss}")
                # print(f"tik_penalty = {tik_penalty}")
                # print("\n")

            loss.backward()
            optimizer.step()

            if np.isnan(loss.data.item()):
                raise ValueError('model returned nan during training')

            t = ttot + time.perf_counter() - tepoch
            fmt = '{:.4f}'
            logger.writerow({'index': ix,
                             'time': fmt.format(t),
                             'loss': fmt.format(loss.item())})

            if (batch_ix % args.log_interval == 0 and batch_ix > 0):
                print('[Epoch %2d, batch %3d] penalized training loss: %.3g' %
                      (epoch, batch_ix, loss.data.item()))
            ix += 1

    if has_cuda:
        torch.cuda.synchronize()

    return ttot + time.perf_counter() - tepoch


# ------------------
# Evaluate test data
# ------------------
testlog = os.path.join(args.logdir, 'test.csv')
testcolumns = ['epoch', 'time', 'fval', 'pct_err', 'train_fval', 'train_pct_err']
with open(testlog, 'w') as f:
    logger = csv.DictWriter(f, testcolumns)
    logger.writeheader()


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
    # for n, p in model.named_parameters():
    #    if 'weight' in n:
    #        w = p.view(p.size(0),-1)
    #        print('%s inf norm: %.2f'%(n, w.norm(1,-1).max()))
    # print('\n\n')

    return test_loss.value()[0], top1.value()[0]


# -------------------------------
# Now cook for 2 hours at 350 F
# -------------------------------
def main():
    save_model_path = os.path.join(args.logdir, 'checkpoint.pth.tar')
    best_model_path = os.path.join(args.logdir, 'best.pth.tar')

    pct_max = 100. * (1 - 1.0 / classes)
    fail_max = 5
    fail_count = fail_max
    time = 0.
    pct0 = 100.
    for e in range(args.epochs):
        time = train(e, time)
        schedule.step()  # Should be called before optimizer.step()

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
    best_acc = 100 - pct0
    with open(os.path.join(args.logdir, 'accuracy.txt'), 'w') as f:
        msg = "Best accuracy on the test set: " + str(best_acc) + "%"
        f.write(msg)

    print(f"Best accuracy on the test set: {best_acc}")

    # if fail_count < 1:
    #     raise ValueError('Percent error has not decreased in %d epochs' % fail_max)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt; exiting')
