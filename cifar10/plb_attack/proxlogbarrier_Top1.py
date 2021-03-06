import argparse
import os, sys
import numpy as np
import pandas as pd
import pickle as pk
import torch
from torch import nn
from torch.autograd import grad
import time

from prox import L0NormProx_Batch, LinfNormProx, L2NormProx_Batch, L1NormProx

def Top1Criterion(x,y,model):
    return model(x).topk(1)[1].view(-1) == y

def CohenCriterion(x,y,model,std,num_samples=100):
    """ Do 100 noisy replications of x, take armax of output, then take the mode """

    xsh = x.shape
    x = x.unsqueeze(1)
    noise = torch.randn(xsh[0],num_samples,*xsh[1:]) * std
    if x.is_cuda:
        noise = noise.cuda()
    x = x + noise
    x = x.clamp(0,1)
    x = x.view(xsh[0]*num_samples,*xsh[1:])
    out = model(x)
    preds = out.argmax(dim=-1)
    preds = preds.view(xsh[0],num_samples)
    final_preds = preds.mode(dim=-1)[0]

    return final_preds == y

def LogRegCriterion(x,y,model,clf):
    """ The Logistic Regression Classification criterion

        x -> images (torch float tensor)
        y -> true labels (torch Long tensor)
        model -> pytorch model that maps images to feature vectors
        clf -> the trained logistic regression classifier that maps features to classes  """

    features, _ = model(x)
    out = clf(features)
    y_pred = out.argmax(dim=-1)

    return y_pred == y

def get_probs(x,model,clf):
    """ Get logit probabilities """

    features, _ = model(x)
    output = clf(features)
    probs = output.softmax(dim=-1)

    return probs

class Attack():

    def __init__(self,model,clf,criterion,norm=0,
                        verbose=True,**kwargs):

        super().__init__()
        self.model = model
        self.clf = clf
        #self.criterion = lambda x,y : criterion(x,y,model=model)
        self.criterion = criterion
        self.labels = None

        self.norm = norm
        self.verbose = verbose

        ##default parameters for MNIST, Fashion-MNIST, CIFAR10
        config = {'bounds':(0,1),
                    'dt' : 0.1,
                    'alpha' : 0.1,
                    'beta' : 0.75,
                    'gamma' : 0.05,
                    'max_outer' : 30,
                    'max_inner': 30,
                    'T': 1}

        config.update(kwargs)
        self.hyperparams = config

    def __call__(self,xorig,xpert,y):
        norm = self.norm
        config = self.hyperparams
        model=self.model
        clf = self.clf
        criterion=self.criterion

        bounds,dt,alpha0,beta,gamma,max_outer,max_inner,T = (
            config['bounds'], config['dt'], config['alpha'],
            config['beta'], config['gamma'], config['max_outer'],
            config['max_inner'], config['T'])

        Nb = len(y)
        ix = torch.arange(Nb,device=xorig.device)

        imshape = xorig.shape[1:]
        PerturbedImages = torch.full(xorig.shape,np.nan,device=xpert.device)

        #perturb only those that are correctly classified
        mis0 = criterion(xorig,y)
        xpert[~mis0] = xorig[~mis0]

        xold = xpert.clone()
        xbest = xpert.clone()
        diffBest = torch.full((Nb,),np.inf,device=xorig.device)

        #initial parameter calls
        dtz = dt*torch.ones(Nb).cuda()
        muz = T*torch.ones(Nb).cuda()

        xpert.requires_grad_(True)

        if norm == 0:
            proxFunc = L0NormProx_Batch()
        elif norm == 2:
            proxFunc = L2NormProx_Batch()
        elif norm == np.inf:
            proxFunc = LinfNormProx()
        elif norm == 1:
            proxFunc = L1NormProx()

        for k in range(max_outer):
            alpha = alpha0*beta**k

            diff = (xpert - xorig).view(Nb,-1).norm(self.norm,-1)
            update= diff>0

            for j in range(max_inner):
                p = get_probs(xpert,model,clf)
                pdiff = p.max(dim=-1)[0] - p[ix,y]
                s = -torch.log(pdiff).sum()
                g = grad(alpha*s,xpert)[0]

                with torch.no_grad():
                    if norm in [0,2]:
                        Nb_ = xpert[update].shape[0]
                        if Nb_ > 0:   # avoid error when Nb_ == 0
                            yPert = xpert[update].view(Nb_,-1) -dtz[update].view(-1,1) * g[update].view(Nb_,-1)
                            y_proxd = proxFunc(yPert,xorig[update].view(Nb_,-1),muz[update])
                            xpert[update] = y_proxd.view(Nb_,*imshape).clamp_(*bounds)
                    elif norm in [1,np.inf]:
                        Nb_ = xpert[update].shape[0]
                        yPert = xpert[update].view(Nb_,-1) -dtz[update].view(-1,1) * g[update].view(Nb_,-1)
                        y_proxd = proxFunc(yPert,xorig[update].view(Nb_,-1),T)
                        xpert[update] = y_proxd.view(Nb_,*imshape).clamp_(*bounds)

                with torch.no_grad():
                    c = criterion(xpert,y)
                    i = 0
                    while c.any():
                        ## backtracking into feasible region ##
                        xpert[c] = xpert[c].clone().mul(gamma).add(1-gamma,xold[c])
                        c = criterion(xpert,y)
                        i += 1
                        if i > 1000:
                            break
                    xpert[c] = xorig[c]  # make difference 0

                ## keep track of best iterate
                diff = (xpert - xorig).view(Nb,-1).norm(self.norm,-1)
                boolDiff = diff <= diffBest
                xbest[boolDiff] = xpert[boolDiff]
                diffBest[boolDiff] = diff[boolDiff]

                xold = xpert.clone()


                if self.verbose:
                    sys.stdout.write('  [%2d outer, %4d inner] median & max distance: (%4.4f, %4.4f)\r'
                         %(k, j, diffBest.median() , diffBest.max()))

        if self.verbose:
            sys.stdout.write('\n')

        switched = ~criterion(xbest,y)
        PerturbedImages[switched] = xbest.detach()[switched]

        return PerturbedImages




