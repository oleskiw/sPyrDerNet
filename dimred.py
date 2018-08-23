import torch
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import fastica
from utils import *
import math

"""
X_mean = torch.mean(X, 1, True)
X = X - X_mean.expand_as(X)
cov = torch.mm(torch.t(X), X)/ (X.shape[0] -1)
printout("Computing svd...")
cov = cov.data.cpu().numpy()
U,S,V = randomized_svd(cov, 32)
#U,S,V = torch.svd(cov)
printout("Computing zcamat ...")
torch.cuda.empty_cache()
V=torch.tensor(V).to(device)
S=torch.tensor(S).to(device)
Smat = torch.diag(1/torch.sqrt(S+1e-5))
zcamat = torch.mm(torch.mm(torch.t(V), Smat), V)
return zcamat[:,:n_components]
"""

def ZCA(X, n_components=32):
    X_mean = torch.mean(X, 1, True)
    X = X - X_mean.expand_as(X)
    printout("Computing svd for zca...")
    U,S,V = torch.svd(X)
    Smat = torch.diag(S)
    PC = torch.mm(torch.mm(V, torch.sqrt(1/(Smat**2 + 1e-4))), torch.t(V)[:,:n_components])
    return PC

def ICA(X, n_components = 32):
    X = X.data.cpu().numpy()
    K = fastica(X,n_components, whiten=True, compute_sources=False)[0]
    return K

def PCA(X, n_components = 32):
   X_mean = torch.mean(X, 1, True)
   X = X - X_mean.expand_as(X)
   printout("Computing svd for pca...")
   U,S,V = torch.svd(X)
   Smat = torch.diag(S)
   PC = torch.mm(V, Smat[:, :n_components]/math.sqrt(X.size()[0]-1))
   return PC
