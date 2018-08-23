import torch
import numpy as np
from sklearn.utils.extmath import randomized_svd
from sklearn.decomposition import fastica
from utils import *
import math


def ZCA(X, n_components=32):
    X_mean = torch.mean(X, 1, True)
    X = X - X_mean.expand_as(X)
    printout("Computing svd for zca...")
    U,S,V = torch.svd(X)
    Smat = torch.diag(S)
    PC = torch.mm(torch.mm(V, torch.sqrt(1/(Smat**2 + 1e-4))), torch.t(V)[:,:n_components])
    return PC
    '''
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
    '''
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


def compute_componentmat(savepath = './output/', dataset = 'mat', dimreduce='pca', use_window = True, num_textures=10, max_patches = 800):
    '''
    can choose between pca, ica, or zca for dimreduce option
    returns matrix of size num_coefficients (output of steerable pyramid and rectification) x num_components
    '''
    if dataset == 'mat':
       impatch, im_inds,matdata = matfile_dataload(precomp = False, num_textures = 10, max_patches = 800)  
    elif dataset == 'lcv':
        matdata = matfile_dataload(rawdat = True)
        impatch, im_inds = h5py_dataload(num_textures=num_textures, max_patches=max_patches)

    if use_window:
        window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
        windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
        windowinds = np.nonzero(windowbool)[0]
        window = torch.tensor(window, dtype=dtype).to(device)
    else:
        window = None
    del matdata
    

    printout("Data loaded, impatch shape:" +  str(impatch.shape))
    printout("Building pyramid...")
    network = V2PCA(imgSize=impatch.shape[1], K=4, N=2, nonlin='smoothabs', window = None, pcaMat = None, ret_pyr = False, ncomp=32)
    x = impatch.reshape([impatch.shape[0], 1, impatch.shape[1], impatch.shape[2]])
    x = torch.tensor(x, dtype=dtype).to(device)
    coeff = network(x)
    if use_window:
        coeff = coeff[:,windowinds]
    printout("Coeffs generated...")

    del x
    torch.cuda.empty_cache()
    printout("Computing dimreduce for " + dimreduce)
    if dimreduce =='zca':
        compmat = ZCA(coeff)
        compmat = compmat.data.cpu().numpy()
    elif dimreduce == 'ica':
        compmat = ICA(coeff)
    elif dimreduce == 'pca':
        compmat = PCA(coeff)
        compmat = compmat.data.cpu().numpy()
    if use_window:
        compmatold = compmat
        compmat = np.zeros((window.size(0), compmatold.shape[1]))+1e-8
        compmat[windowinds,:] = compmatold

    printout(compmat.shape)
    h5f = h5py.File(savepath + dimreduce + 'mat.h5', 'w')
    h5f.create_dataset('dat', data=compmat, compression='gzip')
    h5f.close()

def main():
    compute_componentmat(dimreduce='pca', use_window = True, num_textures = 10, max_patches = 800)



if __name__== "__main__":
    main()
    
