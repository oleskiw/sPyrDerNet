import torch
import os, sys
from config import *
from torch.nn import Softmax


def covratio_loss(X,y):
    
    m = X.size(0)
    eps = 1e-4

    #subtract mean from total data matrix
    X_ms = X - X.mean(0)
    #compute total covariance
    Ct = (1.0/(m-1))*torch.mm(torch.t(X_ms), X_ms)
    
    #get unique large texture indices
    image_inds = torch.unique(y)

    #compute average covariance for patches coming from large textures
    def compute_cov(imind, X, y):
        Xi = X[y==imind]
        Xi_ms = Xi - Xi.mean(0)
        m = Xi_ms.size(0)
        return (1.0/(m-1))*torch.mm(torch.t(Xi_ms), Xi_ms)


    image_covs = torch.zeros(len(image_inds), X.size(1), X.size(1))
    for imind in image_inds:
        image_covs[imind] = compute_cov(imind, X,y)

    Cw = image_covs.mean(0)
    
    Cw = Cw + torch.eye(Cw.size(0)) * eps
    Cw = Cw.to(device)

    #solve gen eig problem using cholesky decomposition
    #solves Cy = lambda*y for C = L^-1 A L^-T ; y = L^T x

    #L = torch.potrf(Cw, upper = False)

    #C = torch.mm(torch.inverse(L), torch.mm(Ct, torch.inverse(L.t())))
    #U,S,V = torch.svd(C)
    U,S,V = torch.svd(torch.mm(torch.inverse(Cw), Ct))
    evals = torch.diag(torch.diag(S))
    print(evals)
    eiggaps = evals[0:-1]-evals[1:]
    softargmax = torch.dot(Softmax()(eiggaps*10),torch.arange(evals.size(0)-1).to(device))
    loss = -torch.mean(evals[evals<5])

    return loss








    

