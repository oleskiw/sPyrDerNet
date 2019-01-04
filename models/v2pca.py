import torch
import torch.nn as nn
from .steerable import *
from .genutils import *

class V2PCA(nn.Module):


    def __init__(self, imgSize, K=4, N=2, includeHF=True, nonlin='smoothabs', pcaMat = None, window = None, ret_pyr = False, ncomp=32):
        super(V2PCA, self).__init__()

        '''
        V2PCA model
        Uses the steerable pyramid, followed by quadratic nonlinearity,
        hamming circular windowing, and PCA on the output coefficients
        
        args:
            imgSize: input image size
            K: number of steerable pyramid orientations
            N: number of steerable pyramid levels
            includeHF: include high frequency residual in vector output
            nonlin: type of nonlinearity after steerable pyramid
            pcaMat: if you already have PCA computed, can use the matrix to compute
            projections of the coefficients onto the PCA components
            window: vector of size Mx1 (where M is the number of steerable coefficients) - elementwise product with coefficients to apply window
            ncomp: number of PCA components to keep
        '''
        self.sPyr = SteerablePyramid(imgSize=imgSize, K=4, N=2, includeHF=True)
        self.ncomp = ncomp
        self.nonlin = nonlin
        self.ret_pyr = ret_pyr
        if pcaMat is not None:
            self.pcaMat = pcaMat[:,0:self.ncomp-1] #size = JxM where J is number of components and M is number of coefficients after steerably pyramid and windowing
        else:
            self.pcaMat = pcaMat
        self.window = window #size = Mx1

    def forward(self, x):
        pyr, pind = self.sPyr(x)
        if self.nonlin == 'smoothabs':
            trans_coeff = smoothabs_activ(pyr)
        else:
            raise NotImplementedError
        
        batchsize = pyr.size(0)
        if self.window is None:
            coeff_out = trans_coeff
        else:
            self.window = self.window.expand(self.window.size(0), batchsize)
            coeff_out = trans_coeff*torch.t(self.window)
        if self.pcaMat is None:
            coeff_out = coeff_out
        else:
            coeff_out = torch.mm(coeff_out, self.pcaMat)
        if self.ret_pyr:
            return coeff_out, pyr, pind
        else:
            return coeff_out
