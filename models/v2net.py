import torch.nn as nn
from .steerable import *
from .genutils import *
from config import *

class V2Net(nn.Module):


    def __init__(self, imgSize, K=4, N=2, includeHF=True, inp = 'image', nonlin='smoothabs', window = None, ret_pyr = False, ncomp = 10, out_activ = False):
        super(V2Net, self).__init__()

        '''
        V2Network model
        Uses the steerable pyramid, followed by quadratic nonlinearity,
        hamming circular windowing, and a linear layer down to ncomp number of hidden units
        
        args:
            imgSize: input image size
            K: number of steerable pyramid orientations
            N: number of steerable pyramid levels
            includeHF: include high frequency residual in vector output
            nonlin: type of nonlinearity after steerable pyramid
            window: vector of size Mx1 (where M is the number of steerable coefficients) - elementwise product with coefficients to apply window
            ncomp: number of output components in the model
        '''


        self.ncomp = ncomp
        self.nonlin = nonlin
        self.ret_pyr = ret_pyr
        self.pyr = None
        self.pind = None
        self.out_activ = out_activ
        self.inp = inp

        if self.inp == 'image':
            self.sPyr = SteerablePyramid(imgSize=imgSize, K=4, N=2, includeHF=True)
            test = self.sPyr(torch.randn(1,1,imgSize,imgSize, device=device))
            self.window = window #size = Mx1
            self.lintrans = nn.Linear(test[0].size(1), ncomp, bias = False)
            del test
        elif self.inp == 'coeff':
            self.window = window
            self.lintrans = nn.Linear(self.window.nonzero().size(0), ncomp, bias = False)



    def forward(self, x):
        if self.inp == 'image':
            self.pyr, self.pind = self.sPyr(x)
            if self.nonlin == 'smoothabs':
                out = smoothabs_activ(self.pyr)
            else:
                raise NotImplementedError
        
            batchsize = self.pyr.size(0)
            if self.window is not None:
                self.window = self.window.expand(self.window.size(0), batchsize)
                out = out*torch.t(self.window)
        elif self.inp == 'coeff':
            out = x

        out = self.lintrans(out)

        if self.out_activ:
            out = smoothabs_activ(out)

        return out
        
        
        
