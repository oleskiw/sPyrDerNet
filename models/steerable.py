"""
Code taken from https://github.com/olivierhenaff/steerablePyramid.git
"""

import torch.nn as nn 
from .steerableUtils import *


class SteerablePyramid(nn.Module):

    def __init__(self, imgSize, K=4, N=4, hilb=False, includeHF=True ):
        super(SteerablePyramid, self).__init__()

        size = [ imgSize, imgSize//2 + 1 ]
        self.hl0 = HL0_matrix( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

        self.l = []
        self.b = []
        self.s = []

        self.K    = K 
        self.N    = N 
        self.hilb = hilb
        self.includeHF = includeHF 

        self.indF = [ freq_shift( size[0], True  ) ]
        self.indB = [ freq_shift( size[0], False ) ] 
        self.pyr = []
        self.pind = torch.zeros([K*N+2, 2])

        for n in range( self.N ):

            l = L_matrix_cropped( size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            b = B_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)
            s = S_matrix(      K, size ).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(device)

            self.l.append( l.div_(4) )
            self.b.append( b )
            self.s.append( s )

            size = [ l.size(-2), l.size(-1) ]

            self.indF.append( freq_shift( size[0], True  ) )
            self.indB.append( freq_shift( size[0], False ) )

    def forward(self, x):

        fftfull = torch.rfft(x,2)
        xreal = fftfull[... , 0]
        xim = fftfull[... ,1]
        x = torch.cat((xreal.unsqueeze(1), xim.unsqueeze(1)), 1 ).unsqueeze( -3 )
        x = torch.index_select( x, -2, self.indF[0] )
        x   = self.hl0 * x 
        h0f = x.select( -3, 0 ).unsqueeze( -3 )
        l0f = x.select( -3, 1 ).unsqueeze( -3 )
        lf  = l0f 

        output = []

        for n in range( self.N ):

            bf = self.b[n] *               lf 
            lf = self.l[n] * central_crop( lf ) 
            if self.hilb:
                hbf = self.s[n] * torch.cat( (bf.narrow(1,1,1), -bf.narrow(1,0,1)), 1 )
                bf  = torch.cat( ( bf , hbf ), -3 )
            if self.includeHF and n == 0:
                bf  = torch.cat( ( h0f,  bf ), -3 )

            output.append( bf )

        output.append( lf  ) 

        for n in range( len( output ) ):
            output[n] = torch.index_select( output[n], -2, self.indB[n] )
            sig_size = output[n].shape[-2]
            output[n] = torch.stack((output[n].select(1,0), output[n].select(1,1)),-1)
            output[n] = torch.irfft( output[n], 2, signal_sizes = [sig_size, sig_size])

        if self.includeHF:
            output.insert( 0, output[0].narrow( -3, 0, 1                    ) )
            output[1]       = output[1].narrow( -3, 1, output[1].size(-3)-1 )

        for n in range( len( output ) ):
            if self.hilb:
                if ((not self.includeHF) or 0 < n) and n < len(output)-1:
                    nfeat = output[n].size(-3)//2
                    o1 = output[n].narrow( -3,     0, nfeat ).unsqueeze(1)
                    o2 = output[n].narrow( -3, nfeat, nfeat ).unsqueeze(1)
                    output[n] = torch.cat( (o1, o2), 1 ) 
                else:
                    output[n] = output[n].unsqueeze(1)

        # convert outputs to similar form as in matlab code
        pind_ind = 0

        # swap row/column ordering
        for scale in range(len(output)):
            if scale == 0:
                self.pyr = output[0].permute(0,1,2,4,3).contiguous().view(output[0].size(0),-1)
            else:
                self.pyr = torch.cat((self.pyr, output[scale].permute(0,1,2,4,3).contiguous().view(output[scale].size(0),-1)), 1) 
            for i in range(output[scale].size(2)):
                self.pind[pind_ind]=torch.Tensor([output[scale].size(-2), output[scale].size(-1)])
                pind_ind += 1

        return self.pyr, self.pind
