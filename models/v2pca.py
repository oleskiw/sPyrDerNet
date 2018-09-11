from torch.tensor import *
from torch import nn


class V2PCA(nn.Module):

    def __init__(self, imgSize=64, K=4, N=2, includeHF=True, weights=None, components=None,
                 transferFunction='softAbs', ncomp=32):
        super(V2PCA, self).__init__()

        """
        V2PCA model
        Uses the steerable pyramid, followed by quadratic nonlinearity,
        hamming circular windowing, and PCA on the output coefficients
        """

        self.ncomp = ncomp
        self.transferFunction = transferFunction
        if components is not None:
            self.components = torch.Tensor(components[:, 0:self.ncomp])  # size = coefficients x components
        else:
            self.components = None

        if weights is not None:
            self.weights = torch.Tensor(weights)  # size = Mx1
        else:
            self.weights = None

    def forward(self, x):
        # transform to pyramid
        # pyr, pind = self.sPyr(x)
        # here we assume x is the pyramid coefficients

        # apply nonlinearity
        if self.transferFunction == 'softAbs':
            pyr_trans = Tensor.sqrt(x ** 2 + 0.001)
        else:
            raise NotImplementedError

        # apply weights to coefficients
        if self.weights is None:
            pyr_weighted = pyr_trans
        else:
            pyr_weighted = pyr_trans * Tensor.t(self.weights)

        # apply components to coefficients
        component_expression = Tensor.mm(pyr_weighted, self.components)

        return component_expression, pyr_weighted
