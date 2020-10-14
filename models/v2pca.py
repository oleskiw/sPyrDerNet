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
            pyr_trans = Tensor.sqrt(x ** 2 + 0.000001)
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

    def window_power(self, x):
        # apply weights to coefficients
        if self.weights is None:
            pyr_weighted = x
        else:
            pyr_weighted = x * Tensor.t(self.weights)

        wp = Tensor.norm(pyr_weighted)

        return wp


class V2FIT(nn.Module):

    def __init__(self, stimuli):
        super(V2FIT, self).__init__()

        """
        V2 model fit
        """

        self.stimuli = stimuli



    def forward(self, spyr, inTran, outTran):

        f = nn.functional.relu
        model_response = torch.mm(spyr.t(),  self.stimuli.t());
        output_response =  outTran[0] + outTran[1]*f(model_response) + outTran[2]*f(-1*model_response)

        #output_response = outTran[0] + outTran[1] * (model_response) + outTran[2] * torch.pow((model_response), 2)




        # model_response = torch.clamp(torch.mm(spyr, inTran[4] * self.stimulia), 0, 100)

        # (((((x - p(1)) / p(2)). ^ 2 + p(3)). ^ (1 / 2) - sqrt(p(3))). ^ p(4))



        # without input transfer
        # model_response = torch.clamp(torch.mm(spyr, self.stimuli), 0, 100)

        return output_response