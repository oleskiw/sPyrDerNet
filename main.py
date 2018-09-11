from models.v2pca import *
from utils import *
import numpy as np
from data_loader import loadmodel_mat, loadtargets_mat, saveoutput_mat
from torch.optim import SGD

import torch


def main():
    # set filename parameter
    filename = 'bundle_64_1024_1-2-3_1'
    dirname = './data/'

    # load network datafile
    network = loadmodel_mat(data_path=dirname+filename+'_model.mat')
    printout('Network loaded from model .mat file')

    # load datafile with images and desired component expressions
    dataTargets = loadtargets_mat(data_path=dirname+filename+'_targets.mat')

    printout('Targets loaded from .mat file')

    # do some processing
    # outputImg = np.zeros(dataTargets['img'].shape)
    outputImgSpyr = np.zeros(dataTargets['imgSpyr'].shape)
    outputExp = np.zeros((dataTargets['img'].shape[0], 32))
    interExp = []
    interObj = []
    interSpyr = []

    for i in range(dataTargets['img'].shape[0]):

        # get target data
        imgSpyr = dataTargets['imgSpyr'][i, :]
        imgSpyrTensor = torch.tensor(imgSpyr, dtype=dtype).to(device)
        expTarget = torch.tensor(dataTargets['expressionTarget'][i, :], dtype=dtype).to(device)

        imgSpyrDescender = imgSpyrTensor.clone()
        imgSpyrDescender.requires_grad_()

        # iteration details
        stepMax = 2000
        stepSaveResult = 400

        resultExp = []
        resultObj = []
        resultSpyr = []

        # prepare optimizer
        optimizer = SGD([imgSpyrDescender], lr=0.05)

        print("Processing image " + str(i) + " of " + str(dataTargets['img'].shape[0]))
        for stepCount in range(stepMax):
            def closure():
                optimizer.zero_grad()
                [exp, _] = network(imgSpyrDescender)

                expResidual = (exp - expTarget)
                obj = expResidual.norm()
                obj.backward(retain_graph=True)

                if stepCount % stepSaveResult == 0:
                    # save intermediate results
                    resultExp.append(exp.clone().detach().numpy())
                    resultObj.append(obj.clone().detach().numpy())
                    resultSpyr.append(imgSpyrDescender.clone().detach().numpy())

                    # display intermediate values
                    # print("Residual: " + str(expResidual.detach().numpy()))
                    # print("Objective: " + str(obj.detach().numpy()))
                return obj

            optimizer.step(closure)
            # imagelist.append(x.clone())

        # save final and intermediate reults
        [OptimExp, _] = network(imgSpyrDescender)
        outputExp[i, :] = OptimExp.detach().numpy()
        outputImgSpyr[i, :] = imgSpyrDescender.detach().numpy()
        interExp.append(resultExp)
        interObj.append(resultObj)
        interSpyr.append(resultSpyr)

    printout('Processing complete')

    # save output
    dataOut = {'imgSpyr': outputImgSpyr, 'expression': outputExp,
               'intermediateExpression': interExp, 'intermediateObjective': interObj, 'intermediateSpyr': interSpyr}

    saveoutput_mat(data_path='../sciTest/'+filename+'_output.mat', data=dataOut)
    printout('output .mat file written')


if __name__ == "__main__":
    main()
