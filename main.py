import argparse
import torch
from config import *
from models.v2pca import *
from utils import *

from data_loader import loadmodel_mat, loadtargets_mat, saveoutput_mat
from torch.optim import SGD
from pathlib import Path
import torch
import numpy as np
import h5py
import os
import sys

def main():
    np.seterr(invalid='ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('-f', action='store', nargs='?', dest='file', default=[], type=str,
                        help='file to be processed.')
    parser.add_argument('-p', action='store', nargs='?', dest='part', default=1, type=int,
                        help='Number of cross-validation partitions to run.')
    parser.add_argument('-l', action='append', nargs='*', dest='reg', default=[], type=float,
                        help='Add regularization value to optimization list.')
    parser.add_argument('-b', action='store', dest='bsf', default=.1, type=float,
                        help='Batch size fraction.')
    parser.add_argument('-step', action='store', default='.00001', type=float,
                        help='Set pyramid step size.')
    parser.add_argument('-stepn', action='store', default='.333', type=float,
                        help='stet nonlinear transform step size.')
    parser.add_argument('-smax', action='store', default='3000', type=int,
                        help='Maximum number of optimization steps.')
    parser.add_argument('-smin', action='store', default='0', type=int,
                        help='Maximum number of optimization steps.')

    #parse and collect input arguments
    args = parser.parse_args()

    inputDir = args.indir
    outputDir = args.outdir

    if len(args.reg) == 0:
        lambdaSet = [[0, .01, .02, 0.3, .04, 0.6, .08, .12, .16]]
    else:
        lambdaSet = args.reg
    if args.part == 0:
        partSet = [0]
    else:
        partSet = list(range(1, args.part + 1))

    batchSizeFraction = args.bsf
    #flatten lambdaSet
    lambdaSet = [item for sublist in lambdaSet for item in sublist]

    # set up optimization range (regularization factors)
    stepMin = args.smin
    stepSize = [args.step, args.stepn, args.stepn]
    stepMax = args.smax

    #setup output directory
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    #load file, extract info
    fname = args.file
    datafile = h5py.File(inputDir + '/' + fname, mode='r')

    dataStim = np.array(datafile['neuronData']['stimSpyr'])
    dataResponse = np.array(datafile['neuronData']['response'])
    #dataSmootherAx = torch.DoubleTensor(np.array(datafile['neuronData']['Mx']), device=device)
    #dataSmootherAy = torch.DoubleTensor(np.array(datafile['neuronData']['My']), device=device)
    dataResponseWeights = np.array(datafile['neuronData']['responseWeights']).transpose()
    dataCVPart = np.array(datafile['neuronData']['crossValPart']).transpose()

    dataOutLambdas= []
    for l in range(len(lambdaSet)):
        dataOutPartitions = []
        for p in partSet:
            print('==================================================================================================')
            print('= Optimizing: ' + fname + ', Lambda: %2.3f' % lambdaSet[l] + ', Partition %i/%i' %(p, len(partSet)) )
            print('==================================================================================================')
            sys.stdout.flush()

            #gather init data
            dataInitRf = torch.tensor(np.array(datafile['neuronData']['initRf']), dtype=dtype, device=torch.device("cpu"))
            dataInitIn = torch.tensor(np.array(datafile['neuronData']['initIn']), dtype=dtype, device=torch.device("cpu"))
            dataInitOut = torch.tensor(np.array(datafile['neuronData']['initOut']), dtype=dtype, device=torch.device("cpu"))
            dataRegWeights = torch.tensor(np.array(datafile['neuronData']['regWeight']), dtype=dtype, device=torch.device("cpu"))

            #use partition set to make fit/test stim/response matrices

            dataFitStim = torch.tensor(dataStim[:, dataCVPart.flatten() != p], dtype=dtype, device=torch.device("cpu"))
            dataFitResponse = torch.tensor(dataResponse[0, dataCVPart.flatten() != p], dtype=dtype, device=torch.device("cpu"))
            dataFitResponseWeights = torch.tensor(dataResponseWeights[dataCVPart.flatten() != p], dtype=dtype, device=torch.device("cpu"))
            fitDataset = torch.utils.data.TensorDataset(dataFitStim.t(), dataFitResponse.t(), dataFitResponseWeights)
            dataLoaderParams = {'batch_size': round(dataResponse.shape[1]*batchSizeFraction + 0.5),
                                'pin_memory': True,
                                'shuffle': True}
            fitDataloader = torch.utils.data.DataLoader(fitDataset, **dataLoaderParams)

            dataTestStim = torch.tensor(dataStim[:, dataCVPart.flatten() == p], dtype=dtype, device=torch.device("cpu"))
            dataTestResponse = torch.tensor(dataResponse[0, dataCVPart.flatten() == p], dtype=dtype, device=torch.device("cpu"))
            dataTestResponseWeights = torch.tensor(dataResponseWeights[dataCVPart.flatten() == p], dtype=dtype, device=torch.device("cpu"))
            testDataset = torch.utils.data.TensorDataset(dataTestStim.t(), dataTestResponse.t(), dataTestResponseWeights)
            testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=99999, pin_memory=True, shuffle=False)

            #run optimization
            optimItem = optimize_network(fitDataloader, testDataloader,
                                       dataInitRf, dataInitIn, dataInitOut, dataRegWeights,
                                       stepSize, lambdaSet[l], p, stepMax, stepMin)

            #save data for all partitions
            dataOutPartitions.append(optimItem)

        #save data for all lambdas
        dataOutLambdas.append(dataOutPartitions)

    #save optim output
    saveoutput_mat(outputDir + '/' + fname.replace('.mat', '_out.mat') , data=dataOutLambdas)


def optimize_network(fitDataloader, testDataloader,
    dataInitRf, dataInitIn, dataInitOut, dataRegWeights,
    stepSize, lambdaValue, partitionValue, epocMax, epocMin):

    # set up optimization parameters
    spyrDescender = dataInitRf.clone().to(device)
    inTransferDescender = dataInitIn.squeeze().clone().to(device)
    outTransferDescender = dataInitOut.squeeze().clone().to(device)
    regWeights = dataRegWeights.squeeze().clone().to(device)

    #hold all data in cpu for evaluation
    fitDataset = fitDataloader.dataset
    fitStim, fitResponse, fitWeights = fitDataset.__getitem__(range(fitDataset.__len__()))
    modelFit = V2FIT(fitStim)
    testDataset = testDataloader.dataset
    testStim, testResponse, testWeights = testDataset.__getitem__(range(testDataset.__len__()))
    modelTest = V2FIT(testStim)

    spyrDescender.requires_grad_()
    inTransferDescender.requires_grad_()
    outTransferDescender.requires_grad_()

    # create parameter list
    params = list()
    params.append(spyrDescender)
    params.append(inTransferDescender)
    params.append(outTransferDescender)

    # make optimizer
    optimizer = SGD([
        {'params': spyrDescender, 'lr': stepSize[0], 'momentum': 0.9},
        # {'params': inTransferDescender, 'lr': stepSize[1], 'momentum': 0.25},
        {'params': outTransferDescender, 'lr': stepSize[2], 'momentum': 0.7}
    ])

    resultSpyr = []
    resultInTran = []
    resultOutTran = []
    resultFitResp = []
    resultTestResp = []
    resultObjRate = []
    resultObjTestRate = []
    resultObjReg = []
    resultStep = []

    allResultStep = []
    allResultStepCount = []
    allResultFitEv = []
    allResultTestEv = []
    allResultObjRate = []
    allResultObjReg = []

    epocSaveResult = 20
    epocDisplayResult = 10

    rFitNorm = torch.sqrt(torch.sum((torch.pow(fitResponse, 2) * fitWeights.t())))
    rTestNorm = torch.sqrt(torch.sum((torch.pow(testResponse, 2) * testWeights.t())))

    #best step vars
    testEvMax = -9
    testEvMaxEpoc = 0
    objTrainMin = 9
    epocBack = round(epocMax * 0.1);

    #print formatters
    getdatum = lambda x: x.data.cpu().double().numpy()
    np.set_printoptions(formatter={"float_kind": lambda x: "%2.2f" % x})

    order = 1
    if order == 1:
        lambdaScale = 1;
    elif order == 2:
        lambdaScale = 10;
    ln = int(spyrDescender.shape[0] / 2)
    a = lambdaValue * lambdaScale

    stopFlag = False
    stepCount = 0
    for epocCount in range(epocMax):
        if stopFlag:
            break

        for z in range(2):
            for batchStim, batchResponse, batchWeights in fitDataloader:
                # Transfer to GPU
                batchStim, batchResponse, batchWeights = batchStim.to(device), batchResponse.to(device), batchWeights.to(device)

                #descend on fit data
                optimizer.zero_grad()

                #apply model to data
                batchModel = V2FIT(batchStim)
                batchPrediction = batchModel(spyrDescender, inTransferDescender, outTransferDescender)

                #compute objectives
                rBatchNorm = torch.sqrt(torch.sum((torch.pow(batchResponse, 2) * batchWeights.t())))
                batchRate = torch.sqrt(torch.sum((torch.pow((batchPrediction - batchResponse), 2) * batchWeights.t()))) / rBatchNorm
                batchReg = a * torch.norm((torch.sqrt(torch.pow(spyrDescender[range(ln - 1)], 2) + torch.pow(spyrDescender[range(ln, 2 * ln - 1)], 2))), p=order)
                batchRegNls = a * .01 * (torch.norm(inTransferDescender, p=2) + torch.norm(outTransferDescender, p=2))

                #final loss and gradient
                loss = batchRate + batchReg + batchRegNls  # + objRegSmooth
                loss.backward()

                #take a step
                optimizer.step()
                stepCount = stepCount + 1

        fitPrediction = modelFit(spyrDescender.to(torch.device('cpu')), inTransferDescender.to(torch.device('cpu')), outTransferDescender.to(torch.device('cpu')))
        testPrediction = modelTest(spyrDescender.to(torch.device('cpu')), inTransferDescender.to(torch.device('cpu')), outTransferDescender.to(torch.device('cpu')))

        #compute proper fit/test explained variance
        objFitRate = torch.sqrt(torch.sum(torch.pow((fitPrediction - fitResponse), 2) * fitWeights.t())) / rFitNorm
        objTestRate = torch.sqrt(torch.sum(torch.pow((testPrediction - testResponse), 2) * testWeights.t())) / rTestNorm
        objReg = a * torch.norm(regWeights[range(ln-1)]*torch.squeeze(torch.sqrt(torch.pow(spyrDescender[range(ln - 1)], 2) + torch.pow(spyrDescender[range(ln, 2 * ln - 1)], 2))), p=order)
        objRegNls = a * .01 * (torch.norm(inTransferDescender, p=2) + torch.norm(outTransferDescender, p=2))
        fitEv = (1 - ((torch.var(fitResponse - fitPrediction) / torch.var(fitResponse))))
        testEv = (1 - ((torch.var(testResponse - testPrediction) / torch.var(testResponse))))

        #descend
        if epocCount % epocDisplayResult == 0:
            print("-- epoc %3i " % epocCount + " step %4i " % stepCount +
                  "| FitRate: %1.4f" % getdatum(objFitRate) +
                  ", TestRate: %1.4f" % getdatum(objTestRate) +
                  ", RegSpyr: %1.4f" % getdatum(objReg) +
                  #", RegSmooth: " + float_formatter(objRegSmooth.detach().numpy()) +
                  ", RegNonlin: %1.4f" % getdatum(objRegNls) +
                  ", TestEv: %2.2f / %2.2f" % (getdatum(testEv), testEvMax))
            sys.stdout.flush()
            #print("  resp: " + str(fitResp[0, range(7)].detach().numpy()))
            #print("  rate: " + str(dataFitResponse[0, range(7)].detach().numpy()))
            #print("----inTran: %1.4f" % getdatum(inTransferDescender) + ", outTran: " + str(getdatum(outTransferDescender)))

        #store a short history
        fitEpocBack = 999
        if epocCount > epocBack:
            fitEpocBack = allResultObjRate[epocCount-epocBack]

        #check various termination critera
        exitTag = 'complete'

        if getdatum(objFitRate) > 3:
            stopFlag = True
            printout("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Terminating, fit unstable")
            exitTag = 'unstable'

        if epocCount > epocMin / 3:
            if getdatum(objFitRate) > 0.95 and epocCount > epocBack:
                stopFlag = True
                printout("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Terminating, fit not found")
                exitTag = 'defeated'

        if epocCount > epocMin:
            if abs(fitEpocBack - getdatum(objFitRate)) < 0.001:
                stopFlag = True
                printout("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Terminating, fit plateau reached")
                exitTag = 'plateau'
            elif (testEvMax - getdatum(testEv)) > .1 and (epocCount - testEvMaxEpoc) > epocBack:
                stopFlag = True
                printout("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Terminating, validation failing")
                exitTag = 'optima'

        #save all optimation data on infrequenct steps
        if epocCount % epocSaveResult == 0 or stopFlag:
            # save pre-step results
            resultSpyr.append(getdatum(spyrDescender))
            resultInTran.append(getdatum(inTransferDescender))
            resultOutTran.append(getdatum(outTransferDescender))
            resultFitResp.append(getdatum(fitPrediction))
            resultTestResp.append(getdatum(testPrediction))
            resultObjRate.append(getdatum(objFitRate))
            resultObjTestRate.append(getdatum(objTestRate))
            resultObjReg.append(getdatum(objReg))
            resultStep.append(epocCount)

        #save all loss function computations
        allResultStep.append(epocCount)
        allResultStepCount.append(stepCount)
        allResultFitEv.append(getdatum(fitEv))
        allResultTestEv.append(getdatum(testEv))
        allResultObjRate.append(getdatum(objFitRate))
        allResultObjReg.append(getdatum(objReg))

        #store test rate minimums
        if getdatum(testEv) > testEvMax:
            testEvMax = getdatum(testEv)
            testEvMaxEpoc = epocCount
            #store optimization params for best fit
            bestSol = {'spyr': getdatum(spyrDescender),
                       'inTran': getdatum(inTransferDescender),
                       'outTran': getdatum(outTransferDescender)}

    # final step has been preformed
    finalSol = {'spyr': getdatum(spyrDescender),
                'inTran': getdatum(inTransferDescender),
                'outTran': getdatum(outTransferDescender)}

    allResult = {'step': allResultStep,
                 'fitEv': allResultFitEv,
                 'testEv': allResultTestEv,
                 'objRate': allResultObjRate,
                 'objReg': allResultObjReg}

    optimParams = {#'fitStim': getdatum(dataFitStim),
                   #'testStim': getdatum(dataTestStim),
                   'initRf': getdatum(dataInitRf),
                   'initIn': getdatum(dataInitIn),
                   'initOut': getdatum(dataInitOut)}

    #check for 0 partition (no test set) and overwrite bestsol
    if partitionValue == 0:
        bestSol = finalSol

    # save output
    dataOut = {'optimParams': optimParams,
               'lambda': lambdaValue,
               'partition': partitionValue,
               'exitTag': exitTag,
               'resultSpyr': resultSpyr,
               'resultInTran': resultInTran,
               'resultOutTran': resultOutTran,
               'resultFitResp': resultFitResp,
               'resultTestResp': resultTestResp,
               'resultObjRate': resultObjRate,
               'resultObjTestRate': resultObjTestRate,
               'resultObjReg': resultObjReg,
               'resultStep': resultStep,
               'finalSol': finalSol,
               'bestSol': bestSol,
               'allResult': allResult}

    return dataOut

def synthesize_image(filename, directory):

    # load network datafile
    network = loadmodel_mat(data_path=directory + filename + '_model.mat')
    printout('Network loaded from model .mat file')

    # load datafile with images and desired component expressions
    dataTargets = loadtargets_mat(data_path=directory + filename + '_targets.mat')

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
        imgSpyrTensor = torch.Tensor(imgSpyr, device=device)
        expTarget = torch.Tensor(dataTargets['expressionTarget'][i, :], device=device)

        # construct feature vector for descent
        imgSpyrDescender = imgSpyrTensor.clone()
        imgSpyrDescender.requires_grad_()

        # iteration details
        stepMax = 1000
        stepSaveResult = 1000

        resultExp = []
        resultObj = []
        resultSpyr = []

        # prepare optimizer
        optimizer = SGD([imgSpyrDescender], lr=0.3)
        wp = network.window_power(imgSpyrDescender)
        wpFactor = 0;

        print("Processing image " + str(i) + " of " + str(dataTargets['img'].shape[0]))
        for stepCount in range(stepMax):
            def closure():
                optimizer.zero_grad()
                [exp, _] = network(imgSpyrDescender)

                expResidual = (exp - expTarget)
                obj = expResidual.norm() + wpFactor * Tensor.abs(((wp - network.window_power(imgSpyrDescender)) / wp))
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

        print("Exp Norm: " + str((OptimExp - expTarget).norm().detach().numpy()) +
              ", WP Val: " + str(wpFactor * ((wp - network.window_power(imgSpyrDescender)) / wp).detach().numpy()))

    printout('Processing complete')

    # save output
    dataOut = {'imgSpyr': outputImgSpyr, 'expression': outputExp,
               'intermediateExpression': interExp, 'intermediateObjective': interObj, 'intermediateSpyr': interSpyr}

    saveoutput_mat(data_path='../sciTest/' + filename + '_output.mat', data=dataOut)
    printout('output .mat file written')

if __name__ == "__main__":
    main()
