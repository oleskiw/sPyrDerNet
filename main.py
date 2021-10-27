import argparse
from config import *
from v2 import *
from utils import *

from data_loader import loadmodel_mat, loadtargets_mat, saveoutput_mat
from torch.optim import SGD
from pathlib import Path
import torch
import numpy as np
import h5py
import sys

def main():
    np.seterr(invalid='ignore')

    parser = argparse.ArgumentParser()
    parser.add_argument('indir', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('-f', action='store', nargs='?', dest='file', default=[], type=str,
                        help='file to be processed.')
    parser.add_argument('-fo', action='store', nargs='?', dest='fileOut', default=[], type=str,
                        help='output file.')
    parser.add_argument('-p', action='store', nargs='?', dest='part', default=1, type=int,
                        help='Number of cross-validation partitions to run.')
    parser.add_argument('-l', action='append', nargs='*', dest='reg', default=[], type=float,
                        help='Add regularization value to optimization list.')
    parser.add_argument('-u', action='append', nargs='+', dest='units', default=[], type=int, choices=range(1,1000),
                        help='units to parse')
    parser.add_argument('-b', action='store', dest='bsf', default='.2', type=float,
                        help='Batch size fraction.')
    parser.add_argument('-nl', action='store', dest='nl', nargs=3, default=[], type=float,
                        help='Output nonlinearity coefficients')
    parser.add_argument('-reg', action='store', nargs='?', dest='use_regweights', default=False, type=bool,
                        help='use regularization scaling weights')
    parser.add_argument('-step', action='store', default='.00001', type=float,
                        help='Set pyramid step size.')
    parser.add_argument('-stepn', action='store', default='.1', type=float,
                        help='stet nonlinear transform step size.')
    parser.add_argument('-model', action='store', nargs=1, default='v2', type=str, choices=('v2', 'v2pos', 'v2mpb'),
                        help='V2 model to fit')
    parser.add_argument('-smax', action='store', default='3000', type=int,
                        help='Maximum number of optimization steps.')
    parser.add_argument('-smin', action='store', default='0', type=int,
                        help='Maximum number of optimization steps.')

    #parse and collect input arguments
    args = parser.parse_args()

    inputDir = args.indir
    outputDir = args.outdir

    if len(args.reg) == 0:
        lambdaSet = [[0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16]]
    else:
        lambdaSet = args.reg
    if len(args.units) == 0:
        unitSet = [[1]]
    else:
        unitSet = args.units
    if args.part == 0:
        partSet = [0]
    else:
        partSet = list(range(1, args.part + 1))

    if args.model == ['v2mpb']:
        model_V2 = V2MPBFIT
        model_nl = 1
    elif args.model == ['v2pos']:
        model_V2 = V2FIT
        coeff_pos = True
        model_nl = 2
    else:
        model_V2 = V2FIT
        coeff_pos = False
        model_nl = 2

    use_regweights = args.use_regweights
    coeff_output = args.nl

    batchSizeFraction = args.bsf
    #flatten lambdaSet
    lambdaSet = [item for sublist in lambdaSet for item in sublist]

    #flatten units
    unitSet = [item for sublist in unitSet for item in sublist]
    unitSetIndex = np.array(unitSet)-1;

    # set up optimization range (regularization factors)
    stepMin = args.smin
    stepSize = [args.step, args.stepn, args.stepn]
    stepMax = args.smax

    #setup output directory
    Path(outputDir).mkdir(parents=True, exist_ok=True)

    #load file, extract info
    fname = args.file
    if args.fileOut == []:
        foutname = fname.replace('.mat', '_out.mat')
    else:
        foutname = args.fileOut

    datafile = h5py.File(inputDir + '/' + fname, mode='r')

    dataStim = np.array(datafile['neuronData']['stimSpyr'])
    dataResponse = np.array(datafile['neuronData']['response'])
    #dataSmootherAx = torch.DoubleTensor(np.array(datafile['neuronData']['Mx']), device=device)
    #dataSmootherAy = torch.DoubleTensor(np.array(datafile['neuronData']['My']), device=device)
    dataResponseWeights = np.array(datafile['neuronData']['responseWeights']).transpose()
    #dataResponseWeights = np.ones(dataResponseWeights.shape) #leave this on to disable response weights for poission-like noise
    dataCVPart = np.array(datafile['neuronData']['crossValPart']).transpose()

    dataOutLambdas= []
    for l in range(len(lambdaSet)):
        dataOutPartitions = []
        for p in partSet:
            print('==================================================================================================')
            print('= Optimizing ' + fname + ' Unit: ' + ','.join(str(*unitSet)) + ', Lambda: %2.3f' % lambdaSet[l] + ', Partition %i/%i' %(p, len(partSet)) )
            print('==================================================================================================')
            sys.stdout.flush()

            #gather init data
            dataInitRf = torch.tensor(np.array(datafile['neuronData']['initRf']), dtype=dtype, device=torch.device("cpu"))
            dataInitIn = torch.tensor(np.array(datafile['neuronData']['initIn']), dtype=dtype, device=torch.device("cpu"))
            dataInitOut = torch.tensor(np.array(datafile['neuronData']['initOut']), dtype=dtype, device=torch.device("cpu"))
            dataRegWeights = torch.tensor(np.array(datafile['neuronData']['regWeight']), dtype=dtype, device=torch.device("cpu"))

            # clear reg weights if not used
            if use_regweights == False:
                dataRegWeights = torch.ones(size=dataRegWeights.size(), dtype=dtype, device=torch.device("cpu"))
                dataRegWeights = dataRegWeights / torch.norm(dataRegWeights,p=2)

            # store output coefficients
            if not (coeff_output) == False:
                dataInitOut = torch.tensor(np.array([[coeff_output[0]], [coeff_output[1]], [coeff_output[2]]]),
                                           dtype=dtype, device=torch.device("cpu"))

            #override output coeffients
            if len(args.nl) == 3:
                nl = np.tile(np.array(args.nl).reshape([3,1]),(1,dataInitOut.shape[1]))
                dataInitOut = torch.tensor(nl, dtype=dtype, device=torch.device("cpu"))

            #extract only the used units
            dataInitRf = dataInitRf[:, unitSetIndex]
            dataInitIn = dataInitIn[:, unitSetIndex]
            dataInitOut = dataInitOut[:, unitSetIndex]
            dataRegWeights = dataRegWeights[:, unitSetIndex]



            if dataResponse.shape[0] == 1:
                dataResponse = dataResponse.transpose()

            # use partition set to make fit/test stim/response matrices
            dataFitStim = torch.tensor(dataStim[:, dataCVPart.flatten() != p], dtype=dtype, device=torch.device("cpu"))
            dataFitResponse = torch.tensor(dataResponse[dataCVPart.flatten() != p,:][:,unitSetIndex], dtype=dtype, device=torch.device("cpu"))
            dataFitResponseWeights = torch.tensor(dataResponseWeights[dataCVPart.flatten() != p,:][:,unitSetIndex], dtype=dtype, device=torch.device("cpu"))
            fitDataset = torch.utils.data.TensorDataset(dataFitStim.t(), dataFitResponse, dataFitResponseWeights)
            dataLoaderParams = {'batch_size': round(dataResponse.shape[0]*batchSizeFraction + 0.5),
                                'pin_memory': True,
                                'shuffle': True}
            fitDataloader = torch.utils.data.DataLoader(fitDataset, **dataLoaderParams)

            dataTestStim = torch.tensor(dataStim[:, dataCVPart.flatten() == p], dtype=dtype, device=torch.device("cpu"))
            dataTestResponse = torch.tensor(dataResponse[dataCVPart.flatten() == p,:][:,unitSetIndex], dtype=dtype, device=torch.device("cpu"))
            dataTestResponseWeights = torch.tensor(dataResponseWeights[dataCVPart.flatten() == p,:][:,unitSetIndex], dtype=dtype, device=torch.device("cpu"))
            testDataset = torch.utils.data.TensorDataset(dataTestStim.t(), dataTestResponse, dataTestResponseWeights)
            testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=99999, pin_memory=True, shuffle=False)

            #run optimization
            optimItem = optimize_network(model_V2, model_nl, coeff_pos, fitDataloader, testDataloader,
                                       dataInitRf, dataInitIn, dataInitOut, dataRegWeights,
                                       stepSize, lambdaSet[l], p, stepMax, stepMin, False)

            #save data for all partitions
            dataOutPartitions.append(optimItem)

        #save data for all lambdas
        dataOutLambdas.append(dataOutPartitions)

    #save optim output
    saveoutput_mat(outputDir + '/' + foutname , data=dataOutLambdas)


def optimize_network(V2MODEL, model_nl, coeff_pos, fitDataloader, testDataloader,
    dataInitRf, dataInitIn, dataInitOut, dataRegWeights,
    stepSize, lambdaValue, partitionValue, epocMax, epocMin, DO_BATCH):

    # set up optimization parameters
    spyrDescender = dataInitRf.clone().to(device)
    regWeights = dataRegWeights.clone().to(device)
    inTransferDescender = dataInitIn.clone().to(device)
    outTransferDescender = dataInitOut.clone().to(device)


    #hold all data in cpu for evaluation
    fitDataset = fitDataloader.dataset
    fitStim, fitResponse, fitWeights = fitDataset.__getitem__(range(fitDataset.__len__()))
    if fitResponse.ndim == 1:
        fitResponse = fitResponse.unsqueeze(1)
        fitWeights = fitWeights.unsqueeze(1)
    modelFit = V2MODEL(fitStim, model_nl, coeff_pos)
    testDataset = testDataloader.dataset
    testStim, testResponse, testWeights = testDataset.__getitem__(range(testDataset.__len__()))
    if testResponse.ndim == 1:
        testResponse = testResponse.unsqueeze(1)
        testWeights = testWeights.unsqueeze(1)
    modelTest = V2MODEL(testStim, model_nl, coeff_pos)

    #if not doing batch learning, get all data onto gpu
    if DO_BATCH == False:
        fitStimDevice, fitResponseDevice, fitWeightsDevice = fitStim.to(device), fitResponse.to(device), fitWeights.to(device)
        if fitResponseDevice.ndim == 1:
            fitResponseDevice = fitResponseDevice.unsqueeze(1)
            fitWeightsDevice = fitWeightsDevice.unsqueeze(1)
        fitModelDevice = V2MODEL(fitStimDevice, model_nl, coeff_pos)

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
        {'params': outTransferDescender, 'lr': stepSize[2], 'momentum': 0.9}
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
    epocDisplayResult = 20

    rFitNorm = torch.sqrt(torch.sum((torch.pow(fitResponse, 2) * fitWeights),dim=0))
    rTestNorm = torch.sqrt(torch.sum((torch.pow(testResponse, 2) * testWeights), dim=0))

    #best step vars
    testEvMax = torch.tensor(-9*np.ones(spyrDescender.shape[1]), dtype=dtype, device=torch.device("cpu"))
    testEvMaxEpoc = 0
    epocBack = round(epocMax * 0.2);

    #print formatters
    getdatum = lambda x: x.data.cpu().double().numpy()
    dstr = lambda x: ("%1.3F_" % w for w in getdatum(x))
    np.set_printoptions(formatter={"float_kind": lambda x: "%2.2f" % x})


    ln = int(spyrDescender.shape[0] / 2)
    a = 1 * lambdaValue

    stopFlag = False
    stepCount = 0
    for epocCount in range(epocMax):
        if stopFlag:
            break

        if DO_BATCH:
            for batchStim, batchResponse, batchWeights in fitDataloader:
                # Transfer to GPU
                batchStim, batchResponse, batchWeights = batchStim.to(device), batchResponse.to(device), batchWeights.to(device)

                #descend on fit data
                optimizer.zero_grad()

                #apply model to data
                batchModel = V2MODEL(batchStim, model_nl, coeff_pos)
                batchPrediction = batchModel(spyrDescender, inTransferDescender, outTransferDescender)

                # #if use, update with latest objective
                # #compute objectives
                # rBatchNorm = torch.sqrt(torch.sum((torch.pow(batchResponse, 2) * batchWeights),0))
                # batchRate = torch.sqrt(torch.sum((torch.pow((batchPrediction.t() - batchResponse), 2) * batchWeights),0)) / rBatchNorm
                # batchReg = torch.norm(regWeights[range(ln),:]*(((spyrDescender[range(ln),:]) ** 2 + (spyrDescender[range(ln, 2*ln),:]) ** 2) ** (1/2)),p=order, dim=0)
                # batchRegNls = torch.norm(outTransferDescender[range(0,3)], p=2)
                # #batchSparsity = a * (torch.mean(((batchPrediction-torch.mean(batchPrediction))/torch.std(batchPrediction))**4))
                # #final loss and gradient
                # loss = torch.norm(batchRate + (batchReg*batchRegNls), p=1) #+ batchSparsity  + objRegSmooth
                loss.backward()

                #take a step
                optimizer.step()
                stepCount = stepCount + 1
        else:
            for z in range(10):
                # descend on fit data
                optimizer.zero_grad()

                # apply model to data
                fitPredictionDevice = fitModelDevice(spyrDescender, inTransferDescender, outTransferDescender)

                # compute objectives
                rBatchNorm = torch.sqrt(torch.sum((torch.pow(fitResponseDevice, 2) * fitWeightsDevice), 0))
                batchRate = torch.sqrt(torch.sum((torch.pow((fitPredictionDevice.t() - fitResponseDevice), 2) * fitWeightsDevice), 0)) / rBatchNorm
                batchReg2 = torch.norm(regWeights[range(ln), :]*(((spyrDescender[range(ln), :]) ** 2 + (spyrDescender[range(ln, 2 * ln), :]) ** 2) ** (1 / 2)),p=2, dim=0)
                batchReg1 = torch.norm(regWeights[range(ln), :]*(((spyrDescender[range(ln), :]) ** 2 + (spyrDescender[range(ln, 2 * ln), :]) ** 2) ** (1 / 2)), p=1, dim=0)
                batchReg = a*(batchReg1/batchReg2)
                batchRegNls = torch.norm(outTransferDescender[range(0,3)], p=2)
                # batchSparsity = a * (torch.mean(((batchPrediction-torch.mean(batchPrediction))/torch.std(batchPrediction))**4))
                # final loss and gradient
                loss = torch.norm(batchRate + (batchReg), p=1)  # + batchSparsity  + objRegSmooth
                loss.backward()

                # take a step
                optimizer.step()
                stepCount = stepCount + 1

        fitPrediction = modelFit(spyrDescender.to(torch.device('cpu')), inTransferDescender.to(torch.device('cpu')), outTransferDescender.to(torch.device('cpu')))
        testPrediction = modelTest(spyrDescender.to(torch.device('cpu')), inTransferDescender.to(torch.device('cpu')), outTransferDescender.to(torch.device('cpu')))

        #compute proper fit/test explained variance
        objFitRate = torch.sqrt(torch.sum(torch.pow((fitPrediction.t() - fitResponse), 2) * fitWeights,0)) / rFitNorm
        objTestRate = torch.sqrt(torch.sum(torch.pow((testPrediction.t() - testResponse), 2) * testWeights,0)) / rTestNorm
        objReg2 = torch.norm((((regWeights[range(ln),:]*spyrDescender[range(ln),:])**2) + ((regWeights[range(ln, 2*ln),:]*spyrDescender[range(ln, 2*ln),:])**2))**(1/2),p=2, dim=0)
        objReg1 = torch.norm((((regWeights[range(ln),:]*spyrDescender[range(ln),:])**2) + ((regWeights[range(ln, 2*ln),:]*spyrDescender[range(ln, 2*ln),:])**2))**(1/2),p=1, dim=0)
        objReg = a*(objReg1/objReg2)
        objRegNls = torch.norm(outTransferDescender[range(0,3)], p=2)

        fitEv = (1 - (torch.var(fitResponse.t() - fitPrediction, 1) / torch.var(fitResponse)))
        testEv = (1 - (torch.var(testResponse.t() - testPrediction, 1) / torch.var(testResponse)))

        #descend
        if epocCount % epocDisplayResult == 0:
            print("-- epoc %3i " % epocCount + " step %4i " % stepCount +
                  "| FitRate: "+ "".join(dstr(objFitRate)) +
                  ", TestRate: "+ "".join(dstr(objTestRate)) +
                  ", Reg: " + "".join(dstr(objReg)) +
                  ", Reg1: "+ "".join(dstr(objReg1)) +
                  ", Reg2: " + "".join(dstr(objReg2)) +
                  #", RegSmooth: " + float_formatter(objRegSmooth.detach().numpy()) +
                  #", RegNonlin: %.4f" % objRegNls +
                  ", TestEv: "+"".join(dstr(testEv)) + " Max: "+ "".join(dstr(testEvMax)))
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

        if (min(getdatum(objFitRate)) > 3 or all(np.isnan(getdatum(objFitRate))) ) and epocCount > epocBack:
            stopFlag = True
            printout("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Terminating, fit unstable")
            exitTag = 'unstable'

        if epocCount > epocMin :
            if min(getdatum(objFitRate)) > .9 and epocCount > epocBack:
                stopFlag = True
                printout("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Terminating, fit not found")
                exitTag = 'defeated'

        if epocCount > epocMin:
            if max(abs(fitEpocBack - getdatum(objFitRate))) < 0.01:
                stopFlag = True
                printout("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Terminating, fit plateau reached")
                exitTag = 'plateau'
            elif max(abs((getdatum(testEvMax) - getdatum(testEv)))) > .1 and (epocCount - testEvMaxEpoc) > epocBack:
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
        if any(testEv > testEvMax) or epocCount == 0:
            testEvMax[testEv > testEvMax] = testEv[testEv > testEvMax]
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

    # do not store spyr for every step on all partitions
    if partitionValue > 1:
        resultSpyr = {}


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
               'allResult': allResult,
               'model_nl': model_nl}

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
