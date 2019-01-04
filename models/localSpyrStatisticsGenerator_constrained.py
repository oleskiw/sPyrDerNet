import pyrtools
import matplotlib.pyplot as plt
import torch
import numpy as np
import pyrtools as pt
from sklearn.feature_extraction.image import extract_patches
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class localSpyrStatisticsGenerator:
    def __init__(self,rectification='FullWave', power=2,
                 neighborhoodSize=np.array([8,4,2]), baseLevel=1,
                 reweightScales=True, order=3,logscale=True):
        self.rectification = rectification # 'FullWave','HalfWave',or 'Linear/None'
        self.power = power
        self.neighborhoodSize = neighborhoodSize 
        self.baseLevel = baseLevel  # typically 0 or 1.  1 will skip the highest scale when building the specified neighborhood. 0 won't.
        self.reweightScales = reweightScales
        self.order = order #  order = number of orientation bands - 1
        self.logscale = logscale
        
    def generate(self,im):
        
        # generate pyramid response
        pyr = pt.SFpyr(im,self.baseLevel+len(self.neighborhoodSize),self.order)
        coefficients = [torch.tensor(mat,dtype=torch.double, requires_grad=True) for mat in pyr.pyr]

        
        if self.rectification=='FullWave': # only one implemented right now!!
            coeff = [torch.sqrt(mat.pow(2)+1e-4) for mat in coefficients]
        else:
            raise ValueError( self.rectification + ' is not an implemented option for rectification')
            
        # get the local bundles
        samples = self.getNeighborhoods(coeff)
    
        if self.logscale:
            samples = torch.log(samples)

        # raise to the indicated power 
        samples = samples**self.power
    

        return pyr, coefficients, samples

    def generateFromCoeff(self,coefficients):
        
        if self.rectification=='FullWave': # only one implemented right now!!
            coeff = [torch.sqrt(mat.pow(2)+1e-4)    for mat in coefficients]
        else:
            raise ValueError( self.rectification + ' is not an implemented option for rectification')
        
        # get the local bundles
        samples = self.getNeighborhoods(coeff)

        if self.logscale:
            samples = torch.log(samples)

        # raise to the indicated power 
        samples = samples**self.power
        

        return samples

        
    
    def getNeighborhoods(self,coeff):
        
        nbdSz = self.neighborhoodSize
        baseLev = self.baseLevel
        
        nOri = self.order+1 # number of orientations
        levs = len(nbdSz) # number of scales
        nfeatures = np.sum(nbdSz**2)*nOri # total number of features


        for lNum in range(0,levs): # for each relevant scale in the neighborhood
            lev = baseLev + lNum
            nsz = nbdSz[lNum]  # spatial size at this scale
            sub = int(2**(levs-lNum-1)) # sampling rate for this scale

            for j in range(0,nOri): # do the sampling for each orientation band
                nb = lev*(nOri)+j+1
                coeff[nb]= coeff[nb].unfold(0,int(nbdSz[lNum]),sub).unfold(1,int(nbdSz[lNum]),sub)
                coeff[nb] = coeff[nb].reshape((coeff[nb].shape[0]*coeff[nb].shape[1],coeff[nb].shape[2]*coeff[nb].shape[3]))
                coeff[nb] = coeff[nb].t()


                if self.reweightScales:
                    coeff[nb] = coeff[nb] * sub**2
                
        samples = torch.cat(coeff[baseLev*nOri+1: baseLev*nOri + levs*nOri+1],0)
        return samples