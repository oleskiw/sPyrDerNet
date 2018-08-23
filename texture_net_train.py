import torch
import os, sys
import h5py
from models.genutils import *
from config import *
import numpy as np
from models.v2net import *
from models.objectives import covratio_loss
from utils import *
from data_loader import *
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
np.set_printoptions(suppress=True)

np.random.seed(5)

def test_v2net(savepath, inptype = 'coeff'):
    h5f = h5py.File(savepath + 'learnedtransform.h5', 'r')

    

def train_v2net(savepath, inptype = 'coeff'):
   
    matdata = matfile_dataload(rawdat = True)
    window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
    window = torch.tensor(window, dtype=dtype).to(device)
    if inptype == 'image':
        num_textures = 20
        max_patches = 800
        dataset = TextureImageDataset('./data/cropped512-gray-jpg.h5', num_textures=20, max_patches=800, transform=ToImageTensor())
    elif inptype == 'coeff':
        num_textures = 100
        max_patches = 800
        dataset = TextureCoeffDataset('./data/cropped512-gray-jpg.h5', num_textures=200, max_patches=1000, transform=ToCoeffTensor())


    dataloader = DataLoader(dataset, batch_size = num_textures*max_patches, shuffle = True)
    print('Data loaded...')
    model = V2Net(imgSize = 64, K=4, N=2, includeHF=True, inp = inptype, nonlin = 'smoothabs', window = window, ret_pyr = False, ncomp = 10, out_activ = True).to(device)
    optimizer = Adam(model.parameters(), lr = 0.01)
    loss_prev = 0
    iter_no_decrease = 0
    for epochs in range(5000):
        for i_batch, sample_batched in enumerate(dataloader):
            components = model(sample_batched[inptype])
            loss = covratio_loss(components, sample_batched['textureind'])
            print("Loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss.data.cpu().numpy() < loss_prev - 0.0001*np.abs(loss_prev):
            print("Loss prev, loss:", loss_prev, loss.data.cpu().numpy())
            loss_prev = loss.data.cpu().numpy()
            iter_no_decrease = 0
        else:
            iter_no_decrease += 1
            if iter_no_decrease == 20:
                break

    lin_trans_weights = model.lintrans.weight.data.cpu().numpy() 
    windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
    windowinds = np.nonzero(windowbool)[0]
    x = np.zeros((10,len(windowbool))) + 1e-8
    x[:,windowinds] = lin_trans_weights
    h5f = h5py.File(savepath + 'learnedtransform.h5','w')
    h5f.create_dataset('dat', data=x, compression='gzip')
    h5f.close()
    


def main():
    train_v2net('./output/')

if __name__== "__main__":
    main()






