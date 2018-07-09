import torch
import os,sys
import h5py
from models.genutils import *
from config import *
import numpy as np
from models.v2pca import *
from dimred import ZCA,ICA,PCA
from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import matfile_dataload, h5py_dataload
from torch.optim import SGD
np.set_printoptions(suppress=True)
np.random.seed(5)
    
 

def compute_componentmat(savepath = './output/', dataset = 'mat', dimreduce='pca', use_window = True, num_textures=10, max_patches = 800):
    '''
    can choose between pca, ica, or zca for dimreduce option
    returns matrix of size num_coefficients (output of steerable pyramid and rectification) x num_components
    '''
    if dataset == 'mat':
       impatch, im_inds,matdata = matfile_dataload(precomp = False, num_textures = 10, max_patches = 800)  
    elif dataset == 'lcv':
        matdata = matfile_dataload(rawdat = True)
        impatch, im_inds = h5py_dataload(num_textures=num_textures, max_patches=max_patches)

    if use_window:
        window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
        windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
        windowinds = np.nonzero(windowbool)[0]
        window = torch.tensor(window, dtype=dtype).to(device)
    else:
        window = None
    del matdata
    

    printout("Data loaded, impatch shape:" +  str(impatch.shape))
    printout("Building pyramid...")
    network = V2PCA(imgSize=impatch.shape[1], K=4, N=2, nonlin='quadratic', window = window, pcaMat = None, ret_pyr = False, ncomp=32)
    x = impatch.reshape([impatch.shape[0], 1, impatch.shape[1], impatch.shape[2]])
    x = torch.tensor(x, dtype=dtype).to(device)
    coeff = network(x)
    if use_window:
        coeff = coeff[:,windowinds]
    printout("Coeffs generated...")

    del x
    torch.cuda.empty_cache()
    printout("Computing dimreduce for " + dimreduce)
    if dimreduce =='zca':
        compmat = ZCA(coeff)
        compmat = compmat.data.cpu().numpy()
    elif dimreduce == 'ica':
        compmat = ICA(coeff)
    elif dimreduce == 'pca':
        compmat = PCA(coeff)
        compmat = compmat.data.cpu().numpy()
    if use_window:
        compmatold = compmat
        compmat = np.zeros((window.size(0), compmatold.shape[1]))
        compmat[windowinds,:] = compmatold

    printout(compmat.shape)
    h5f = h5py.File(savepath + dimreduce + 'mat.h5', 'w')
    h5f.create_dataset('dat', data=compmat, compression='gzip')
    h5f.close()

    


def component_gradsynth(x, network, comp_index = 0, opt_type ='max', num_steps=4):
    '''
    For a given network and input image x, optimize the pixels in x such that the output
    component (given by comp_index) is maximized or minimized (given by opt_type).
    Generate num_steps gradient steps and return the image list
    '''
    imagelist = []
    output_orig = network(x.clone())
    optimizer = SGD([x],lr=5)
    index_vec = np.arange(output_orig.size(1))
    index_vec = np.delete(index_vec, comp_index)
    for i in range(num_steps):
        #renormalize images to mean 0 std 1 before next gradient step
        def closure():
            x.data = (x.data-x.data.mean())/x.data.std()
            optimizer.zero_grad()
            output = network(x)
            if opt_type == 'max':
                loss1 = -output[0,comp_index]
            elif opt_type == 'min':
                loss1 = output[0, comp_index]
            #loss2 = torch.mean((output_orig[0,index_vec] - output[0, index_vec])**2)
            loss = loss1
            loss.backward(retain_graph=True)
           # printout("Component Vector: " + str(np.around(output.data.cpu().numpy(), decimals=2)))
           # printout("Loss over other components: " + str(loss2.data.cpu().numpy()))
           # printout("Component " + str(comp_index) + " " + str(output[0,comp_index].data.cpu().numpy()))
            return loss

        optimizer.step(closure)
        imagelist.append(x.clone())
    return imagelist

    

def run_gradsynth(savepath = './output/', dataset='mat', dimreduce='pca', use_window = True):
    '''
    Main method that utilizes the component_gradsynth to generate images that max/min
    output components and saves images in a single image file
    '''

    if dataset == 'mat':
       impatch, im_inds,matdata = matfile_dataload(precomp = False, num_textures = 10, max_patches = 800)  
    elif dataset == 'lcv':
        matdata = matfile_dataload(rawdat = True)
        impatch, im_inds = h5py_dataload(num_textures=num_textures, max_patches=max_patches)
    if use_window:
        window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
        windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
        windowinds = np.nonzero(windowbool)[0]
        window = torch.tensor(window, dtype=dtype).to(device)
    else:
        window = None
    del matdata

    dimreducemat = h5py.File(savepath + dimreduce + 'mat.h5', 'r')['dat'][()]
    dimreducemat = torch.tensor(dimreducemat, dtype=dtype).to(device)   

    network = V2PCA(imgSize=impatch.shape[1], K=4, N=2, nonlin='quadratic', window = window, pcaMat = dimreducemat, ret_pyr = False, ncomp=32)
    
    printout("Network and data loaded...")

    impatches = []
    imsize = 64
    wn_image = np.random.randn(imsize, imsize)
    impatches.append(wn_image)
    numpatches = 4
    for i in range(numpatches):
        ind = np.random.randint(0,impatch.shape[0])
        impatches.append(impatch[ind].reshape([imsize,imsize]))
    num_steps = 4
    for comp_index in range(1,6):
        synth_list = []
        printout("Running synthesis for component " + str(comp_index))
        num_patch = 0 
        for im in impatches:
            num_patch += 1
            synth_list_im = []
            printout("Running image " + str(num_patch))
            for opt_type in ['max', 'min']:
                x = torch.tensor(im.reshape([1,1,im.shape[0], im.shape[1]]), dtype=dtype).to(device)
                x.requires_grad_()
                synth_images = component_gradsynth(x, network, comp_index, opt_type = opt_type,num_steps=num_steps)
                synth_list_im.append(synth_images)
            synth_list.append(synth_list_im)
    

        f, arrax = plt.subplots(len(impatches),num_steps*2 + 1, figsize=(30,20))
        imsize = 64
        for i in range(len(impatches)):
            for j in range(num_steps*2+1):
                if j == num_steps:
                    im = window_im(impatches[i], window)
                elif j < num_steps:
                    im = window_im(synth_list[i][1][-(j+1)].data.cpu().numpy().reshape([imsize, imsize]), window)
                elif j > num_steps:
                    im = window_im(synth_list[i][0][j-num_steps-1].data.cpu().numpy().reshape([imsize,imsize]), window)
                arrax[i,j].imshow(im, cmap = 'gray')
                arrax[i,j].axis('off')
        
        f.suptitle('Component ' + str(comp_index) + ' Gradient Synthesis')
        plt.savefig(savepath+'component_' + str(comp_index) + '.png')
        plt.close()




def main():
   # compute_componentmat(dimreduce='pca', use_window = True, num_textures = 10, max_patches = 800)
    run_gradsynth()



if __name__== "__main__":
    main()
    
