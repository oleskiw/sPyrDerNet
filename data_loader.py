from scipy.io import loadmat,savemat
from sklearn.feature_extraction.image import extract_patches_2d
import h5py
import numpy as np 
import os
import glob
from skimage.io import imread
from skimage.transform import resize
import numpy as np
import torch
from models.v2pca import *
from torch.utils.data import Dataset, DataLoader

def extract_patches_full(large_ims, patchsize, num_ims, max_patches):
    largeim_size = large_ims.shape[-1]
    impatch = np.empty((0,patchsize*patchsize), float)
    im_inds = []
    for i in range(num_ims):
        print(i)
        ims = extract_patches_2d(large_ims[i,:,:], (patchsize,patchsize), max_patches = max_patches)
        ims = np.reshape(ims, [ims.shape[0], ims.shape[1]*ims.shape[1]])
        #normed_ims = (ims - ims.mean(axis=1, keepdims=1))/ims.std(axis=1, keepdims=1)
        impatch = np.vstack((impatch, ims))
        im_inds.extend([i]*max_patches)

    impatch = impatch.reshape(-1, patchsize, patchsize)
    inds = np.arange(impatch.shape[0])
    np.random.shuffle(inds)
    impatch = impatch[inds]
    im_inds = np.array(im_inds)
    im_inds = im_inds[inds]

    return impatch, im_inds

def converth5tomat(datapath=None, savepath = None):
    d = {}
    arr = h5py.File(datapath, 'r')['dat'][()]
    d['dat'] = arr
    savemat(savepath, d)

def matfile_dataload(data_path='./data/bundleClassify_64_12_1.mat',rawdat = False, precomp = True, num_textures = 12, patchsize = 64, max_patches = 1000):
    '''
    mat file contains nested structs of following structure
    data
        input (input data images and coefficients from steerably pyramid output)
            imgSample (sample large images that patches are taken from)
            imgSampleSmall (resized version of imgSample to 64x64)
            dataMatrix (classification data matrix)
            dataClassificationGroup
            dataTextureStatistics
            dataImg (sampled image patches from original imgSample images)
            dataScrambledImg (sampled image patches but phase scrambled)
            
        f (component filter data)
            filter
                coeff
                coeffPermutation
                coeffRecon
            coeffWeights
            coeffIndex
            coeffTransferSet
            componentResponseMatrix
            componentError_l
            componentError_q
        spyr (name of steerable pyramid filters used from simoncelli code)
            'sp3Filters'
    '''

    matdata = loadmat(data_path)
    num_ims = num_textures
    patchsize = patchsize
    if rawdat:
        return matdata
    else:
        window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
        if precomp:
            precomp_im = np.array(matdata['data'][0][0]['input'][0][0]['dataImg'])
            impatch = precomp_im[:,:,:]
            im_inds = None
        else:
            fn = '/scratch/np1742/texture-modeling/dat/' + data_path.strip('.mat').split('/')[-1] + '_' + str(num_ims) + '_' + str(max_patches) + '.h5'

            if os.path.isfile(os.path.abspath(fn)):
                h5f = h5py.File(fn, 'r')
                impatch = np.array(h5f['dat'])
                im_inds = np.array(h5f['inds'])
            else:
                large_ims = np.array(matdata['data'][0][0]['input'][0][0]['imgSample'])
                large_ims = large_ims[:,0,:,:]
                impatch, im_inds = extract_patches_full(large_ims, patchsize, num_ims, max_patches)
                h5f = h5py.File('/scratch/np1742/texture-modeling/dat/' + data_path.strip('.h5').split('/')[-1] + '_' + str(num_ims) + '_' + str(max_patches) + '.h5', 'w')
                h5f.create_dataset('dat', data = impatch, compression='gzip')
                h5f.create_dataset('inds', data = im_inds, compression = 'gzip')
                h5f.close() 
            

        return impatch,im_inds, matdata

def aggregate_image_h5py(image_dir = None, imsize = 512):
    extensions = ['.jpg', '.gif', '.png', '.tiff']
    numims = 0
    imlist = []
    for ext in extensions:
        imlisttmp = glob.glob(os.path.abspath(image_dir) + '/*' + ext)
        numims += len(imlisttmp)
        imlist += imlisttmp
    print("Num images found = " + str(numims))
    imarr = np.empty((numims,imsize,imsize), np.float32)
    i = 0
    for imname in imlist:
        im = imread(imname, as_gray = True)
        im = resize(im, (imsize, imsize), anti_aliasing = True)
        imarr[i,:,:] = im
        i += 1

    dirname = image_dir.split('/')[-1]
    imarrh5 = h5py.File('/scratch/np1742/texture-modeling/dat/' + dirname +'.h5', 'w')
    imarrh5.create_dataset('dat', data = imarr, compression="gzip")
    imarrh5.close()

        
class TextureImageDataset(Dataset):

    def __init__(self, datapath, num_textures, max_patches, transform = None):
        self.impatches, self.iminds = h5py_dataload(datapath = datapath, num_textures=num_textures, max_patches=max_patches)
        self.transform = transform
        
    
    def __len__(self):
        return len(self.impatches)


    def __getitem__(self, idx):
        sample = {'image': self.impatches[idx], 'textureind': self.iminds[idx]}
        if self.transform:
            return self.transform(sample)

        return sample

class ToImageTensor(object):
    def __call__(self, sample):
        image, ind = sample['image'], sample['textureind']

        image = torch.tensor(image.reshape([1,image.shape[0], image.shape[1]]), dtype=dtype).to(device)
        return {'image': image, 'textureind': ind}



def h5py_dataload(datapath = './data/cropped512-gray-jpg.h5', num_textures = 100, patchsize = 64, max_patches = 1000):
    '''
    Extract image patches out of homogeneous textures stored in a 
    h5py file numimages x im_height x im_width
    '''

    large_ims = h5py.File(datapath, 'r')['dat']

    num_ims = num_textures
    patchsize = patchsize
    fn = '/scratch/np1742/texture-modeling/dat/' + datapath.strip('.h5').split('/')[-1] + '_' + str(num_ims) + '_' + str(max_patches) + '.h5'
    if os.path.isfile(os.path.abspath(fn)):
        h5f = h5py.File(fn, 'r')
        impatch = np.array(h5f['dat'])
        im_inds = np.array(h5f['inds'])
    else:
        impatch, im_inds = extract_patches_full(large_ims, patchsize, num_ims, max_patches)

        h5f = h5py.File('/scratch/np1742/texture-modeling/dat/' + datapath.strip('.h5').split('/')[-1] + '_' + str(num_ims) + '_' + str(max_patches) + '.h5', 'w')
        h5f.create_dataset('dat', data = impatch, compression='gzip')
        h5f.create_dataset('inds', data = im_inds, compression = 'gzip')
        h5f.close() 

    return impatch, im_inds


def compute_spyr(datapath = './data/cropped512-gray-jpg.h5', num_textures = 100, max_patches = 800):
    matdata = matfile_dataload(rawdat = True)
    window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
    windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
    windowinds = np.nonzero(windowbool)[0]
    window = torch.tensor(window, dtype=dtype).to(device)
    dataset = TextureImageDataset(datapath, num_textures=num_textures, max_patches=max_patches, transform=ToImageTensor())
    batch_size = 1000
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    model = V2PCA(imgSize = 64, K = 4, N = 2, includeHF = True, nonlin = 'smoothabs', window = window, pcaMat = None, ret_pyr = False, ncomp = 32)
    coeffs = np.zeros((num_textures*max_patches, windowinds.shape[0]))
    inds = np.zeros(num_textures*max_patches)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch)
        coeffbatch = model(sample_batched['image']).data.cpu().numpy()
        indbatch = sample_batched['textureind']
        coeffs[i_batch*batch_size:i_batch*batch_size + batch_size, :] = coeffbatch[:, windowinds]
        inds[i_batch*batch_size:i_batch*batch_size + batch_size] = indbatch
        torch.cuda.empty_cache()
        del coeffbatch
        

    h5f = h5py.File(datapath.rstrip('.h5') + '_' + str(num_textures) + "_" + str(max_patches)+ "_windcoeff.h5", 'w')
    h5f.create_dataset('dat', data = coeffs, compression='gzip')
    h5f.create_dataset('inds', data = inds, compression = 'gzip')
    h5f.close() 
    
    return coeffs, inds
    

    
def h5py_coeff_dataload(datapath, num_textures = 100, max_patches = 800):
    '''
    load pyrcoeffs and inds from h5 file
    '''
    fn = datapath.rstrip('.h5') + '_' + str(num_textures) + "_" + str(max_patches) + "_windcoeff.h5" 

    if os.path.isfile(os.path.abspath(fn)):
        h5f = h5py.File(fn, 'r')
        imcoeffs = np.array(h5f['dat'])
        im_inds = np.array(h5f['inds'])
    else:
        imcoeffs, im_inds = compute_spyr(datapath, num_textures, max_patches)

    return imcoeffs, im_inds

class TextureCoeffDataset(Dataset):

    def __init__(self, datapath, num_textures, max_patches, transform = None):
        self.imcoeffs, self.iminds = h5py_coeff_dataload(datapath = datapath, num_textures=num_textures, max_patches=max_patches)
        self.transform = transform
        
    
    def __len__(self):
        return len(self.imcoeffs)


    def __getitem__(self, idx):
        sample = {'coeff': self.imcoeffs[idx], 'textureind': int(self.iminds[idx])}
        if self.transform:
            return self.transform(sample)

        return sample

class ToCoeffTensor(object):
    def __call__(self, sample):
        coeff, ind = sample['coeff'], sample['textureind']

        coeff = torch.tensor(coeff, dtype=dtype).to(device)
        return {'coeff': coeff, 'textureind': ind}

