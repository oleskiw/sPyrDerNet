from scipy.io import loadmat, savemat
from sklearn.feature_extraction.image import extract_patches_2d
import h5py
import numpy as np 
import os
import glob
from skimage.io import imread
from skimage.transform import resize
from v2 import *


def extract_patches_full(large_ims, patchsize, num_ims, max_patches):
    # largeim_size = large_ims.shape[-1]
    impatch = np.empty((0, patchsize*patchsize), float)
    im_inds = []
    for i in range(num_ims):
        ims = extract_patches_2d(large_ims[i, :, :], (patchsize, patchsize), max_patches=max_patches)
        ims = np.reshape(ims, [ims.shape[0], ims.shape[1]*ims.shape[1]])
        normed_ims = (ims - ims.mean(axis=1, keepdims=1))/ims.std(axis=1, keepdims=1)
        impatch = np.vstack((impatch, normed_ims))
        im_inds.extend([i]*max_patches)

    impatch = impatch.reshape(-1, patchsize, patchsize)
    inds = np.arange(impatch.shape[0])
    np.random.shuffle(inds)
    impatch = impatch[inds]
    im_inds = np.array(im_inds)
    im_inds = im_inds[inds]

    return impatch, im_inds


def converth5tomat(datapath=None, savepath=None):
    d = {}
    arr = h5py.File(datapath, 'r')['dat'][()]
    d['dat'] = arr
    savemat(savepath, d)


def loadmodel_mat(data_path=None):
    datamodel = loadmat(data_path)

    weights = np.array(datamodel['dataModel'][0][0]['weights'])
    # weightIndex = np.array(datamodel['dataModel'][0][0]['weightIndex'])
    components = np.array(datamodel['dataModel'][0][0]['components'])
    transferFunction = datamodel['dataModel'][0][0]['transferFunction'][0]

    return V2(imgSize=64, K=4, N=2,
              transferFunction=transferFunction, components=components, weights=weights,
              ncomp=32)


def loadtargets_mat(data_path=None):
    # dataTargets.imgNumber = nan(1, numTargets);
    # dataTargets.k = nan(numTargets, 1);
    # dataTargets.sample = nan(numTargets, 1);
    # dataTargets.step = nan(numTargets, 1);
    # dataTargets.expressionTarget = nan(numTargets, size(data.f.compomnentResponseStasitics, 2));
    # dataTargets.img = nan(numTargets, 64, 64);

    dt = loadmat(data_path)
    dtImg = np.array(dt['dataTargets'][0][0]['img'])
    dtExpression = np.array(dt['dataTargets'][0][0]['expression'])
    dtExpressionTarget = np.array(dt['dataTargets'][0][0]['expressionTarget'])
    dtImgSpyr = np.array(dt['dataTargets'][0][0]['imgSpyr'])

    dataTarget = {'img': dtImg, 'expression': dtExpression,
                  'expressionTarget': dtExpressionTarget, 'imgSpyr': dtImgSpyr}

    return dataTarget


def saveoutput_mat(data_path=None, data=None):
    savemat(data_path, {'output': data})


def matfile_dataload(data_path='./data/bundleClassify_64_12_1.mat', rawdat=False, precomp=True, num_textures=12,
                     patchsize=64, max_patches=1000):
    """
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
    """

    matdata = loadmat(data_path)
    num_ims = num_textures
    patchsize = patchsize
    if rawdat:
        return matdata
    else:
        # window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
        if precomp:
            precomp_im = np.array(matdata['data'][0][0]['input'][0][0]['dataImg'])
            impatch = precomp_im[:, :, :]
            im_inds = None
        else:
            fn = './data/' + data_path.strip('.mat').split('/')[-1] + '_' + str(num_ims) + '_' +\
                 str(max_patches) + '.h5'

            if os.path.isfile(os.path.abspath(fn)):
                h5f = h5py.File(fn, 'r')
                impatch = np.array(h5f['dat'])
                im_inds = np.array(h5f['inds'])
            else:
                large_ims = np.array(matdata['data'][0][0]['input'][0][0]['imgSample'])
                large_ims = large_ims[:, 0, :, :]
                impatch, im_inds = extract_patches_full(large_ims, patchsize, num_ims, max_patches)
                h5f = h5py.File('./data/' + data_path.strip('.h5').split('/')[-1] + '_' + str(num_ims) + '_' +
                                str(max_patches) + '.h5', 'w')
                h5f.create_dataset('dat', data=impatch, compression='gzip')
                h5f.create_dataset('inds', data=im_inds, compression='gzip')
                h5f.close() 

        return impatch, im_inds, matdata


def aggregate_image_h5py(image_dir=None, imsize=512):
    extensions = ['.jpg', '.gif', '.png', '.tiff']
    numims = 0
    imlist = []
    for ext in extensions:
        imlisttmp = glob.glob(os.path.abspath(image_dir) + '/*' + ext)
        numims += len(imlisttmp)
        imlist += imlisttmp
    print("Num images found = " + str(numims))
    imarr = np.empty((numims, imsize, imsize), np.float32)
    i = 0
    for imname in imlist:
        im = imread(imname, as_gray=True)
        im = resize(im, (imsize, imsize))
        imarr[i, :, :] = im
        i += 1

    dirname = image_dir.split('/')[-1]
    imarrh5 = h5py.File('./data/' + dirname + '.h5', 'w')
    imarrh5.create_dataset('dat', data=imarr, compression="gzip")
    imarrh5.close()


def h5py_dataload(datapath='./data/cropped512-gray-jpg.h5', num_textures=100, patchsize=64, max_patches=1000):
    """
    Extract image patches out of homogeneous textures stored in a
    h5py file numimages x im_height x im_width
    """

    large_ims = h5py.File(datapath, 'r')['dat']

    num_ims = num_textures
    patchsize = patchsize
    fn = './data/' + datapath.strip('.h5').split('/')[-1] + '_' + str(num_ims) + '_' + str(max_patches) + '.h5'
    if os.path.isfile(os.path.abspath(fn)):
        h5f = h5py.File(fn, 'r')
        impatch = np.array(h5f['dat'])
        im_inds = np.array(h5f['inds'])
    else:
        impatch, im_inds = extract_patches_full(large_ims, patchsize, num_ims, max_patches)

        h5f = h5py.File('./data/' + datapath.strip('.h5').split('/')[-1] + '_' + str(num_ims) + '_' +
                        str(max_patches) + '.h5', 'w')
        h5f.create_dataset('dat', data=impatch, compression='gzip')
        h5f.create_dataset('inds', data=im_inds, compression='gzip')
        h5f.close() 

    return impatch, im_inds
