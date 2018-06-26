import torch
import numpy as np
from v2pca import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
from config import *
from sklearn.feature_extraction.image import extract_patches_2d

def matfile_dataload(data_path='../data/bundleClassify_64_12_1.mat', precomp = True):
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
    window = torch.tensor(np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights']), device=device, dtype=dtype).to(device)
    if precomp:
        precomp_im = np.array(matdata['data'][0][0]['input'][0][0]['dataImg'])
        impatch = precomp_im[:,:,:]
    else:
        num_ims = 12
        patchsize = 64
        impatch = np.empty((0,imsize*imsize), float)
        for i in range(num_ims):
            ims = extract_patches_2d(large_ims[0,0,:,:].reshape([256,256]), (64,64), max_patches = 0.02)
            ims = np.reshape(ims, [ims.shape[0], ims.shape[1]*ims.shape[1]])
            normed_ims = (ims - ims.mean(axis=1, keepdims=1)/ims.std(axis=1, keepdims=1)
            impatch = np.vstack((impatches, normed_ims))

        impatch = impatch.reshape(-1, patchsize, patchsize)

    return impatch, window

def patch_extract(image_dir = '../../dat/cropped512-gray-jpg'):
    '''
    extract normalized image patches from jpeg image dir
    '''

    large_i
    


network1 = V2PCA(imgSize=im.shape[1], K=4, N=2, nonlin='quadratic', window = window, pcaMat = None, ret_pyr = True, ncomp=32)


# In[4]:


x = im.reshape([im.shape[0],1,im.shape[1],im.shape[2]])
x = torch.tensor(x, requires_grad=True, dtype = dtype,device = device)
coeff,pyr,pind  = network1(x)
from scipy.io import savemat
a = {}
a['pyrdata'] = pyr.data.cpu().numpy()
savemat('pyrdata', a)


# In[ ]:


coeffnump = coeff.data.cpu().numpy()
'''
coeffsmall = np.zeros((504,3944))
for i in range(coeffsmall.shape[0]):
    row = coeffnump[i,:]
    coeffsmall[i,:] = row[np.nonzero(row)]
'''
def PCA(X):
    X_mean = torch.mean(X,1, True)
    X = X - X_mean.expand_as(X)
    U,S,V = torch.svd(X)
    return V

#coeff = Variable(torch.Tensor(coeff).cuda())

V = PCA(coeff)
pcamat = V.data.cpu().numpy()
print(pcamat.shape)
from scipy.io import savemat
a = {}
#a['pcadata'] = A_
a['pcadata'] = pcamat
savemat('data', a)


# In[ ]:


res = torch.mm(coeff, V)
np.argmax(res[:,2].data.cpu().numpy())


# In[ ]:


network2 = V2PCA(imgSize=im.shape[1], K=4, N=2, nonlin='quadratic', window=window, pcaMat = pcaMat, ncomp=32)


# In[ ]:


from torch.optim import SGD
x = im[16,:,:].reshape([1,1,im.shape[1],im.shape[2]])
x = torch.Tensor(x)
x = x.cuda()
x = Variable( x, requires_grad=True ) 

def run_grad_op(model, image):
    imagelist = []
    optimizer = SGD([image],lr=6)
    for i in range(5):
        def closure():
            #optimizer.zero_grad()
            output = network2(image)
            loss = output[0,2]
            loss.backward()
            return loss
        loss=optimizer.step(closure)
        imagelist.append(x.clone())
    return imagelist

outputim = run_grad_op(network2, x)


# In[ ]:





# In[ ]:


def window_im(xorig, win):
    new_im = np.multiply(xorig.reshape([4096,1]), win[4096:8192].data.cpu().numpy())
    return new_im.reshape([64,64])

xwin = window_im(im[16,:,:], window)
plt.figure()
plt.imshow(xwin.reshape([64,64]), cmap='gray')
plt.axis('off')


# In[ ]:


for i in range(len(outputim)):
    plt.figure()
    outputim2 = outputim[i].data.cpu().numpy().reshape([64,64])
    outputwin = window_im(outputim2, window)
    plt.imshow(outputwin, cmap='gray')
    plt.axis('off')


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        ax.tick_params(labelbottom=False, labelleft=False)
f = plt.figure(figsize=(10,8))
K = 4
nR = int(np.ceil(K/2))
nC = K+1-H
print(W,H)
gs0 = gridspec.GridSpec(nR, nC)
gs0.update(wspace=0.025, hspace=0.025) # set the spacing between axes. 

img = y[1].data.select(0,0).select(0,0)           
img = img.view(-1, img.size(-2), img.size(-1) )
img2 = y[2].data.select(0,0).select(0,0)
img2 = img2.view(-1, img.size(-2), img.size(-1))

for i in range(img.size(0)):
    if i < H:
        print(i, W-1)
        loc = gs0[i,-1]
    else:
        print(-1, W-i)
        loc = gs0[-1,W-i]
    ax1 = plt.Subplot(f, loc)
    ax1 = f.add_subplot(ax1)
    im = img[i]
    scale=127.5/max(im.max(), -im.min())
    im =im.mul(scale).add(127.5)
    ax1.imshow(im, cmap='gray')

N=2
for k in range(N):
    inner_grid = gridspec.GridSpecFromSubplotSpec(3, 3,
            subplot_spec=outer_grid[i], wspace=0.0, hspace=0.0)


make_ticklabels_invisible(f)
'''
for i in range()
gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])

ax1 = plt.Subplot(f, gs00[-1, -1])
f.add_subplot(ax1)
ax2 = plt.Subplot(f, gs00[-1, -2])
f.add_subplot(ax2)
ax3 = plt.Subplot(f, gs00[-1, 0])
f.add_subplot(ax3)
make_ticklabels_invisible(f)'''


# In[ ]:


len(y)-


# In[ ]:




fig.show()
from PIL import Image
img = y[1].data.select(0,0).select(0,0)           
img = img.view(-1, img.size(-2), img.size(-1) )
print( i, img.size() )
for j in range(img.size(0)):
    plt.figure()
    plt.imshow(img[j], cmap='gray')

