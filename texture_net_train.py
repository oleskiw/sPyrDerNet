import torch
import os, sys
import h5py
from models.genutils import *
from config import *
from dimred import *
import numpy as np
from models.v2net import *
from models.objectives import covratio_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import *
from data_loader import *
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
import seaborn as sb
import pandas as pd
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
np.set_printoptions(suppress=True)

np.random.seed(5)

def compute_componentmat(savepath = './output/',datapath = None, dimreduce='pca', use_window=True, num_textures=10, max_patches=800):
    """
    can choose between pca, ica, or zca for dimreduce option
    returns matrix of size num_coefficients (output of steerable pyramid and rectification) x num_components
    """

    matdata = matfile_dataload(rawdat = True)
    impatch, im_inds = h5py_dataload(datapath = datapath, num_textures=num_textures, max_patches=max_patches)
    torch.cuda.empty_cache()
    if use_window:
        window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
        windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
        windowinds = np.nonzero(windowbool)[0]
        print(window.shape)
        window = torch.tensor(window, dtype=dtype).to(device)
    else:
        window = None
    del matdata
    

    printout("Data loaded, impatch shape:" +  str(impatch.shape))
    printout("Building pyramid...")
    batchsize = 20000
    network = V2PCA(imgSize=impatch.shape[1], K=4, N=2, nonlin='smoothabs', window = None, pcaMat = None, ret_pyr = False, ncomp=10)
    for batch_i in range(int(impatch.shape[0] / batchsize)):
        impatchtmp = impatch[batch_i*batchsize : (batch_i+1)*batchsize,:,:]
        x = impatchtmp.reshape([batchsize, 1, impatch.shape[1], impatch.shape[2]])
        x = torch.tensor(x, dtype=dtype).to(device)
        del impatchtmp
        coeffbatch = network(x)
        if use_window:
            coeffbatch = coeffbatch[:,windowinds]

        if batch_i == 0:
            coeff = coeffbatch
        else:
            coeff = torch.cat((coeff, coeffbatch),0)

        del coeffbatch
        print("Coeff batch: " + str(batch_i))
    print(coeff.size())

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
        compmat = np.zeros((window.size(0), compmatold.shape[1]))+1e-8
        compmat[windowinds,:] = compmatold

    printout(compmat.shape)
    dataset_name = datapath.split('/')[-1].rstrip('.h5')
    h5f = h5py.File(savepath + dimreduce + '_' + dataset_name + '_' + str(num_textures) + '_' + str(max_patches) + '_'  + 'comp.h5', 'w')
    h5f.create_dataset('dat', data=compmat, compression='gzip')
    h5f.close()
    del compmat
    del coeff

def compute_v2_covratio(savepath = None, datapath = None, num_textures = 50, max_patches=4000):
    '''
    takes the spyr coeff output and solves 
    the generalized eigenvector problem to maximize the cov ratio
    '''
    matdata = matfile_dataload(rawdat = True)
    window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
    window = torch.tensor(window, dtype=dtype).to(device)
    dataset = TextureCoeffDataset(datapath, num_textures=num_textures, max_patches=max_patches, transform=ToCoeffTensor())


    dataloader = DataLoader(dataset, batch_size = num_textures*max_patches, shuffle = True)
    print('Data loaded...')

    for i_batch, sample_batched in enumerate(dataloader):
        components = sample_batched['coeff']
        loss,projmat = covratio_loss(components, sample_batched['textureind'])    
    windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
    windowinds = np.nonzero(windowbool)[0]
    
    x = np.zeros((len(windowbool), 10)) + 1e-8
    x[windowinds,:] = projmat

    dataset_name = datapath.split('/')[-1].rstrip('.h5')
    h5f = h5py.File(savepath + 'geneigtransform' + '_' + dataset_name + '_' + str(num_textures) + '_' + str(max_patches) + '_'  + 'comp.h5', 'w')
    h5f.create_dataset('dat', data=x, compression='gzip')
    h5f.close()

def train_v2net(savepath = None, datapath = None, inptype = 'coeff', num_textures = 50, max_patches=4000):
   
    matdata = matfile_dataload(rawdat = True)
    window = np.array(matdata['data'][0][0]['f'][0][0]['coeffWeights'])
    window = torch.tensor(window, dtype=dtype).to(device)
    if inptype == 'image':
        dataset = TextureImageDataset(datapath, num_textures=num_textures, max_patches=max_patches, transform=ToImageTensor())
    elif inptype == 'coeff':
        dataset = TextureCoeffDataset(datapath, num_textures=num_textures, max_patches=max_patches, transform=ToCoeffTensor())


    dataloader = DataLoader(dataset, batch_size = num_textures*max_patches, shuffle = True)
    print('Data loaded...')
    model = V2Net(imgSize = 64, K=4, N=2, includeHF=True, inp = inptype, nonlin = 'smoothabs', window = window, ret_pyr = False, ncomp = 10, out_activ = True).to(device)
    optimizer = Adam(model.parameters(), lr = 0.1)
    loss_prev = 0
    iter_no_decrease = 0
    
    for epochs in range(5000):
        for i_batch, sample_batched in enumerate(dataloader):
            components =  model(sample_batched[inptype])
            loss, _ = covratio_loss(components, sample_batched['textureind'])
            print("Loss: ", loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss.data.cpu().numpy() < loss_prev - 0.001*np.abs(loss_prev):
            print("Loss prev, loss:", loss_prev, loss.data.cpu().numpy())
            loss_prev = loss.data.cpu().numpy()
            iter_no_decrease = 0
        else:
            iter_no_decrease += 1
            if iter_no_decrease == 5:
                break

    lin_trans_weights = model.lintrans.weight.data.cpu().numpy() 
    windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
    windowinds = np.nonzero(windowbool)[0]
    x = np.zeros((len(windowbool),10)) + 1e-8
    x[windowinds,:] = np.transpose(lin_trans_weights)
    dataset_name = datapath.split('/')[-1].rstrip('.h5')
    h5f = h5py.File(savepath + 'sgdgeneigtransform' + '_' + dataset_name + '_' + str(num_textures) + '_' + str(max_patches) + '_'  + 'comp.h5', 'w')
    h5f.create_dataset('dat', data=x, compression='gzip')
    h5f.close()
    

def compute_test_components(weightfile, savepath, datasource, num_textures, max_patches, out_activ=True):
    h5f = h5py.File(savepath + weightfile, 'r')
    weights = h5f['dat'][()]
    if weights.shape[0] < weights.shape[1]:
        weights = np.transpose(weights)

    weights = weights[:, 0:10]
    
    matdata = matfile_dataload(rawdat = True)
    windowbool = np.array(matdata['data'][0][0]['f'][0][0]['coeffIndex'])[:,0]
    windowinds = np.nonzero(windowbool)[0]
    weights = weights[windowinds,:]
    weights = torch.tensor(weights, dtype=dtype).to(device)

    
    coeffdata = TextureCoeffDataset(datasource, num_textures=num_textures, max_patches=max_patches, transform=ToCoeffTensor())
    
    testsize=10000
    batchsize = testsize
    compmat = np.zeros((testsize, weights.size(1)))
    inds = np.zeros(testsize)


    dataloader = DataLoader(coeffdata, batch_size = batchsize, shuffle = False)

    for i_batch, sample_batched in enumerate(dataloader):

        if (i_batch+1)*batchsize <= testsize:
            comp = torch.mm(sample_batched['coeff'], weights)
            if out_activ:
                comp = smoothabs_activ(comp)
            compmat[i_batch*batchsize:(i_batch+1)*batchsize, :] = comp.data.cpu().numpy()
            inds[i_batch*batchsize:(i_batch+1)*batchsize] = sample_batched['textureind']
    weightname = weightfile.rstrip('.h5').split('-')[0]
    h5f = h5py.File(savepath + weightname + '_' + 'componenttest' + '_' + str(num_textures) + '_' + str(max_patches) + '_' + str(testsize) + '.h5', 'w')
    h5f.create_dataset('dat', data = compmat, compression = 'gzip')
    h5f.create_dataset('inds', data = inds, compression = 'gzip')
    h5f.close()
     
def summary_plot(componentfile,datafile, savepath, iminds = [], sephist = False, clusters = None):
    n_comp = 8
    ncols = int(n_comp/2)
    nrows = 2
    h5f = h5py.File(savepath + componentfile, 'r')
    components = h5f['dat'][()]
    components = components[:,0:n_comp]
    imarr = h5py.File(datafile,'r')['dat'][0:10000]
    inds = h5py.File(datafile, 'r')['inds'][0:10000]
    imlist = iminds
    colors = [1,2,3,4,5,6]
    colorval = 0
    for i in range(len(imlist)):
        colorval +=1
        n = imlist[i]
        indsval = np.where(inds== n)[0]
        xout = components[indsval, :]
        colorvals = [colors[colorval-1] for i in range(len(indsval))]
        if i == 0:
            xarr = xout
            colorinds = colorvals
        else:
            xarr = np.concatenate((xarr, xout), axis = 0)
            colorinds.extend(colorvals)

    if sephist:
        fig, axs = plt.subplots(nrows, ncols, tight_layout = True, figsize=(30,20))
        for i in range(nrows*ncols):
            axs[int(np.floor(i/ncols)),int(i%ncols)].hist(components[:,i], bins=20)
            axs[int(np.floor(i/ncols)), int(i%ncols)].set_title("Component " + str(i+1))

        plt.savefig(savepath + componentfile.rstrip('.h5') + '_hist' + '.png')
        plt.close()

    labels = ['Component ' + str(i) for i in range(1,n_comp+1)]
    print(xarr.shape)
    d = pd.DataFrame(data = xarr, columns = labels)
    if clusters is not None:
        d.loc[:,'clust'] = pd.Series(colorinds)
        pairplot = sb.pairplot(d,diag_kind = 'hist', plot_kws={'alpha':0.2}, hue ='clust')
    else:
        pairplot = sb.pairplot(d, diag_kind = "kde", plot_kws={'alpha':0.2})
    for i,j in zip(*np.triu_indices_from(pairplot.axes,1)):
        pairplot.axes[i,j].set_visible(False) 
    if clusters is not None:
        pairplot.savefig(savepath + componentfile.rstrip('.h5') + '_pairplotclust' + '.png')
    else:
        pairplot.savefig(savepath + componentfile.rstrip('.h5') + '_pairplot' + '.png')



def cluster_patches(componentfile, savepath):
    h5f = h5py.File(savepath + componentfile, 'r')
    components = h5f['dat'][()]
    #n_clusters = silhouette_analyze(components, savepath, componentfile)
    n_clusters = 4
    clusterer = KMeans(n_clusters = n_clusters, random_state = 10)
    cluster_labels = clusterer.fit_predict(components)
    h5f = h5py.File(savepath + componentfile.rstrip('.h5') + '_clusterinds.h5', 'w')
    h5f.create_dataset('dat', data = cluster_labels, compression = 'gzip')
    h5f.close()

    return cluster_labels


def silhouette_analyze(X, savepath, componentfile):

    range_n_clusters = [2,3,4,5,6]
    sil_scores = [0 for i in range(len(range_n_clusters))]    
    j = 0
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
        sil_scores[j] = silhouette_avg
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        j +=1
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        
        plt.savefig(savepath + componentfile.rstrip('.h5') + '_silhouette_' + str(n_clusters) + '.png')
        plt.close()
            


    ind = np.argmax(sil_scores)
    n_cluster_init = range_n_clusters[ind]

    return n_cluster_init




    





def main():

    #getting component transformations from pca or geneig
    datapath1 = '/scratch/np1742/texture-modeling/dat/cropped384-gray-jpg.h5'
    datapath2 = '/scratch/np1742/texture-modeling/dat/cropped512-gray-jpg.h5'
    #compute_componentmat(datapath = datapath1, dimreduce = 'pca', use_window = True, num_textures = 40, max_patches=4000)
    #torch.cuda.empty_cache()
    #compute_componentmat(datapath = datapath2, dimreduce = 'pca', use_window = True, num_textures = 40, max_patches=4000)
    #torch.cuda.empty_cache()
#    train_v2net(savepath = './output/', datapath = datapath1, inptype = 'coeff', num_textures = 40, max_patches = 4000)
 #   torch.cuda.empty_cache()
  #  train_v2net(savepath = './output/', datapath = datapath2, inptype = 'coeff', num_textures = 40, max_patches = 4000)
   # torch.cuda.empty_cache()
    #compute_v2_covratio('./output/', datapath1, 40, 4000)
    #torch.cuda.empty_cache()
    #compute_v2_covratio('./output/', datapath2, 40, 4000)
    #h5py_coeff_dataload('./data/cropped512-gray-jpg.h5', num_textures = 200, max_patches = 1000) 
    compute_test_components('sgdgeneigtransform_cropped384-gray-jpg_40_4000_comp.h5', './output/',datapath1, 40, 4000)
    compute_test_components('geneigtransform_cropped384-gray-jpg_40_4000_comp.h5', './output/',datapath1, 40, 4000, out_activ = False)
    compute_test_components('pca_cropped384-gray-jpg_40_4000_comp.h5', './output/',datapath1, 40, 4000, out_activ = False)
    compute_test_components('sgdgeneigtransform_cropped512-gray-jpg_40_4000_comp.h5', './output/',datapath2, 40, 4000)
    compute_test_components('geneigtransform_cropped512-gray-jpg_40_4000_comp.h5', './output/',datapath2, 40, 4000, out_activ = False)
    compute_test_components('pca_cropped512-gray-jpg_40_4000_comp.h5', './output/',datapath2, 40, 4000, out_activ = False)
    #summary_plot('componenttest_200_1000_10000.h5', './output/')
#    cluster_inds = cluster_patches('componenttest_200_1000_10000.h5', './output/')
#    cluster_inds = cluster_inds + 1
#    cluster_inds = np.array(['Clust '+ str(x) for x in cluster_inds])
     #summary_plot('componenttest_200_1000_10000.h5','/scratch/np1742/texture-modeling/dat/cropped512-gray-jpg_200_1000.h5','./output/', iminds=[0,2,3,8,27,67], clusters=1)
     #summary_plot('p_componenttest_200_1000_10000.h5', '/scratch/np1742/texture-modeling/dat/cropped512-gray-jpg_200_1000.h5', './output/', iminds=[0,2,3,8,27,67], clusters=1)

if __name__== "__main__":
    main()






