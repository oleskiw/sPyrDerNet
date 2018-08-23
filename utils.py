import os,sys
import numpy as np
"""
Utility functions for main.py
"""


def printout(string):
    print(string)
    sys.stdout.flush()


def window_im(xorig, win):
    imflat_size = xorig.shape[0]*xorig.shape[1]
    new_im = np.multiply(xorig.reshape([imflat_size,1]), win[imflat_size:imflat_size*2].data.cpu().numpy())
    return new_im.reshape(xorig.shape)
    
