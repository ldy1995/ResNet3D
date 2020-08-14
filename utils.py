 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 6 16:38:09 2020

@author: lidanyang1995@smu.edu.cn
"""

import math
import torch
import re
import pylab
import torch.nn as nn
import scipy.io as io
import numpy as np
from torch.autograd import Variable
from   skimage.measure.simple_metrics import compare_psnr
import os
import glob 

HU_min   = -1024
HU_max   = 1500
mu_water = 0.0192 

def findLastCheckpoint(save_dir):
    
    file_list = glob.glob(os.path.join(save_dir, '*iter*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*iter(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
            
    return initial_epoch


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]): # for there is only one channel image
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])


def mu2nor(input):  
    
    # mu -> CT
    input_HU = (input - mu_water)/mu_water * 1e3
    
    # CT -> nor
    input_nor = (input_HU - HU_min)/(HU_max - HU_min)
    
    return input_nor
    
def CT2nor(input):

    # CT -> nor
    input_nor = (input - HU_min)/(HU_max - HU_min)
    
    return input_nor

def nor2CT(input):  
    
    # CT -> nor
    input_CT = input*(HU_max - HU_min) + HU_min
    
    return input_CT


if __name__ == '__main__':
    
    
    read_path = '../../my_data/mayo_data_img/'
    pat       = 'L067_0.25dose_proj_par_3mm_fbp.mat'
    win       = [44,44,24]
    stride    = [1, 1, 1]
    
    img_data  = io.loadmat(os.path.join(read_path, pat))['img_par']
    
    print('Done!')

                
                