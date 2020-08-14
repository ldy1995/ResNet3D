#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 6 20:17:25 2020

@author: lidanyang1995@smu.edu.cn
"""

import os
import cv2
import glob
import h5py
import torch
import random
import os.path
import numpy            as np
import torch.utils.data as udata
from   utils import *

class my_Dataset_Train(udata.Dataset):
    def __init__(self, data_path='.'):
        super(my_Dataset_Train, self).__init__()

        self.data_path = data_path
        print('Loading training dataset ...')
        self.target_path = os.path.join(self.data_path, 'target_img_training.h5')
        self.input_path  = os.path.join(self.data_path, 'input_img_training.h5')

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f = h5py.File(self.input_path, 'r')

        self.keys = list(target_h5f.keys())
        #random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f  = h5py.File(self.input_path, 'r')

        key    = str(index)
        
        target = mu2nor(np.array(target_h5f[key], dtype = 'float32'))
        input  = mu2nor(np.array(input_h5f[key],  dtype = 'float32'))

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input).unsqueeze(0), torch.Tensor(target).unsqueeze(0)


class my_Dataset_Test(udata.Dataset):
    def __init__(self, data_path='.'):
        super(my_Dataset_Test, self).__init__()

        self.data_path = data_path
        print('Loading testing dataset ...')
        self.target_path = os.path.join(self.data_path, 'target_img_testing.h5')
        self.input_path  = os.path.join(self.data_path, 'input_img_testing.h5')
        
        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f  = h5py.File(self.input_path, 'r')

        self.keys = list(target_h5f.keys())
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_h5f = h5py.File(self.target_path, 'r')
        input_h5f  = h5py.File(self.input_path, 'r')

        key    = str(index)
        target = mu2nor(np.array(target_h5f[key], dtype = 'float32'))
        input  = mu2nor(np.array(input_h5f[key],  dtype = 'float32'))
        
        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input).unsqueeze(0), torch.Tensor(target).unsqueeze(0)


# %%
if __name__ == '__main__':
    
    data_path  = 'data'
    
    train_dataset = my_Dataset_Train(data_path)
    test_dataset  = my_Dataset_Test(data_path)
    
    for i, (img_noise, img_truth) in enumerate(test_dataset, 0):
        
        
        print('Done!')
