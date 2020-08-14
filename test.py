#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 6 21:21:15 2020

@author: lidanyang1995@smu.edu.cn
"""

import os
import cv2
import time 
import torch
import argparse
import numpy    as np
import scipy.io as io
from torch.autograd   import Variable
from torch.utils.data import DataLoader

from utils    import *
from loss     import Loss
from network  import *
from dataset  import *
from SSIM     import *

#%%
parser = argparse.ArgumentParser(description="Testing ResNet3D")
parser.add_argument("--model_path", type=str,   default='model',   help='path to pre-trained model and log files')
parser.add_argument("--data_path",  type=str,   default='data',    help='path of dataset')
parser.add_argument("--save_path",  type=str,   default='output',  help='path of output data')
parser.add_argument("--is_GPU",     type=bool,  default=True,      help='use GPU or not')
parser.add_argument("--gpu_id",     type=str,   default='3',       help='GPU id')
opt = parser.parse_args()

if opt.is_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def main():
    dataset_test = my_Dataset_Test(data_path=opt.data_path)
    loader_test  = DataLoader(dataset=dataset_test, num_workers=4, batch_size=1, shuffle=False)    
    print('Loading model ...')
    model       = ResNet3D()
        
    if opt.is_GPU:
        model  = model.cuda()
        
    model.load_state_dict(torch.load(os.path.join(opt.model_path, 'net_test.pth')))
    model.eval()

    count = 0 
    time_test_preTrain  = 0
    
    for i, (img_noise, img_truth) in enumerate(loader_test, 0): 
            
        img_noise, img_truth = Variable(img_noise), Variable(img_truth)
            
        if opt.is_GPU:
            img_noise, img_truth = img_noise.cuda(), img_truth.cuda()
        
        img_out             = model(img_noise)
        img_denoise         = img_noise - img_out
        count += 1
        
        img_ssim = 0
        img_psnr = 0
        for j in range(img_denoise.shape[4]):
            img_ssim += ssim(img_denoise[:,:,:,:,j], img_truth[:,:,:,:,j])
            img_psnr += batch_PSNR(img_denoise[:,:,:,:,j], img_truth[:,:,:,:,j], 1.)
        img_ssim = img_ssim/(j+1)
        img_psnr = img_psnr/(j+1)
        
        img_denoise = img_denoise.data.cpu().numpy().squeeze()
        img_noise   = img_noise.data.cpu().squeeze().numpy()
        img_truth   = img_truth.data.cpu().squeeze().numpy()
        img_ssim    = img_ssim.data.cpu().numpy()
        
        print('Network image: %sth, SSIM: %.4f, PSNR: %.4f' % (str(i), img_ssim, img_psnr))
            
        img_denoise = nor2CT(img_denoise)
        img_noise   = nor2CT(img_noise)
        img_truth   = nor2CT(img_truth)
        
        # save the reconstructed image
        save_name = ''
        io.savemat(os.path.join(opt.save_path, 'Gen',   str(i)), {save_name.join(['img_denoise']):img_denoise})
        io.savemat(os.path.join(opt.save_path, 'Noise', str(i)), {save_name.join(['img_noise']):img_noise})
        io.savemat(os.path.join(opt.save_path, 'Truth', str(i)), {save_name.join(['img_truth']):img_truth})          
        count = count + 1
    
    print('Done!')
    
#%%
if __name__ == "__main__":
    main()
 
