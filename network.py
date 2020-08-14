#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 6 18:12:04 2020

@author: lidanyang1995@smu.edu.cn
"""

import torch
import torch.nn as nn
from   torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2
import random
import time
import os


class base_block(nn.Module):
    def __init__(self, in_ch, out_ch, ksize = (3,3,3), stride = (1, 1, 1),
                 padding = (1,1,1), layer_norm = nn.BatchNorm3d ,act_f = nn.ReLU):
        super(base_block, self).__init__()
        
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.ksize   = ksize
        self.stride  = stride
        self.padding = padding
        
        conv3d   = nn.Conv3d(self.in_ch, self.out_ch, kernel_size = self.ksize, 
                             stride = self.stride, padding = self.padding)
        BN       = layer_norm(self.out_ch)
        act_f    = act_f(inplace=True)
        self.net = nn.Sequential(conv3d, BN, act_f)
        
    def forward(self, input):
        
        return self.net(input)

class base_block1(nn.Module):
    def __init__(self, in_ch, out_ch, ksize = (3,3,3), stride = (1, 1, 1),
                 padding = (1,1,1), layer_norm = nn.BatchNorm3d ,act_f = nn.ReLU):
        super(base_block1, self).__init__()
        
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.ksize   = ksize
        self.stride  = stride
        self.padding = padding
        
        conv3d   = nn.Conv3d(self.in_ch, self.out_ch, kernel_size = self.ksize, 
                             stride = self.stride, padding = self.padding)
        BN       = layer_norm(self.in_ch)
        act_f    = act_f(inplace=True)
        self.net = nn.Sequential(BN, act_f, conv3d)
        
    def forward(self, input):
        
        return self.net(input)

class base_ResNet(nn.Module):
    def __init__(self, in_ch, temp_ch = 64, ksize = (3,3,3), ksize_temp = (3,3,1), stride = (1, 1, 1),
                 padding = (1,1,1), padding_temp = (1,1,0), layer_norm = nn.BatchNorm3d ,act_f = nn.ReLU, n_layer = 2):
        super(base_ResNet, self).__init__()
        
        self.in_ch   = in_ch
        self.ksize   = ksize
        self.stride  = stride
        self.padding = padding
        
        self.temp_ch      = temp_ch
        self.ksize_temp   = ksize_temp
        self.padding_temp = padding_temp
        
        conv3d = nn.Conv3d(self.in_ch, self.temp_ch, kernel_size = self.ksize, 
                           stride = self.stride, padding = self.padding)
        conv3d_temp = nn.Conv3d(self.temp_ch, self.temp_ch, kernel_size = self.ksize_temp, 
                                stride = self.stride, padding = self.padding_temp)
        BN          = layer_norm(self.in_ch)
        BN_temp     = layer_norm(self.temp_ch)
        act_f       = act_f(inplace=True)
        
        self.net = nn.Sequential(BN, act_f, conv3d, BN_temp, act_f, conv3d_temp)
        
    def forward(self, input):
        
        temp = self.net(input)
        
        return temp + input
      
class ResNet3D(nn.Module):
    def __init__(self, in_ch = 1, out_ch = 1, temp_ch = 64, ksize = (3,3,3), ksize_temp = (3,3,1),
                 stride = (1, 1, 1), padding = (1,1,1), padding_temp = (1,1,0),
                 layer_norm = nn.BatchNorm3d ,act_f = nn.ReLU):
        super(ResNet3D, self).__init__()
        
        self.in_ch   = in_ch
        self.out_ch  = out_ch
        self.ksize   = ksize
        self.stride  = stride
        self.padding = padding
        
        self.temp_ch      = temp_ch
        self.ksize_temp   = ksize_temp
        self.padding_temp = padding_temp
        
        self.c1_conv = base_block(self.in_ch,   self.temp_ch, ksize = self.ksize, padding = self.padding) 
        self.c2_conv = base_block(self.temp_ch, self.temp_ch, ksize = self.ksize_temp, padding = self.padding_temp)
        self.c3_conv = base_block(self.temp_ch, self.temp_ch, ksize = self.ksize, padding = self.padding)
        
        self.c4_conv = nn.Conv3d(self.temp_ch, self.temp_ch, 
                                 kernel_size = self.ksize_temp, stride = self.stride,
                                 padding = self.padding_temp)
        
        self.res1 = base_ResNet(self.temp_ch, self.temp_ch, 
                                ksize = self.ksize, ksize_temp = self.ksize_temp,
                                padding = self.padding, padding_temp = self.padding_temp)
        self.res2 = base_ResNet(self.temp_ch, self.temp_ch, 
                                ksize = self.ksize, ksize_temp = self.ksize_temp,
                                padding = self.padding, padding_temp = self.padding_temp)
        self.res3 = base_ResNet(self.temp_ch, self.temp_ch, 
                                ksize = self.ksize, ksize_temp = self.ksize_temp,
                                padding = self.padding, padding_temp = self.padding_temp)
        self.c10_conv = base_block1(self.temp_ch, self.temp_ch, ksize = self.ksize, padding = self.padding) 
        self.c11_conv = base_block1(self.temp_ch, self.out_ch, ksize = self.ksize_temp, padding = self.padding_temp)
        
    def forward(self, input):
        
        temp = self.c1_conv(input)
        temp = self.c2_conv(temp)
        temp = self.c3_conv(temp)
        temp = self.c4_conv(temp)
        temp = self.res1(temp)
        temp = self.res2(temp)
        temp = self.res3(temp)
        temp = self.c10_conv(temp)
        out  = self.c11_conv(temp)
        
        return out

                               
        
if __name__ == '__main__':
    
    
    
    in_ch   = 1
    out_ch  = 64
    ksize   = (3,3,3)
    padding = (1,1,1) 
    stride  = (1,1,1)
    
    input_data = Variable(torch.zeros(1,1,44,44,24), requires_grad = True).cuda()
    
    model1 = base_block(in_ch, out_ch, ksize = ksize, stride = stride, padding = padding)
    model2 = base_ResNet(in_ch, out_ch, ksize = ksize, stride = stride, padding = padding)
    model3 = ResNet3D()
    
    model1.cuda()
    model2.cuda()
    model3.cuda()
    
    out1 = model1(input_data)
    out2 = model2(input_data)
    out3 = model3(input_data)
    
    print('Done!')
    
   
        