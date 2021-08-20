#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 12:43:28 2021

@author: jimmytabet
"""

#%% load data
import numpy as np

with np.load('/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/annotated_fov286_0820.npz') as f:
    X = f['X']
    y = f['y']

#%% create images - ~3.5 mins, 25 GB
import os
import matplotlib
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

base_folder = '/home/nel/Desktop/cell data andrea'

# stages = np.unique(y)
# stages = ['TBD','anaphase','early_prophase','metaphase','other','prometaphase','prophase','telophase']
stages = ['interphase','blurry','anaphase','early_prophase','metaphase','prometaphase','prophase','telophase']

for stage in stages:
    # mask cells
    dat = X[y==stage]
    
    randperm = np.random.permutation((y==stage).sum())
    dat = dat[randperm]
    dat = dat[randperm]
    
    # make tif folder
    if not os.path.isdir(os.path.join(base_folder, 'tif', stage)):
        os.makedirs(os.path.join(base_folder, 'tif', stage))

    # make pdf folder
    if not os.path.isdir(os.path.join(base_folder, 'pdf', stage)):
        os.makedirs(os.path.join(base_folder, 'pdf', stage))
    
    # save data
    for count, i in enumerate(dat.squeeze()):
        matplotlib.image.imsave(os.path.join(base_folder, 'tif', stage, f'{count}.tiff'), i, cmap='gray')
        matplotlib.image.imsave(os.path.join(base_folder, 'pdf', stage, f'{count}.pdf'), i, cmap='gray')
        if count == 49:
            break