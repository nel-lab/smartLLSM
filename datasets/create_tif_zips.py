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

#%% create tiff zips - ~3.5 mins, 25 GB
import os
import matplotlib

base_folder = '/home/nel/Desktop/cell data'

for stage in ['TBD','anaphase','early_prophase','metaphase','other','prometaphase','prophase','telophase']:#np.unique(y):
    # mask cells
    dat = X[y==stage]
    
    # make folder
    if not os.path.isdir(os.path.join(base_folder, stage)):
        os.makedirs(os.path.join(base_folder, stage))
    
    # save data
    for count, i in enumerate(dat.squeeze()):
        matplotlib.image.imsave(os.path.join(base_folder, f'{stage}/{count}.tiff'), i, cmap='gray')