#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:40:42 2021

@author: jimmytabet
"""

import glob
import numpy as np
import matplotlib.pyplot as plt

fls = glob.glob('/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/Cellpose_annotated_masks/*.npy')
bad = '/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/Cellpose_annotated_masks/Data1_Position6_Scan_Iter_0000_0005_metaphase_seg.npy'

fls = [fl for fl in fls if fl != bad]

first = ['/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/Cellpose_annotated_masks/Data1_Position5_Scan_Iter_0007_0007_metaphase_seg.npy',
         '/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/Cellpose_annotated_masks/Data1_Position2_Scan_Iter_0009_0001_metaphase_seg.npy',
         '/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/Cellpose_annotated_masks/Data1_Position2_Scan_Iter_0009_0009_metaphase_seg.npy',
         '/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/Cellpose_annotated_masks/Data1_Position4_Scan_Iter_0004_0009_metaphase_seg.npy']

X = []
y = []
for fl in fls:
    metafirst = 0
    metasecond = 0
    label = fl.split('_')[-2].split('.')[0].lower()
    data = np.load(fl, allow_pickle=True).item()

    raw = data['img']

    masks = data['masks']

    num_masks = np.max(masks)

    plt.close('all')


    if num_masks == 0:            
        # plt.imshow(raw, cmap='gray')
        # plt.pause(.2)
        X.append(raw)
        y.append(label)
        continue
    
    elif label == 'anaphase':
        start = 1
        end = num_masks+1
    
    elif label == 'metaphase':
        if num_masks == 2:
            if fl in first:
                metafirst = 1
                start = 1
                end = 2
            else:
                metasecond = 1
                start = 1
                end = 2
        else:
            start = 1
            end = num_masks+1

    elif label == 'prophase':        
        if num_masks==2:
            start = 1
            end = num_masks
        else:
            start = 1
            end = num_masks+1 
        
            
    elif label == 'telophase':
        if num_masks==3:
            start = 2
            end = num_masks+1 

        else:
            start = 1
            end = num_masks+1 

    else:
        print('something messed up')
    
    for i in range(start, end):
        # fig,ax = plt.subplots(1,2)
        # ax[0].imshow(raw, cmap='gray')
        raw_isolate = raw.copy()
        if metafirst:
            raw_isolate[masks!=1] = 0

        elif metasecond:
            raw_isolate[masks!=2] = 0
            
        else:
            raw_isolate[masks!=i] = 0
            
        # ax[1].imshow(raw_isolate, cmap='gray')
        # plt.pause(.2)
        X.append(raw_isolate)
        y.append(label)

np.savez('annotated_isolate', X=X, y=y)