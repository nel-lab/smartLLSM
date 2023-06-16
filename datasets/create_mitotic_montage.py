#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 15:10:26 2023

@author: jimmytabet
"""

import os,glob
import numpy as np
from scipy import ndimage
import matplotlib

path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
files = sorted(glob.glob(os.path.join(path,'**','*.npz'), recursive=True))
no_tiles = len(files)

# bounding box square 192 pixels long centered at 150
bb_size = 200
half_size = bb_size//2
INTEREST = ['anaphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']

for i in INTEREST:
    os.makedirs(os.path.join('/home/nel/Desktop', i), exist_ok=True)

count = 0
runner = 0
for file in files:
    dat = np.load(file, allow_pickle=True)
    raw = dat['raw']
    masks = dat['masks']
    
    num_masks = masks.max()
    
    for idx, mask_id in enumerate(range(1,num_masks+1)):
        
        # check if cell stage is of interest
        if dat['labels'][idx] in INTEREST:
        
            # find center of mass of cell
            center = ndimage.center_of_mass(masks==mask_id)
            center = np.array(center).astype(int)
            
            # original bounding box
            r1_o = center[0]-half_size
            r2_o = center[0]+half_size
            c1_o = center[1]-half_size
            c2_o = center[1]+half_size
            
            # find bounding box indices to fit in tile
            r1 = max(0, center[0]-half_size)
            r2 = min(raw.shape[0], center[0]+half_size)
            c1 = max(0, center[1]-half_size)
            c2 = min(raw.shape[1], center[1]+half_size)
                
            # pad new bounding box with constant value (mean, 0, etc.)
            final = np.zeros([half_size*2, half_size*2], dtype=raw.dtype)
            final += raw[masks==0].mean().astype('int')
        
            # store original bb in new bb
            final[r1-r1_o:r2-r1_o,c1-c1_o:c2-c1_o] = raw[r1:r2,c1:c2]
    
            matplotlib.image.imsave(os.path.join('/home/nel/Desktop', dat['labels'][idx], f'{runner}.pdf'), final, cmap='gray', dpi=300)
            runner += 1

    if count == 0:
        print(f'{count}\tof\t{no_tiles}')
        
    count += 1
    if count%50 == 0:
        print(f'{count}\tof\t{no_tiles}')
        
    if count == no_tiles:
        print(f'{count}\tof\t{no_tiles}\nDONE')
                
