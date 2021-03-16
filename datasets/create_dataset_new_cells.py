#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:22:37 2021

@author: jimmytabet
"""

import os,glob
import numpy as np
from scipy import ndimage

# pick tiles that have not been annotated
path_to_data = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/data_1_cellpose'
tiles = sorted(glob.glob(os.path.join(path_to_data,'**','*.npy'), recursive=True))
tiles = [file for file in tiles if not 'finished' in file]
no_tiles = len(tiles)

# bounding box square 192 pixels long centered at 150
bb_size = 191
half_size = bb_size//2

# loop through each tile
X = []
count = 0
for tile in tiles[:2000]:
    # load image and masks
    dat = np.load(tile, allow_pickle=True).item()
    raw = dat['img']
    masks = dat['masks']
    num_masks = np.max(masks)
    
    # loop through each mask/cell in tile
    for mask_id in range(1,num_masks+1):
        # center of cell
        center = ndimage.center_of_mass(masks==mask_id)
        center = np.array(center).astype(int)
        
        # fix if on edge
        r1 = max(0, center[0]-half_size)
        r2 = min(raw.shape[0], center[0]+half_size+1)
        c1 = max(0, center[1]-half_size)
        c2 = min(raw.shape[1], center[1]+half_size+1)
        
        rfix = half_size*2 - (r2-r1-1)
        cfix = half_size*2 - (c2-c1-1)
        if r1 == 0: r2 += rfix
        if r2 == raw.shape[0]: r1 -= rfix
        if c1 == 0: c2 += cfix
        if c2 == raw.shape[1]: c1 -= cfix   
        
        # isolate cell and append to X
        raw_isolate = raw.copy()
        raw_isolate[masks!=mask_id] = 0

        X.append(raw_isolate[r1:r2,c1:c2])

    # print progress
    if count == 0:
        print(f'{count}\tof\t{no_tiles}')
        
    count += 1
    if count%50 == 0:
        print(f'{count}\tof\t{no_tiles}')
        
    if count == no_tiles:
        print(f'{count}\tof\t{no_tiles}\nDONE')
        
# print end results
X = np.stack(X)
print(X.shape)

# save new_cells
# np.savez('/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/new_cells_2000_tiles.npz', X=X)