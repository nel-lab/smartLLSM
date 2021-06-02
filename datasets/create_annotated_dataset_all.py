#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:56:52 2021

@author: jimmytabet
"""

#%% imports
import os,glob
import numpy as np
from scipy import ndimage

#%% new annotated data
path = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
files = sorted(glob.glob(os.path.join(path,'**','*.npz'), recursive=True))
files = [file for file in files if not 'og_backup' in file]
no_tiles = len(files)

# bounding box square 192 pixels long centered at 150
bb_size = 286
half_size = bb_size//2

X = []
y = []
count = 0
for file in files:
    dat = np.load(file, allow_pickle=True)
    # print(dat.files)
    labels_dict = dat['labels_dict'].item()
    # labels.append([labels_dict[j] for j in dat['labels']])

    raw = dat['raw']
    masks = dat['masks']
    
    num_masks = masks.max()
    
    for idx, mask_id in enumerate(range(1,num_masks+1)):
        
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
        
        # # OLD - MOVE CELL OFF CENTER INSTEAD OF PADDING
        # rfix = half_size*2 - (r2-r1)
        # cfix = half_size*2 - (c2-c1)
        # if r1 == 0: r2 += rfix
        # if r2 == raw.shape[0]: r1 -= rfix
        # if c1 == 0: c2 += cfix
        # if c2 == raw.shape[1]: c1 -= cfix   
        
        # # copy raw to save isolated cell
        # raw_isolate = raw.copy()
        # raw_isolate[masks!=mask_id] = 0
        # X.append(raw_isolate[r1:r2,c1:c2])

        # pad new bounding box with constant value (mean, 0, etc.)
        final = np.zeros([half_size*2, half_size*2], dtype=raw.dtype)
        final += raw[masks==0].mean().astype('int')
    
        # store original bb in new bb
        final[r1-r1_o:r2-r1_o,c1-c1_o:c2-c1_o] = raw[r1:r2,c1:c2]

        X.append(final)
        y.append(labels_dict[dat['labels'][idx]])

    if count == 0:
        print(f'{count}\tof\t{no_tiles}')
        
    count += 1
    if count%50 == 0:
        print(f'{count}\tof\t{no_tiles}')
        
    if count == no_tiles:
        print(f'{count}\tof\t{no_tiles}\nDONE')
        
#%%
X = np.stack(X)
y = np.array(y)
print(X.shape)
print(y.shape)

#np.savez('/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update_no_isolate_286_5_6.npz', X=X, y=y)

#%%
from collections import Counter
 
stages_dict = dict(Counter(y))
stages_dict = dict(sorted(stages_dict.items(), key = lambda item: item[1], reverse = True))

no_cells = len(y)

print(f'RESULTS OF {no_tiles} TILES:\n')
for k,v in stages_dict.items():
    if len(k) < 7:
        print(f'{k}: \t\t{100*v/no_cells:-3.0f}% ({v})')
    else:
        print(f'{k}: \t{100*v/no_cells:-3.0f}% ({v})')
        
print('-------------------------')
print(f'             100% ({no_cells})')