#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 15:56:52 2021

@author: jimmytabet
"""

#%% imports
import os,glob
import numpy as np

#%% new annotated data
path = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
files = sorted(glob.glob(os.path.join(path,'**','*.npz'), recursive=True))
files = [file for file in files if not 'og_backup' in file]
no_tiles = len(files)

X = []
y = []
count = 0
for file in files:
    dat = np.load(file, allow_pickle=True)
    labels_dict = dat['labels_dict'].item()
    labels = [labels_dict[j] for j in dat['labels']]

    raw = dat['raw']    
    X.append(raw)
    
    if 'prophase' in labels:
        y.append(1)
    else:
        y.append(0)

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

# np.savez('/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/prophase_tiles_binary_6_2.npz', X=X, y=y)

#%%
print(f'RESULTS OF {no_tiles} TILES:\n')
print(f'   prophase:   {(100*y.sum()/no_tiles).round(2)}% ({y.sum()})')
print(f'NO prophase:  {(100*(no_tiles-y.sum())/no_tiles).round(2)}% ({no_tiles-y.sum()})')
print('--------------------------')
print(f'             100.00% ({no_tiles})')