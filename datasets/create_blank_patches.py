#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 10:17:13 2021

@author: jimmytabet
"""

#%%
import os, glob
import numpy as np
from scipy import ndimage

import matplotlib.pyplot as plt
from skimage.util import montage

path_to_data = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles'
tiles = sorted(glob.glob(os.path.join(path_to_data,'**','*.npy'), recursive=True))
tiles = [file for file in tiles if not '_finished' in file]
tiles = [file for file in tiles if not '_preprocessed' in file]

np.random.shuffle(tiles)


# bounding box square 192 pixels long centered at 150
bb_size = 286
half_size = bb_size//2

points = []
x_grid, y_grid = np.meshgrid(range(half_size, 800-half_size), range(half_size, 800-half_size))
for i,j in zip(x_grid.flatten(), y_grid.flatten()):
    points.append([i,j])
    
points = np.array(points)

X = []
y = []
blanks = 0
for i, tile in enumerate(tiles):
    
    if not i%100:
        print(f'{i} of {len(tiles)}')
        print(f'number of blanks: {blanks}')
    
    data = np.load(tile, allow_pickle=True).item()
    raw = data['img']
    masks = data['masks']
    num_masks = np.max(masks)
    
    if num_masks == 0:
        blanks += 1
        point = np.random.randint(half_size, raw.shape[0]-half_size, 2)
    
    elif num_masks == 1:
        center = ndimage.center_of_mass(masks==1)
        center = np.array(center).astype(int)
        
        # point = center
        # while any((center-point)<(bb_size+10)):
        #     print('finding ideal point...')
        #     points = np.random.randint(half_size, raw.shape[0]-half_size, (10,2))
        #     dist = np.linalg.norm(center-points, axis=1)
        #     point = points[np.argmax(dist)]
        
        dist = np.linalg.norm(center-points, axis=1)
        point = points[np.argmax(dist)]        
        
        if any(abs(center-point)<(bb_size)):
            print(abs(center-point))
        else:
            blanks += 1
        
    else:
        continue
        
    # original bounding box
    r1_o = point[0]-half_size
    r2_o = point[0]+half_size
    c1_o = point[1]-half_size
    c2_o = point[1]+half_size
    
    # find bounding box indices to fit in tile
    r1 = max(0, point[0]-half_size)
    r2 = min(raw.shape[0], point[0]+half_size)
    c1 = max(0, point[1]-half_size)
    c2 = min(raw.shape[1], point[1]+half_size)

    # pad new bounding box with constant value (mean, 0, etc.)
    final = np.zeros([half_size*2, half_size*2], dtype=raw.dtype)
    final += raw[masks==0].mean().astype('int')

    # store original bb in new bb
    final[r1-r1_o:r2-r1_o,c1-c1_o:c2-c1_o] = raw[r1:r2,c1:c2]

    X.append(final)
    y.append('blank')
        
    # if blanks > 800:
    #     break
    
#%%
X = np.stack(X)
y = np.array(y)
print(X.shape)
print(y.shape)

# np.savez('/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_BLANKS.npz', X=X, y=y)

plt.imshow(montage(X), cmap='gray')