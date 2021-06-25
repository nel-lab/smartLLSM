#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:35:56 2021

@author: jimmytabet
"""

import glob, cv2
import numpy as np

# input_size = (286, 286)
# half_size = input_size[0]//2

path = '/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results/data_2_cellpose/Position 105_results'

files = sorted(glob.glob(path+'/*.npz'))

gt_pro = []
FCN_max = []
center_dist = []

for file in files:
    dat = np.load(file, allow_pickle=True)
    
    raw = dat['raw']
    mask = dat['masks']
    labels = dat['labels']
    labels_dict = dat['labels_dict'].item()
    
    stages = [labels_dict[lab] for lab in labels]
    
    pro_id = np.where(np.array(stages) == 'prophase')[0]+1
    gt_pro.append(len(pro_id))
    
    raw_FCN = raw - raw.mean()
    raw_FCN /= raw.std()
    '''
    pass raw_FCN through FCN to get prophase centroid
    '''
    FCN_pro_max = False
    FCN_max.append(FCN_pro_max)
    
    # skip if no prophase cells in tile (distance = inf)
    if len(pro_id) == 0:
        center_dist.append(np.inf)
        continue
    
    '''
    calc FCN prophase centroid
    '''
    FCN_x = False
    FCN_y = False
    
    # calc gt prophase centroid(s)
    dists = []
    for mask_id in pro_id:

        # get moments for cell to calculate center of mass
        M = cv2.moments(1*(mask==mask_id), binaryImage=True)
    
        if M["m00"] != 0:
          cX = M["m10"] / M["m00"]
          cY = M["m01"] / M["m00"]
        else:
          cX, cY = 0, 0 
    
        dist = np.linalg.norm([cX-FCN_x, cY-FCN_y])
        dists.append(dist)
        
    # keep shorter distance
    center_dist.append(min(dists))
  
#%%
gt_pro = np.array(gt_pro)
FCN_max = np.array(FCN_max)
center_dist = np.array(center_dist)

#%%
pro_thresh = .7
dist_thresh = 100

# FCN_pro = FCN_max > pro_thresh

FCN_pro = np.array([1,0,0,1,1,0])
gt_pro =  np.array([1,0,1,1,0,2])
center_dist = np.array([50,np.inf, 70, 150, np.inf, 200])

fp = 0
fn = 0
tp = 0
tn = 0
oor = 0

for FCN, gt, dis in zip(FCN_pro, gt_pro, center_dist):
    if gt == 0:
        if FCN == 0:
            tn += 1
        else:
            fp += 1
            
    else:
        if FCN == 0:
            fn += 1
        elif dis < dist_thresh:
            tp += 1
        else:
            oor += 1

print('tp:', tp)
print('tn:', tn)
print('fp:', fp)
print('fn:', fn)
print('OOR:', oor)