#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:12:21 2021

@author: jimmytabet
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
from tqdm import tqdm

#%% set up analysis and screening
path_to_yolo_repo = '/home/nel/Software/yolov5'
path_to_weights = '/home/nel/Software/yolov5/runs/train/exp20/weights/best.pt'

# yolo model
nn_model = torch.hub.load(path_to_yolo_repo, 'custom', path=path_to_weights, source='local')

# list of test file paths
test_files = np.load('/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/YOLO_test_files_0830.npy')

# stages to screen out
screen = [
    'edge',
    'blurry',
    # 'interphase',
    # 'prophase',
    # 'prometaphase',
    # 'metaphase',
    # 'anaphase',
    # 'telophase',
    'junk',
    'TBD',
    'other'
    ]

# min/max radius for area screen
'''
# summary stats of cell areas
Q1 = 7900
Q3 = 16000
IQR = Q3-Q1

#%% cell_size array
cell_size = np.array(cell_size)        

#%% eval cell_size
# remove outliers (1.5*IQR)
q75, q25 = np.percentile(cell_size, [75 ,25])
iqr = q75 - q25

iqr_mask = (cell_size<q75) & (cell_size>q25)

cell_mask = cell_size[iqr_mask]

#%%
plt.hist(cell_size, bins = 50)
plt.hist(cell_mask, bins = 50)
plt.legend(['og', 'size mask'])
'''
r_min = 50
r_max = 100

# option to visualize centroids
viz = False
error_thresh = 100
viz_count = 0
viz_limit = 2

#%% loop through test files
# init loc_error
loc_error = []

# count number of Cellpose cells not screened out
cellpose_count = 0

# count number of yolo cells
yolo_count = 0

# loop through each test file
for file in tqdm(test_files):

    # load relevant data       
    with np.load(file) as dat:
        raw = dat['raw']
        masks = dat['masks']
        labels = dat['labels']
    
    # iterate over all cells to screen for 'condition'
    for i in range(masks.max()):
        # mask cell
        cell_mask = (masks == i+1)
        # screen for 'condition': cell stage/size/etc
        screen_condition = labels[i] in screen or cell_mask.sum() < np.pi*r_min**2 or cell_mask.sum() > np.pi*r_max**2
        
        # if cell is screened, set pixels to be background
        if screen_condition:
            masks[cell_mask] = 0
  
    # count number of labels not screened out
    # -1 to remove 0/background class
    cellpose_count += np.unique(masks).size-1  
  
    # calculate centers using masks
    centers = ndimage.center_of_mass(masks, masks, np.unique(masks)[1:])
    centers = np.array(centers) 
        
    # add extra dimension to image (if not tif stack)
    if raw.ndim == 2:
        raw = raw[np.newaxis,...]
    
    # preprocess raw image
    # convert uint16 to uint8 for yolo
    raw_8bit = ((raw-raw.min(axis=(1,2))[:,None,None])/(raw.max(axis=(1,2))[:,None,None]-raw.min(axis=(1,2))[:,None,None])*(2**8-1)).astype('uint8')
    raw_8bit = list(raw_8bit) # must be passed as list of images for yolo
            
    # forward pass
    res = nn_model(raw_8bit)
        
    # get output as list of pandas df's
    output = res.pandas().xywh[0]
    
    # count number of yolo cells
    yolo_count += len(output)
    
    # screen yolo cells for cell stage
    # output = output[~output.name.isin(screen)]
        
    # extract yolo centers
    centers_yolo = output.filter(['xcenter','ycenter']).to_numpy() 
    
    # skip if no Cellpose or YOLO centers (error will be raised when trying to calc distances)
    if centers.size == 0 or centers_yolo.size == 0:
        continue
    
    # swap columns to mach with yolo_centers
    centers = centers[:,[1,0]] 
    
    # create distance matrix between all points
    distances = cdist(centers, centers_yolo)
    
    # assign points using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(distances)

    # append loc_error to array
    errors = distances[row_ind, col_ind]
    loc_error.append(errors)

    # viualize if an error is > error_thresh
    if viz and errors.max() > error_thresh and viz_count < viz_limit:
        # show yolo results
        res.show()
        
        # show image
        plt.figure()
        plt.imshow(raw.squeeze(), cmap='gray')
        
        # show matching centers
        for num,(i,j) in enumerate(zip(row_ind, col_ind)):
            plt.scatter(centers[i,0], centers[i,1], marker = '*', color = f'C{num}', label = num*'_' + 'Cellpose centers')
            plt.scatter(centers_yolo[j,0], centers_yolo[j,1], marker = 'o', color = f'C{num}', label = num*'_' + 'YOLO centers')
        
        plt.legend()
        
        # break if viz_count reaches limit
        viz_count += 1
        if viz_count >= viz_limit:
            break
    
#%% convert loc_errors to 1D array
loc_error_final = np.concatenate(loc_error, axis=0)

#%% process loc_errors
# mean, std, median, min, and max
mean = loc_error_final.mean().round(3)
std = loc_error_final.std().round(3)
median = np.median(loc_error_final).round(3)
minn = loc_error_final.min().round(3)
maxx = loc_error_final.max().round(3)

# plot hist + stats
plt.figure()
plt.hist(loc_error_final, bins = 50, range=(0,50))
plt.axvline(mean, ls = '--', c='k', label = 'mean')
plt.axvline(median, ls = ':', c='k', label= 'median')
plt.title('YOLO Localization Error, px')
ax = plt.gca()
plt.text(0.5, 0.3, f'YOLO Cells: {yolo_count}\nCellpose Cells: {cellpose_count}\nMatched Cells: {len(loc_error_final)}\n\nMean: {mean}\nStd: {std}\n\nMedian: {median}\nMin: {minn}\nMax: {maxx}', transform=ax.transAxes)
plt.legend()