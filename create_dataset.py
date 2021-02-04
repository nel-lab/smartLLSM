#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:18:12 2021

@author: jimmytabet
"""

#%% modules and functions
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy import ndimage
from skimage.util import montage

# stitch function
def create_stitch(folder, overlap, im_shape, im_per_ax, dtype):

    # init stitch array with correct dtype
    stitch = np.zeros([im_per_ax*ax for ax in im_shape], dtype=dtype)
    for row in range(im_per_ax):
        for col in range(im_per_ax):
            im = plt.imread(folder+'/Scan_Iter_000'+str(row)+'_000'+str(col)+'.tif')
            # must flip horizontally to read image properly
            im = im[:,::-1]
            
            row_start = row*(im_shape[0]-overlap)
            row_end = row_start+im_shape[0]
    
            col_start = col*(im_shape[1]-overlap)
            col_end = col_start+im_shape[1]
    
            stitch[row_start:row_end, col_start:col_end] = im
        
    # return trimmed stitch
    return stitch[0:row_end,0:col_end]

# seperate cells function (optionally, isolate mask and include edge cases)
#   xy: mx2 array of bottom left points of bounding box
#   masks (optional): Cellpose masks used to isolate cell in case of nearby cells
#   add_edge: boolean to add edge cases to cell dataset
#   outliers: array of cells to skip 
#   returns KxHxW array of K cell images with same dims (H,W)
def sep_cells(pic, ids, xy, bb_size_px, masks=[], add_edge=False, outliers=[]):
    
    cells=[]
    outlier_cells=[]
    edge_cells=[]
    
    for idx in ids:
        idx = int(idx)
    
        # reset outlier/edge
        out = False
        edge = False
    
        bl = xy[idx]
        r1 = max(0, min(bl[0], pic.shape[0]))
        r2 = max(0, min(bl[0]+bb_size_px+1, pic.shape[0]))
        c1 = max(0, min(bl[1], pic.shape[1]))
        c2 = max(0, min(bl[1]+bb_size_px+1, pic.shape[1]))
            
        # check if outlier or edge
        if idx in outliers:
            out = True
            print(idx, 'outlier')      
        
        if r2-r1-1!=bb_size_px or c2-c1-1!=bb_size_px:
            edge = True
            print(idx, 'edge')
        
        # fix (if on edge)
        rfix = bb_size_px - (r2-r1-1)
        cfix = bb_size_px - (c2-c1-1)
        
        if r1 == 0: r2 += rfix
        if r2 == pic.shape[0]: r1 -= rfix
        if c1 == 0: c2 += cfix
        if c2 == pic.shape[1]: c1 -= cfix
        
        # isolate mask (optional)
        if len(masks):
            # create pic copy to isolate mask
            isolate_pic = pic.copy()
            isolate_pic[masks!=idx+1] = 0
            cell = isolate_pic[r1:r2,c1:c2]       
        else:
            cell = pic[r1:r2,c1:c2]        
        
        # assign cell to right category
        if out:
            outlier_cells.append(cell)
        elif edge and not add_edge:
            edge_cells.append(cell)
        else:
            cells.append(cell)
            
    cells, outlier_cells, edge_cells = np.array(cells), np.array(outlier_cells), np.array(edge_cells)
    
    return cells, outlier_cells, edge_cells

#%% create and save stitches
positions = 20

# loop through Positition folders
for pos in range(positions):
    folder = 'data_1/Position '+str(pos)

    # sample image
    test = plt.imread(folder+'/Scan_Iter_0000_0000.tif')
    # image shape
    im_shape = test.shape
    # total images
    total_im = len(os.listdir(folder))
    # images per axis (assuming square larger image)
    im_per_ax = int(np.sqrt(total_im))
    # image data type
    dtype = test.dtype
    
    overlap = 255
    stitch = create_stitch(folder, overlap, im_shape, im_per_ax, dtype)

    # create stitch
    stitch_im = Image.fromarray(stitch)

    # save stitch as tif
    if not os.path.exists('Cellpose/stitch '+str(pos)+'.tif'):
        stitch_im.save('Cellpose/stitch '+str(pos)+'.tif')

#%% RUN THROUGH CELLPOSE
    
#%% compare avg_dia/low/high for range(positions) vs range(0,1) (first image) ...
#   for bounding box (takes ~15s per image)
# range(positions): (136.8563771554042, 65.34859164652445, 208.3641626642838)
# range(0,1):       (143.0706769284303, 48.77965545353999, 237.3616984033206)
# so first image is good enough predictor        

dias_all=[]
for pos in range(positions):
    cellpose_data = np.load('Cellpose/stitch '+str(pos)+'_seg.npy', allow_pickle=True).item()
    masks = cellpose_data['masks']

    # find mask centroids
    num_masks = np.max(masks)
    
    center_masks = ndimage.center_of_mass(masks, masks, np.arange(1,num_masks+1))
    center_masks = np.array(center_masks)

    # find average cell diameter (est. as circle)
    areas = []
    for i in range(1,num_masks+1):
        areas.append((masks==i).sum())
    areas = np.array(areas)
    dias = np.sqrt(areas*4/np.pi)
    
    dias_all.append(dias)
    
    print(pos+1, 'of', positions)
    
dias_all = np.concatenate(dias_all, axis=0)
avg_dia = np.mean(dias_all)
low = avg_dia-3*np.std(dias_all)
high = avg_dia+3*np.std(dias_all)
print(avg_dia, low, high)
plt.hist(dias_all, bins=100)
plt.axvline(low, color='r')
plt.axvline(high, color='r')

#%% create cell dataset - take first image for bounding box (takes ~30s per image)
%%time

positions = 20

for run, pos in enumerate(range(positions)):
    stitch = plt.imread('Cellpose/stitch '+str(pos)+'.tif')
    cellpose_data = np.load('Cellpose/stitch '+str(pos)+'_seg.npy', allow_pickle=True).item()
    # outlines = cellpose_data['outlines']
    masks = cellpose_data['masks']

    # find mask centroids
    num_masks = np.max(masks)
    center_masks = ndimage.center_of_mass(masks, masks, np.arange(1,num_masks+1))
    center_masks = np.array(center_masks)

    # find average cell diameter (est. as circle)
    areas = []
    for i in range(1,num_masks+1):
        areas.append((masks==i).sum())
    areas = np.array(areas)
    dias = np.sqrt(areas*4/np.pi)
        
    avg_dia = np.mean(dias)

    # determine bounding box size for cells (based on first run)
    if run == 0:
        low = avg_dia-3*np.std(dias)
        high = avg_dia+3*np.std(dias)
    
    bb_size = high-low
    #set bounding box size as integer for indexing
    bb_size_px = (np.ceil(bb_size/2)*2).astype(int)
    # xy = bottom left point as integer for indexing
    xy = np.floor(center_masks-bb_size/2).astype(int)

    # outliers within stitch    
    outliers = np.argwhere((dias < avg_dia-3*np.std(dias)) | (dias > avg_dia+3*np.std(dias)))

    # get cells array
    cells, cells_outliers, cells_edges = sep_cells(stitch, range(num_masks), xy, bb_size_px, masks=masks, fix=False, outliers=outliers)
    
    # add cells to cell dataset (cells_all)
    if run == 0:
        cells_all = cells
    else:
        cells_all = np.concatenate([cells_all, cells], axis=0)
    
    # # create montage
    # fig, ax = plt.subplots(1,2)
    # fig.set_tight_layout(True)
    # fig.suptitle('Position '+str(pos))
    
    # # create montage - ID cells
    # mont = montage(cells, fill=.7*np.iinfo(cells.dtype).max, rescale_intensity=True, padding_width = 25)
    # ax[0].imshow(mont, cmap='gray')
    # ax[0].set_xticks([])
    # ax[0].set_yticks([])
    # ax[0].set_title('Identified Cells ('+str(cells.shape[0])+')')

    # # create montage - outlier cells (gray box if no outliers)
    # if cells_outliers.size == 0:
    #     ax[1].imshow([[(.7,.7,.7)]], cmap='gray')
    #     ax[1].set_title('Rejected Cells (None Found/Fixed)')
    # else:
    #     mont_outliers = montage(cells_outliers, fill=.7*np.iinfo(cells_outliers.dtype).max, rescale_intensity=True, padding_width = 25)    
    #     ax[1].imshow(mont_outliers, cmap='gray')
    #     ax[1].set_title('Rejected Cells')

    # ax[1].set_xticks([])
    # ax[1].set_yticks([])
    
    # DPI = fig.get_dpi()
    # fig.set_size_inches(1920.0/float(DPI),1080.0/float(DPI))
    # fig.savefig('montages/Position '+str(pos), dpi=DPI, bbox_inches='tight')
    plt.close('all')
    
    print(pos+1, 'of', positions)

#%%
# save cell dataset
np.save('datasets/cell_dataset_isolate_'+str(cells_all.shape[0]), cells_all)
