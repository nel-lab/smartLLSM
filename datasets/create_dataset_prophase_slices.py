#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 16:59:25 2021

@author: jimmytabet
"""

#%% IMPORTS AND FUNCTIONS
import numpy as np
import glob, os
from PIL import Image
from scipy.fftpack import dct
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.util import montage

# create array for tif_stack function
def tif_stack(tif_path):
    dataset = Image.open(tif_path)
    slices = dataset.n_frames
    h,w = dataset.size
    tif_array = np.zeros([slices, h, w])
    for i in range(slices):
       dataset.seek(i)
       tif_array[i,:,:] = np.array(dataset)

    return tif_array

# MATLAB wentropy function
def shannon_entropy(image):
    s = dct(image, norm='ortho')
    s = s[s>0]
    SE = -1*np.sum(s**2 * np.log(s**2))
    
    return SE

# check if entropy result matches raw image function
def check_entropy(raw, stack):
    ents=[]
    for z in stack:
        ents.append(shannon_entropy(z))
      
    sl = np.argmin(ents)
    match = (raw==stack[sl]).all()
    
    return match
    
#%% GET TARGET CELLS
# define tif stack and annotated paths
stack_folder = r'Y:\Katie\LLS_CNN_training_2020_09_29\data_1'
stack_tifs = sorted(glob.glob(os.path.join(stack_folder, '**', '*.tif'), recursive=True))
annotated_folder = r'C:\Users\LegantLab\NEL-LAB Dropbox\NEL\Datasets\smart_micro\Cellpose_tiles\annotation_results\data_1_cellpose'
annotated_files = sorted(glob.glob(os.path.join(annotated_folder, '**','*.npz'), recursive=True))

# set bounding box size
bb_size = 286
half_size = bb_size//2

# init values
error = False
target = 'prophase'
z_spread = 2
X = []
y = []
ID = []
slice_ID = []

# loop over each annotated file
count = 0
for file in annotated_files:

    # check for target cell
    dat = np.load(file, allow_pickle=True)    
    labels_dict = dat['labels_dict'].item()
    labels_stage = [labels_dict[i].lower() for i in dat['labels']]

    # skip if no target cells found
    if not target in labels_stage:
        continue
    else:
        raw = dat['raw']
        masks = dat['masks']
    
    # find path identifier to match with tif stack
    identifier = [i.split('_')[:-1] for i in file.split(os.sep)[-3:]]
    identifier = ['_'.join(i) for i in identifier]
    identifier = os.path.join(*identifier)
    potential_stacks = [x for x in stack_tifs if identifier in x]
    if len(potential_stacks) != 1:
        print('file error in',identifier)
        error=True
        break
    else:
        stack_path = potential_stacks[0]
    
    # more efficient option - DOES NOT CHECK THAT FILE EXSISTS
    # for i in stack_tifs:
    #     if identifier in i:
    #         stack_path = i
    #         break

    stack = tif_stack(stack_path)

    # check entropy on entire tile
    SE_works = check_entropy(raw, stack)
    if not SE_works:
        print('SE error in',identifier)

    # loop through target mask_ids to get different slices
    target_ids = np.array([i for i, x in enumerate(labels_stage) if x == target])
    for mask_id in target_ids+1:        
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
        
        # save bounding box indices for no_isolate, centered cell
        r1_o_fix = r1
        r2_o_fix = r2
        c1_o_fix = c1
        c2_o_fix = c2
        
        # fix bounding box to fit in tile
        rfix = half_size*2 - (r2-r1)
        cfix = half_size*2 - (c2-c1)
        if r1 == 0: r2 += rfix
        if r2 == raw.shape[0]: r1 -= rfix
        if c1 == 0: c2 += cfix
        if c2 == raw.shape[1]: c1 -= cfix   

        # create new array to pad bounding box with constant value (mean, 0, etc.)      
        all_cell = np.zeros([len(stack), half_size*2, half_size*2], dtype = raw.dtype)
        all_cell += raw[masks==0].mean().astype('int')
        
        # copy stack to save isolated cell
        stack_isolate = stack.copy()
        individual_cell = np.zeros([len(stack_isolate), half_size*2, half_size*2], dtype = raw.dtype)
        SE_min = np.inf
        
        # loop over each slice to isolate cell and find best slice
        for i,z in enumerate(stack_isolate):
            all_cell[i,r1_o_fix-r1_o:r2_o_fix-r1_o,c1_o_fix-c1_o:c2_o_fix-c1_o] = z[r1_o_fix:r2_o_fix,c1_o_fix:c2_o_fix]
            z[masks!=mask_id] = 0
            individual_cell[i] = z[r1:r2,c1:c2]
            SE = shannon_entropy(individual_cell[i])
            if SE < SE_min:
                SE_min = SE
                best = i
        
        # if entropy did not work compare to original (and/or print best slice)
        if not SE_works:
            # print(best)
            fig,ax = plt.subplots(1,2)
            fig.suptitle(identifier)
            ax[0].imshow(all_cell[best], cmap='gray')
            ax[0].set_title('my_entropy')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[1].imshow(raw[r1:r2,c1:c2], cmap='gray')
            ax[1].set_title('original')
            ax[1].set_xticks([])
            ax[1].set_yticks([])
        
        # check best slice is in range for z_spread
        if best-z_spread < 0 or best+z_spread > len(stack):
            print('ID error in',identifier)
            error=True
            break
        
        # save best slices of cell
        slice_num = -z_spread
        for i in range(best-z_spread,best+z_spread+1):        
            X.append(all_cell[i])
            # X.append(individual_cell[i])
            y.append(target)
            ID.append(count)
            slice_ID.append(slice_num)
            slice_num += 1

        count += 1
        if count%10 == 0:
            print(f'{count}\tof\t103')

# explore results to confirm
X = np.stack(X)
y = np.array(y)
ID = np.array(ID)
slice_ID = np.array(slice_ID)
print(X.shape)
print(y.shape, np.unique(y))
print(ID.shape, np.unique(ID))
print(slice_ID.shape, np.unique(slice_ID))
plt.figure()
plt.imshow(montage(X, padding_width = 10, grid_shape=(np.ceil(len(X)/35),35)), cmap='gray')
plt.xticks([])
plt.yticks([])
plt.show()

#%% SAVE CELLS (IF NO ERROR)
path_to_datasets = r'C:\Users\LegantLab\NEL-LAB Dropbox\NEL\Datasets\smart_micro\datasets'
if not error:
    np.savez(os.path.join(path_to_datasets, target+'_cell_slices_no_isolate_286.npz'), X=X, y=y, ID=ID, slice_ID=slice_ID)