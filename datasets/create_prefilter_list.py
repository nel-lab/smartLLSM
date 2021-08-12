#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:56:25 2021

@author: jimmytabet
"""

############################## SET UP ANNOTATOR ###############################

# imports
import os
import glob
import numpy as np
from scipy import ndimage
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

#----------------------------------USER INPUT---------------------------------#

# boolean to use automatic neural network filter
nn_filter = True
path_to_nn_filter = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotator_filter.hdf5'

# threshold confidence to classify "unique" cells/automatically filter tiles
filter_thresh = 0.7

# path to data (optionally passed in terminal - use '$(pwd)' to pass pwd)
path_to_data = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/data_3_cellpose'

# check that given path is actually a folder
while not os.path.isdir(path_to_data):
    path_to_data = input('ERROR: '+path_to_data+' not found, try again\npath to data: ')
    path_to_data = os.path.abspath(path_to_data)

# change directory to data
os.chdir(path_to_data)
# create list of tiles to annotate (exclude 'finished' tiles)
print('CREATING LIST OF FILES FOR ANNOTATION...')
tiles = sorted(glob.glob(os.path.join(path_to_data,'**','*.npy'), recursive=True))
tiles = [file for file in tiles if not '_finished' in file]

# set up annotator if there are files to annotate
if tiles:
    # set up neural network and filtering function if used
    if nn_filter:
        print('LOADING NEURAL NETWORK FILTER...')
        import tensorflow as tf
        def nn_temp(a,b):return True
        model = tf.keras.models.load_model(path_to_nn_filter, custom_objects={'f1_metric':nn_temp})
        # set filter class/column
        output_classes = np.array(model.name.split('__temp__'))
        filter_class = 'unique'
        while not filter_class in output_classes:
            filter_class = input('Invalid filter class, pick from the following: '\
                                 +str(output_classes)+'\n\tfilter class: ')
        
        filter_col = int(np.where(filter_class == output_classes)[0])
        
        # filtering function to determine if tile contains interesting cell
        def interesting(nn_model, raw_tile, masks_tile, thresh, thresh_col):
        
            # find number of identified cells
            num_masks = np.max(masks_tile)
            
            # loop through each cell and check if interesting
            for mask_id in range(1,num_masks+1):
            
                # find center of mass (as integer for indexing)
                center = ndimage.center_of_mass(masks_tile==mask_id)
                center = np.array(center).astype(int)
                
                # create image to test for filtering with nn
                nn_half_size = nn_model.input_shape[1]//2
                r1_o = center[0]-nn_half_size
                c1_o = center[1]-nn_half_size
                # find bounding box indices to fit in tile
                r1_nn = max(0, center[0]-nn_half_size)
                r2_nn = min(raw_tile.shape[0], center[0]+nn_half_size)
                c1_nn = max(0, center[1]-nn_half_size)
                c2_nn = min(raw_tile.shape[1], center[1]+nn_half_size)
                # pad new bounding box with constant value (mean, 0, etc.)
                nn_test = np.zeros([nn_half_size*2, nn_half_size*2])
                nn_test += raw_tile[masks_tile==0].mean().astype('int')
                # store original bb in new bb
                nn_test[r1_nn-r1_o:r2_nn-r1_o,c1_nn-c1_o:c2_nn-c1_o] = raw_tile[r1_nn:r2_nn,c1_nn:c2_nn]
        
                # normalize image
                mean, std = np.mean(nn_test), np.std(nn_test)
                nn_test = (nn_test-mean)/std
                # add dimension to input to model
                nn_test = nn_test.reshape(1,*nn_model.input_shape[1:])
                preds = nn_model.predict(nn_test).squeeze()
                
                # return True if interesting
                if preds[thresh_col] > thresh:
                    return True
                else:
                    continue

INTERESTING_FILES = []
count = 0
for tile in tiles[8570+7921+0+0:]:
    
    # load Cellpose data (raw image, masks, and outlines)
    data = np.load(tile, allow_pickle=True).item()
    raw = data['img']
    masks = data['masks']
    outlines = data['outlines']

    # if using filter, check for interesting cells in tile and skip if none found
    if nn_filter and interesting(model, raw, masks, filter_thresh, filter_col):
        INTERESTING_FILES.append(os.path.relpath(tile))
        
    count += 1
    if not count%100:
        print(f'{count} of {len(tiles)-8570-7921-0-0}')
        
#%% SAVE PARTS
print(count)
print(len(INTERESTING_FILES))
# np.save('/home/nel-lab/Desktop/data_3_pt_3', INTERESTING_FILES)

#%% CHECK RESULTS
import matplotlib.pyplot as plt

d1 = np.load('/home/nel-lab/Desktop/data_3_pt_1.npy')
d2 = np.load('/home/nel-lab/Desktop/data_3_pt_2.npy')
d3 = np.load('/home/nel-lab/Desktop/data_3_pt_3.npy')

full = np.concatenate([d1,d2,d3])
print(full.shape)

for i in range(10):
    idx = np.random.randint(len(full))
    img = np.load(full[idx], allow_pickle=True).item()['img']
    plt.imshow(img, cmap='gray')
    plt.pause(1)
    
# np.save('/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/data_3_preprocessed.npy', full)
