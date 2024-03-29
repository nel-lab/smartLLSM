#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:12:21 2021

@author: jimmytabet
"""

#%% imports
import os, time, csv, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import torch

#%% user setup
stage_of_interest = ['anaphase',
                    #  'blurry',
                     'interphase',
                     'metaphase',
                     'prometaphase',
                     'prophase',
                     'telophase']
thresh = 0.0

folder_to_watch = '/path/to/folder_to_watch'

# YOLO paths
path_to_weights = '/path/to/yolo_weights.pt'

# delay in seconds between checking folder
delay = 5

# visualize results
visualize_results = False

# store ALL results
store_all = True

# order cells by score to set thresh
set_thresh = False
# minuimum score to show
set_thresh_thresh = 0.1

#%% model setup
nn_model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights)

# ensure stage_of_interest is a list and check for valid inputs
stage_of_interest = list(stage_of_interest)

yolo_classes = ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']
for stage in stage_of_interest:
    if not stage in yolo_classes:
        raise ValueError(f'"{stage}" stage_of_interest is not valid, please choose from {yolo_classes}')

#%% finish setup
# date and csv path
current_date = time.strftime('%m_%d_%y')
results_csv = os.path.join(folder_to_watch, f'results_{current_date}.csv')

# create completed folders if not already
os.makedirs(os.path.join(folder_to_watch, 'completed', current_date), exist_ok=True)

# set folder for finished images
finished_folder = os.path.join(folder_to_watch, 'completed', current_date)

# set up storage folder
if store_all:
    all_found_cells_csv = os.path.join(folder_to_watch, f'all_found_cells_{current_date}.csv')
    # if creating a new file, write yolo classes as file header
    if not os.path.exists(all_found_cells_csv):
        with open(all_found_cells_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(np.insert(yolo_classes,0,'').reshape([1,-1]))
else:
    all_found_cells_csv=False

# set up threshold folder
if set_thresh:
    thresh_folder = os.path.join(folder_to_watch, 'set_thresh')
    os.makedirs(thresh_folder, exist_ok=True)
else:
    thresh_folder=False

#%% pipeline
def run_pipeline(files, nn_model, stage_of_interest, thresh, results_csv,
                 viz=False, store_csv=False,
                 thresh_folder=False, set_thresh_thresh=0.1):
        
    # read in image
    raw = io.imread(files).astype(float)
            
    # add extra dimension if not tif stack
    if raw.ndim == 2:
        individual = True
        raw = raw[np.newaxis,...]
    else:
        individual = False
    
    # preprocess raw image
    # convert uint16 to uint8 for yolo
    raw_8bit = ((raw-raw.min(axis=(1,2))[:,None,None])/(raw.max(axis=(1,2))[:,None,None]-raw.min(axis=(1,2))[:,None,None])*(2**8-1)).astype('uint8')
    raw_8bit = list(raw_8bit) # must be passed as list of images for yolo
            
    # forward pass
    res = nn_model(raw_8bit)

    # get output as list of pandas df's
    output = res.pandas().xywh
    
    # add slice ID
    for i, df in enumerate(output):
        df['slice']=i
    
    # merge output df's
    combined_output = pd.concat(output)
        
    # write distribution of cell stages to file 
    if store_csv:
        # init counts with file name
        counts = [os.path.basename(files)]
        # append corresponding cell count for stage in yolo_classes
        cell_dist = combined_output.name.value_counts()
        for stage in yolo_classes:
            if stage in cell_dist.index:
                counts.append(cell_dist[stage])
            else:
                counts.append(0)
        # write to file
        with open(store_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(counts).reshape([1,-1]))
    
    # mask only results from correct stage
    combined_output_stage = combined_output[combined_output.name.isin(stage_of_interest)]

    # mask stage output above thresh
    combined_output_thresh = combined_output_stage[combined_output_stage.confidence >= thresh].copy() # .copy() needed to avoid pandas warning

    # add file name
    combined_output_thresh['fname'] = os.path.basename(files)

    # remove slice ID if individual image
    if individual:
        combined_output_thresh.slice = ''

    # extract useful info
    combined_output_thresh = combined_output_thresh[['fname','slice','xcenter','ycenter','confidence','name']]
    
    # add xcenter/ycenter coordinates relative to center of image
    combined_output_thresh[['xcenter_rel','ycenter_rel']] = combined_output_thresh[['xcenter','ycenter']] - [i/2 for i in raw.shape[1:3]]
        
    # save to csv
    all_info = combined_output_thresh.to_numpy()
               
    # save images for set_thresh
    if thresh_folder:
        thresh_mask = combined_output_stage.confidence > set_thresh_thresh
        set_thresh_res = combined_output_stage[thresh_mask]
        
        for _,cell in set_thresh_res.iterrows():
            score = cell.confidence
            half_size = 100
            
            # original bounding box
            r1_o = int(cell.ycenter-half_size)
            r2_o = int(cell.ycenter+half_size)
            c1_o = int(cell.xcenter-half_size)
            c2_o = int(cell.xcenter+half_size)
            
            # find bounding box indices to fit in tile
            r1 = max(0, r1_o)
            r2 = min(raw.shape[1], r2_o)
            c1 = max(0, c1_o)
            c2 = min(raw.shape[2], c2_o)
                        
            # pad new bounding box with constant value (mean, 0, etc.)
            final = np.zeros([half_size*2, half_size*2], dtype=raw.dtype)
            
            # store original bb in new bb
            final[r1-r1_o:r2-r1_o,c1-c1_o:c2-c1_o] = raw[cell.slice, r1:r2,c1:c2]
            
            # save to thresh_folder
            matplotlib.image.imsave(os.path.join(thresh_folder, str(round(score*100, 1))+'.tiff'), final, cmap='gray')
    
    # write cell centroid to results csv
    if all_info.size:
        with open(results_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(all_info)
                
        # view results
        if viz:
            n = int(np.ceil(np.sqrt(len(all_info))))
            fig = plt.figure()
            count = 1
            for (_, idx, cx, cy, _, name, _, _) in all_info:
                im = io.imread(files)
                if idx >= 0:
                    im = im[int(idx)]
                                                        
                ax = fig.add_subplot(n,n,count)
                ax.set_title(f'{os.path.basename(files)} {idx} {name}')
                ax.axis('off')
                ax.imshow(im.squeeze(), cmap='gray',
                          vmin=np.percentile(im,1), vmax=np.percentile(im,99))
                ax.scatter(float(cx),float(cy), c='r', marker='*')
                count += 1
            # pause to show image while pipeline runs
            plt.pause(0.001)            
        
        return combined_output_thresh.confidence.max()
    
    # just write file name       
    else:
        with open(results_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(os.path.basename(files)).reshape([-1,1]))
        
        return False

#%% watch folder
start_loop_time = time.time()

while True:
    # look for files
    files_all = sorted(glob.glob(os.path.join(folder_to_watch,'**','*.tif'), recursive=True))
    files_all = [file for file in files_all if not 'completed' in file]
    
    # files_analyzed = files_all[:BATCH_SIZE]
    
    # if len(files_all) >= BATCH_SIZE:
    if files_all:
        files_analyzed = files_all[0]
        # print('----------------------------')
        start = time.time()
        cell_found = run_pipeline(files_analyzed, nn_model, stage_of_interest, thresh, results_csv,
                                  viz=visualize_results, store_csv=all_found_cells_csv,
                                  thresh_folder=thresh_folder, set_thresh_thresh=set_thresh_thresh)
        
        print(f'cell found: {cell_found}')
        print(f'pipeline time: {time.time()-start}')
        print('----------------------------')
        
        # move files to completed folder
        os.replace(files_analyzed, os.path.join(finished_folder, os.path.basename(files_analyzed)))

    else:
        print(f'waiting for files...{len(files_all)} file(s)  time: {time.time()-start_loop_time}', end='\r') #end='\r' will prevent generating a new line and will overwrite this line over and over
        time.sleep(delay)
