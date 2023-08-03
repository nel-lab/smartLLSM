#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:12:21 2021

@author: jimmytabet
"""

#%% installation instructions
'''
conda create -n sm_yolo_pipeline python spyder pandas opencv tqdm matplotlib seaborn scikit-image
conda activate sm_yolo_pipeline
conda install pytorch torchvision=0.10.0=py39cuda111hcd06603_0_cuda torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
'''

#%% imports
import os, time, csv, glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from skimage import io
import torch

#%% user setup
stage_of_interest = 'synapse'
thresh = 0.6

folder_to_watch = '/home/nel/Downloads/THIS WEEK TCELL/watch_folder_5stack'

# YOLO paths
path_to_yolo_repo = '/home/nel/Software/yolov5'
path_to_weights = '/home/nel/Downloads/THIS WEEK TCELL/best_include_target.pt'

# Image identifier
CamA_id = 'CamB_ch0'   # Tcell
CamB_id = 'CamA_ch1'   # Target cell

# delay in seconds between checking folder
delay = 5

'''
WIP, possible to stitch stack?
YOLO is trained on 800x800 images so will not perform on stitched images
for now , stitch should be set to False
'''
# stitch stack
stitch = False
overlap = 255

# visualize results
visualize_results = False

# store ALL results
store_all = False

# order cells by score to set thresh
set_thresh = False
# minuimum score to show
set_thresh_thresh = 0.8

#%% model setup
nn_model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights) #torch.hub.load(path_to_yolo_repo, 'custom', path=path_to_weights, source='local')

# check for valid stage_of_interest
yolo_classes = ['isolated','synapse','contacting','target']
if not stage_of_interest in yolo_classes:
    raise ValueError(f'stage_of_interest ({stage_of_interest}) is not valid, please choose from {yolo_classes}')

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
    # all_found_cells_csv = os.path.join(folder_to_watch, f'all_found_cells_{current_date}.csv')
    all_found_cells_csv = os.path.join(finished_folder, f'all_found_cells_{current_date}.csv')
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

read_times = []
preprocess_times = []
network_times = []
postprocess_times = []

def run_pipeline(files, nn_model, stage_of_interest, thresh, results_csv,
                 stitch = False, overlap = 255,
                 viz=False, store_csv=False,
                 thresh_folder=False, set_thresh_thresh=0.1):
    
    read_start = time.time()
    
    # read in image
    raw = io.imread(files[0]).astype(float)
    n_img = len(raw)
    raw = np.append(raw,io.imread(files[1]).astype(float),axis=0)   
    
    read_times.append(time.time()-read_start)
    
    preprocess_start = time.time()
    
    # add extra dimension if not tif stack
    if raw.ndim == 2:
        individual = True
        raw = raw[np.newaxis,...]
    # convert stack to stitch 
    elif stitch:
        individual = True
        im_per_ax = int(np.sqrt(raw.shape[0]))
        
        # init stitch array with correct dtype
        stitched_im = np.zeros([im_per_ax*ax for ax in raw.shape[1:]])   
    
        idx = 0
        for row in range(im_per_ax):
            for col in range(im_per_ax):
    
                # must flip horizontally to read image properly
                im = raw[idx][:,::-1]
                
                row_start = row*(im.shape[0]-overlap)
                row_end = row_start+im.shape[0]
    
                col_start = col*(im.shape[1]-overlap)
                col_end = col_start+im.shape[1]
    
                stitched_im[row_start:row_end, col_start:col_end] = im
                
                idx += 1
                
        raw = stitched_im[0:row_end,0:col_end]
        
        # add extra dimension
        raw = raw[np.newaxis,...]
    
    else:
        individual = False
    
    # preprocess raw image
    # convert uint16 to uint8 for yolo
    raw_8bit = ((raw-raw.min(axis=(1,2))[:,None,None])/(raw.max(axis=(1,2))[:,None,None]-raw.min(axis=(1,2))[:,None,None])*(2**8-1)).astype('uint8')
    raw_rgb = np.zeros((n_img,)+raw_8bit.shape[1:]+(3,))
    raw_rgb[:,:,:,0] = raw_8bit[:n_img,:,:]
    raw_rgb[:,:,:,1] = raw_8bit[n_img:,:,:]
    raw_rgb = raw_rgb.astype('uint8')
    
    raw_rgb = list(raw_rgb) # must be passed as list of images for yolo
    
    preprocess_times.append(time.time()-preprocess_start)
    
    network_start = time.time()
    
    # forward pass
    res = nn_model(raw_rgb)
        
    network_times.append(time.time() - network_start)

    # get output as list of pandas df's
    output = res.pandas().xywh
    
    postprocess_start = time.time()
    
    # add slice ID
    for i, df in enumerate(output):
        df['slice']=i
    
    # merge output df's
    combined_output = pd.concat(output)
    
    # write distribution of cell stages to file 
    if store_csv:
        # init counts with file name
        counts = [os.path.basename(files[0])]
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
    combined_output_stage = combined_output[combined_output.name == stage_of_interest]

    # mask stage output above thresh
    combined_output_thresh = combined_output_stage[combined_output_stage.confidence >= thresh].copy() # .copy() needed to avoid pandas warning

    # add file name
    combined_output_thresh['fname'] = os.path.basename(files[0])

    # remove slice ID if individual image
    if individual:
        combined_output_thresh.slice = ''

    # extract useful info
    combined_output_thresh = combined_output_thresh[['fname','slice','xcenter','ycenter','confidence']]
        
    # save to csv
    all_info = combined_output_thresh.to_numpy()
           
    # save images for set_thresh
    if thresh_folder:
        thresh_mask = combined_output_stage.confidence > set_thresh_thresh
        set_thresh_res = combined_output_stage[thresh_mask]
        
        for _,cell in set_thresh_res.iterrows():
            score = cell.confidence
            half_size = 250
            
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
            final = np.zeros([half_size*2, half_size*2,3], dtype=raw_rgb[0].dtype)
            
            # store original bb in new bb
            final[r1-r1_o:r2-r1_o,c1-c1_o:c2-c1_o,:] = raw_rgb[cell.slice][r1:r2,c1:c2,:]
            
            # save to thresh_folder
            matplotlib.image.imsave(os.path.join(thresh_folder, cell['name']+str(round(score*100, 2))+'.tiff'), final)
    
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
            for (_, idx, cx, cy, _) in all_info:
                '''
                WIP, need to update if multiple files passed in
                '''
                if stitch:
                    im = raw
                else:
                    im = io.imread(files[0])
                    if idx >= 0:
                        im = im[int(idx)]
                                                        
                ax = fig.add_subplot(n,n,count)
                ax.set_title(f'{os.path.basename(files[0])} {idx}')
                ax.axis('off')
                ax.imshow(im.squeeze(), cmap='gray',
                          vmin=np.percentile(im,1), vmax=np.percentile(im,99))
                ax.scatter(float(cx),float(cy), c='r', marker='*')
                count += 1
            # pause to show image while pipeline runs
            plt.pause(0.001)            
        
        postprocess_times.append(time.time() - postprocess_start)
        return combined_output_thresh.confidence.max()
    
    # just write file name       
    else:
        with open(results_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(os.path.basename(files[0])).reshape([-1,1]))
        
        postprocess_times.append(time.time() - postprocess_start)
        return False

#%% watch folder
start_loop_time = time.time()

times = []

while True:
    # look for files
    files_all_CamA = sorted(glob.glob(os.path.join(folder_to_watch,'*'+CamA_id+'*.tif'), recursive=True))
    files_all_CamA = [file for file in files_all_CamA if not 'completed' in file]
    
    
    files_all_CamB = sorted(glob.glob(os.path.join(folder_to_watch,'*'+CamB_id+'*.tif'), recursive=True))
    files_all_CamB = [file for file in files_all_CamB if not 'completed' in file]    
    
    
    # files_analyzed = files_all[:BATCH_SIZE]
    
    # if len(files_all) >= BATCH_SIZE:
    if files_all_CamA:
        file_CamA = files_all_CamA[0]
        file_CamB = files_all_CamB[0]
        files_analyzed = [file_CamA,file_CamB]
        # print('----------------------------')
        start = time.time()
        cell_found = run_pipeline(files_analyzed, nn_model, stage_of_interest, thresh, results_csv,
                                  stitch=stitch, overlap=overlap,
                                  viz=visualize_results, store_csv=all_found_cells_csv,
                                  thresh_folder=thresh_folder, set_thresh_thresh=set_thresh_thresh)
        
        times.append(time.time()-start)
        print(f'cell found: {cell_found}')
        print(f'pipeline time: {time.time()-start}')
        print('----------------------------')
        
        # move files to completed folder
        os.replace(file_CamA, os.path.join(finished_folder, os.path.basename(file_CamA)))
        os.replace(file_CamB, os.path.join(finished_folder, os.path.basename(file_CamB)))
        
        if len(times)>=5:
            raise ValueError
              
    else:
        print(f'waiting for files...{len(files_all_CamA)} file(s)  time: {time.time()-start_loop_time}', end='\r') #end='\r' will prevent generating a new line and will overwrite this line over and over
        time.sleep(delay)

#%% timing
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

mt = []
for t in times, read_times, preprocess_times, network_times, postprocess_times:
    mt.append(np.array(t).mean())
    
rt, pt, nt, ppt = mt[1], mt[2], mt[3], mt[4]
plt.barh(0,rt+pt+nt+ppt, label=f'postprocess ({ppt.round(3)} s)', color='C3')
plt.barh(0,rt+pt+nt, label=f'forward pass ({nt.round(3)} s)', color='C2')
plt.barh(0,rt+pt, label=f'preprocess ({pt.round(3)} s)', color='C1')
plt.barh(0,rt, label=f'read image ({rt.round(3)} s)', color='C0')
# plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=4)
plt.legend()
plt.xlabel('Time (s)')
plt.xlim([0, rt+pt+nt+ppt])
plt.yticks([])
plt.title(f'YOLO Pipeline Time for 1 TIF Stack (100 tiles)\nTotal Pipeline = {(rt+pt+nt+ppt).round(3)} s')
#%%
# plt.savefig('/home/nel/Desktop/tcell_timing_stack.pdf', dpi=300, transparent=True)