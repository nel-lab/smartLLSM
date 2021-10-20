#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 09:46:24 2021

@author: jimmytabet
"""

#%% imports
import cv2, os, time, platform, glob, csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('-----------------------------------------------')

#%% user setup

stage_of_interest = 'prophase'
thresh = 0.7

folder_to_watch = '/home/nel/Desktop/Smart Micro/full_pipeline_FCN_test/watch folder'
nn_path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/FCN_models/2021-08-24/anaphase_blank_blurry_edge_interphase_metaphase_prometaphase_prophase_telophase.h5'

# number of files to analyze at a time
'''
only works with one file at a time for now!
'''
BATCH_SIZE = 1
# delay in seconds between checking folder
delay = 5

# stitch stack
stitch = False
overlap = 255

# visualize results
visualize_results = False

# order cells by score to set thresh
set_thresh = False
# minuimum score to show
set_thresh_thresh = 0.2

#%% model setup
label = nn_path.split('.')[-2].split('/')[-1].split('_')

train_shape = 200
half_size = train_shape//2

def get_conv(input_shape=(200, 200, 1), filename=None):

    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape), #(None, None, 1)),#X_train.shape[1:])),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, kernel_size=6, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(len(label), kernel_size=1, activation='softmax'),
    # tf.keras.layers.Conv2D(len(np.unique(y_train)), kernel_size=1),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.GlobalMaxPooling2D(),
    # tf.keras.layers.Activation('softmax')
    ])

    if filename:
        model.load_weights(filename)
    return model

nn_model = get_conv(input_shape=(None, None, 1), filename=nn_path)

#%% finish setup
# interest column
filter_col = label.index(stage_of_interest)

# date and csv path
current_date = time.strftime('%m_%d_%y')
results_csv = os.path.join(folder_to_watch, f'results_{current_date}.csv')

# create completed folders if not already
os.makedirs(os.path.join(folder_to_watch, 'completed', current_date), exist_ok=True)

# set folder for finished images
finished_folder = os.path.join(folder_to_watch, 'completed', current_date)

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

def run_pipeline(files, nn_model, half_size, filter_col, thresh, results_csv,
                 stitch = False, overlap = 255,
                 viz=False, thresh_folder=False, set_thresh_thresh=0.2):
    
    read_start = time.time()
    
    # read in image
    raw = io.imread(files).astype(float)
    
    read_times.append(time.time()-read_start)
    
    preprocess_start = time.time()
    
    # add extra dimension if not tif stack
    if raw.ndim == 2:
        individual = True
        raw = raw[np.newaxis,...]
    else:
        individual = False
        
        # convert stack to stitch 
        if stitch:
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
    
    # preprocess raw image
    raw_bigger = np.zeros([raw.shape[0], raw.shape[1]+2*half_size, raw.shape[2]+2*half_size], dtype=float)
    
    for i, im in enumerate(raw):
        im -= im.mean()
        im /= im.std()
        raw_bigger[i] = cv2.copyMakeBorder(im, half_size, half_size, half_size, half_size, 0)
    
    raw_bigger = raw_bigger[...,np.newaxis]
    
    preprocess_times.append(time.time()-preprocess_start)
    
    netowrk_start = time.time()
    
    # forward pass
    res = nn_model.predict(raw_bigger)
    
    network_times.append(time.time() - netowrk_start)
        
    # isolate interest heatmap
    interest = res[:,:,:,filter_col]
    
    # mask for tiles above thresh
    max_interest = interest.max(axis=(1,2))
    mask = max_interest>thresh
    
    # interpolate to get centroids
    cell_centroid_mask = np.zeros([sum(mask), 2])
    
    # exponential interpolation
    for i, im in enumerate(interest[mask]):
        ms_h = 0
        ms_w = 0
        
        top_left = cv2.minMaxLoc(im)[-1]
        sh_y, sh_x = top_left
        
        # default to max if on border
        if sh_y == 0 or sh_y == im.shape[0] or sh_x == 0 or sh_x == im.shape[0]:
          sh_y_n = sh_y
          sh_x_n = sh_x
          
        else:
            # peak registration
            log_xm1_y = np.log(im[sh_x - 1, sh_y])
            log_xp1_y = np.log(im[sh_x + 1, sh_y])
            log_x_ym1 = np.log(im[sh_x, sh_y - 1])
            log_x_yp1 = np.log(im[sh_x, sh_y + 1])
            four_log_xy = 4 * np.log(im[sh_x, sh_y])
        
            sh_x_n = (sh_x - ms_h + (log_xm1_y - log_xp1_y)/(2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
            sh_y_n = (sh_y - ms_w + (log_x_ym1 - log_x_yp1)/ (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
            
        # heatmap to raw image factor
        factor = raw.shape[1]/im.shape[1]
        
        # store center
        cell_centroid_mask[i] = (factor*sh_y_n, factor*sh_x_n)
            
    # store data
    file_name_mask = np.array([os.path.basename(files) for _ in range(sum(mask))])
    if individual:
        file_index_mask = ''
    else:
        file_index_mask = np.where(mask)[0]
    cell_score_mask = max_interest[mask]
    
    all_info = np.column_stack([file_name_mask, file_index_mask, cell_centroid_mask, cell_score_mask])
    
    '''
    WIP, save images for set_thresh
    # save images for set_thresh
    if thresh_folder:
        thresh_mask = preds[:,filter_col] > set_thresh_thresh
        cell_score_mask_set_thresh = preds[:,filter_col][thresh_mask]
        for img, score in zip(X_all[thresh_mask], cell_score_mask_set_thresh):
            matplotlib.image.imsave(os.path.join(thresh_folder, str((score*100).round(1))+'.tiff'), img.squeeze(), cmap='gray')
            # cv2.imwrite(os.path.join(thresh_folder, str((score*100).round(1))+'.png'), img.squeeze())
    '''
    
    # write prophase centroid to results csv
    if all_info.size:
        with open(results_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(all_info)
                
        # view results
        if viz:
            n = int(np.ceil(np.sqrt(len(file_name_mask))))
            fig = plt.figure()
            count = 1
            for (_, idx, cx, cy, _) in all_info:
                '''
                WIP, need to update if multiple files passed in
                '''
                if stitch:
                    im = raw
                else:
                    im = io.imread(files)
                    if idx:
                        im = im[int(idx)]
                
                ax = fig.add_subplot(n,n,count)
                ax.set_title(f'{os.path.basename(files)} {idx}')
                ax.axis('off')
                ax.imshow(im.squeeze(), cmap='gray',
                          vmin=np.percentile(im,1), vmax=np.percentile(im,99))
                ax.scatter(float(cx),float(cy), c='r', marker='*')
                count += 1
            # pause to show image while pipeline runs
            plt.pause(0.1)            
        
        return cell_score_mask.max()
    
    # just write file name       
    else:
        with open(results_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(np.array(os.path.basename(files)).reshape([-1,1]))
        
        return False
    
#%% watch folder
start_loop_time = time.time()

times = []

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
        cell_found = run_pipeline(files_analyzed, nn_model, half_size, filter_col, thresh, results_csv,
                                  stitch=stitch, overlap=overlap,
                                  viz=visualize_results, thresh_folder=thresh_folder, set_thresh_thresh=set_thresh_thresh)
        
        times.append(time.time()-start)
        if len(times)>=5:
            raise ValueError
        
        # print(f'cell found: {cell_found}')
        # print(f'pipeline time: {time.time()-start}')
        # print('----------------------------')
        
        # move files to completed folder
        # os.replace(files_analyzed, os.path.join(finished_folder, os.path.basename(files_analyzed)))
              
    else:
        print(f'waiting for files...{len(files_all)} file(s)  time: {time.time()-start_loop_time}', end='\r') #end='\r' will prevent generating a new line and will overwrite this line over and over
        time.sleep(delay)
        
#%% timing
mt = []
for t in times, read_times, preprocess_times, network_times:
    mt.append(np.array(t).mean())
    
rt, pt, nt = mt[1], mt[2], mt[3]
plt.barh(0,rt+pt+nt, label=f'forward pass ({nt.round(3)} s)', color='C2')
plt.barh(0,rt+pt, label=f'preprocess ({pt.round(3)} s)', color='C1')
plt.barh(0,rt, label=f'read image ({rt.round(3)} s)', color='C0')
plt.legend()
plt.xlabel('Time (s)')
plt.xlim([0, rt+pt+nt])
plt.yticks([])
plt.title(f'FCN Pipeline Time for 1 TIF Stack (100 tiles)\nTotal Pipeline = {(rt+pt+nt).round(3)} s')