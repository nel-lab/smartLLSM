#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 11:13:21 2021

@author: jimmytabet
"""

#%% imports
import os,glob,random,datetime
import numpy as np
from scipy import ndimage
import matplotlib

#%% split Position folders into train and test

path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
all_files = sorted(glob.glob(os.path.join(path,'**','*.npz'), recursive=True))

data_1 = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results/data_1_cellpose'
data_2 = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results/data_2_cellpose'

data_1_folders = os.listdir(data_1)
data_2_folders = os.listdir(data_2)

d1 = [os.path.join(data_1, i) for i in data_1_folders]
d2 = [os.path.join(data_2, i) for i in data_2_folders]

position_folders = d1+d2

# randomize
random.shuffle(position_folders)

train_files = []
test_files = []

i = 0
while len(train_files) <= int(.7*len(all_files)):
    files = glob.glob(os.path.join(position_folders[i], '*.npz'))
    train_files += files
    i += 1

for folder in position_folders[i:]:
    files = glob.glob(os.path.join(folder, '*.npz'))
    test_files += files

#%% set classes
classes = ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']

#%% loop through train and test - takes ~3.5 min
import time

start = time.time()

bb_size = 200

base_folder = '/home/nel/Software/yolov5/smart_micro_datasetsv2'

for mode in ['train', 'test']:
    
    print(mode.upper())

    if mode == 'train':
        files = train_files
    elif mode == 'test':
        files = test_files
    else:
        raise ValueError('not train or test mode!')
            
    count = 0
    for file in files:
        dat = np.load(file, allow_pickle=True)
    
        raw = dat['raw']
        masks = dat['masks']
        labels = dat['labels']
        
        matplotlib.image.imsave(os.path.join(base_folder, 'images', mode, f'{count}.jpg'), raw, cmap='gray', dpi=300)
        
        num_masks = masks.max()
        
        with open(os.path.join(base_folder, 'labels', mode, f'{count}.txt'),'w+') as f:
        
            for idx, mask_id in enumerate(range(1,num_masks+1)):
                
                if labels[idx] not in classes:
                    continue
                
                # find center of mass of cell
                center = ndimage.center_of_mass(masks==mask_id)
                center = np.array(center)/raw.shape[0]
                
                class_id = classes.index(labels[idx])
                x_center = center[1]
                y_center = center[0]
                
                output = f'{class_id} {x_center} {y_center} {bb_size/raw.shape[0]} {bb_size/raw.shape[0]}\n'
                f.write(output)
                
        count += 1
        
        if count%100 == 0:
            print(f'{count}\tof\t{len(files)}')
            
        if count == len(files):
            print(f'{count}\tof\t{len(files)}\nDONE')

print(time.time()-start)
        
#%% save train and test files
np.save(f'/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/YOLO_train_files_{datetime.datetime.now().strftime("%m%d")}', train_files)
np.save(f'/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/YOLO_test_files_{datetime.datetime.now().strftime("%m%d")}', test_files)
