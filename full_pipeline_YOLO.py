#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 10:12:21 2021

@author: jimmytabet
"""

#%% installation instructions
'''
conda create -n sm_yolo_pipeline python, spyder, pandas, opencv, tqdm, matplotlib, seaborn, scikit-image
conda activate sm_yolo_pipeline
conda install pytorch torchvision=0.10.0=py39cuda111hcd06603_0_cuda torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
'''

#%% imports
import os
import pandas as pd
from skimage import io
import torch

#%% user settings
path_to_yolo_repo = '/home/nel/Software/yolov5'
path_to_weights = '/home/nel/Software/yolov5/runs/train/exp20/weights/best.pt'

path_to_stack = '/home/nel/Desktop/yolo_pipeline/Image_stack.tif'

stage = 'anaphase'
thresh = 0.2

#%% load yolo model
model = torch.hub.load(path_to_yolo_repo, 'custom', path=path_to_weights, source='local')

#%% run through stack
import time

s = time.time()
runs = 10

for _ in range(runs):
    stack_file_name = os.path.basename(path_to_stack).split('.tif')[0]
    # input stack
    input_stack = io.imread(path_to_stack)
    
    # convert uint16 to uint8 for yolo
    norm = ((input_stack-input_stack.min(axis=(1,2))[:,None,None])/(input_stack.max(axis=(1,2))[:,None,None]-input_stack.min(axis=(1,2))[:,None,None])*(2**8-1)).astype('uint8')
    norm = list(norm) # must be passed as list of images for yolo
    
    # pass through yolo
    res = model(norm)
    
    # get output as list of pandas df's
    output = res.pandas().xywh
    
    # add slice ID
    for i, df in enumerate(output):
        df['slice']=i
    
    # merge output df's
    res1 = pd.concat(output)
    # add file name
    res1['fname'] = stack_file_name
    # extract useful info
    res1 = res1[['fname','slice','xcenter','ycenter','confidence','name']]
    
    # mask only results from correct stage confidence above thresh value
    res1 = res1[(res1.name == stage) & (res1.confidence >= thresh)]
    
    # remove cell stage to save to csv
    all_info = res1.drop(columns='name').to_numpy()

e = time.time()
print('avg stack time:', (e-s)/runs)
print('avg image time:', (e-s)/runs/len(input_stack))

# import matplotlib.pyplot as plt
# for _,cell in res1.iterrows():
#     plt.cla()
#     plt.imshow(input_stack[cell.slice], cmap='gray')
#     plt.scatter(cell.xcenter, cell.ycenter)
#     plt.pause(0.5)