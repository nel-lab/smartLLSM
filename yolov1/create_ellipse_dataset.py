#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:28:35 2021

@author: jimmytabet
"""

#%% generate binary ellipse dataset
import os, glob, cv2
import numpy as np
import matplotlib.pyplot as plt
# from tqdm import tqdm

# grid size
grid = 800
# number of images to generate
num_images = 10000
# number of cells per image

# visualize dataset
viz = False
min_cells, max_cells = 1,4

# label dictionary
label_dict = {0: 'small',
              1: 'medium',
              2: 'large'}

# remove old examples
base_folder = '/home/nel/Desktop/YOLOv1_ellipse'
old_files = glob.glob(os.path.join(base_folder, '**', '*.npz'), recursive=True)
[os.remove(fil) for fil in old_files]
               
# loop over num_images
for i in range(num_images):
    if i%100:
        print(i)
    num_cells = np.random.randint(min_cells, max_cells)

    # init arrays
    X = np.zeros([grid,grid,1], dtype=np.float32)
    y = np.zeros([num_cells,6], dtype = np.float32)
    
    # create cells and add to iamge
    for j in range(num_cells):
        # ellipse parameters
        cx = np.random.randint(100, grid-100)
        cy = np.random.randint(100, grid-100)
        a = np.random.randint(50,100)
        b = np.random.randint(50,100)
        theta = np.random.randint(90)
        
        # create X (image)
        X = cv2.ellipse(X, (cx,cy), (a,b) ,theta, 0, 360, 1, -1)
        
        # create label based on area (0: small, 1: medium, 2: large)
        # check pi*a*b < [pi*50^2 + (.../3)*(pi*100^2 - pi*50^2)]
        # ---->    a*b < [   50^2 + (.../3)*(   100^2 -    50^2)]
        if a*b < (50**2 + (1/3)*(100**2-50**2)):
            label = 0
        elif a*b < (50**2 + (2/3)*(100**2-50**2)):
            label = 1
        else:
            label = 2
        
        # # create label based on axes lengths (0: small, 1: medium, 2: large)
        # if (a<75 and b<75):
        #     label = 0
        # elif (a<75 or b<75):
        #     label = 1
        # else:
        #     label = 2
        
        # store y (ellipse parameters + label)
        y[j] = np.array([cx,cy,a,b,theta,label], dtype=np.float32)

    # show image with labels
    if viz and i<10:
        plt.cla()
        plt.imshow(X, cmap='gray')
        for (ex,ey,*_,l) in y:
            plt.text(ex,ey,label_dict[l], ha='center', size='x-small')
        plt.axis('off')
        plt.pause(.5)

    # save training data
    if i < int(0.7*num_images):
        np.savez(f'/home/nel/Desktop/YOLOv1_ellipse/train_data/{i}', X=X, y=y)
    # save validation data
    elif i < int(0.9*num_images):
        np.savez(f'/home/nel/Desktop/YOLOv1_ellipse/val_data/{i}', X=X, y=y)
    # save testing data
    else:
        np.savez(f'/home/nel/Desktop/YOLOv1_ellipse/test_data/{i}', X=X, y=y)