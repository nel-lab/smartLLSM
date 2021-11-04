#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:28:35 2021

@author: jimmytabet
"""

#%% generate binary ellipse dataset
import cv2
import numpy as np
import matplotlib.pyplot as plt

grid = 800 # grid size
num_images = 100

# visualize dataset
viz = False

# label dictionary
label_dict = {0: 'small',
              1: 'medium',
              2: 'large'}

# loop over num_images
for i in range(num_images):

    # make between 3 and 9 cells
    num_cells = np.random.randint(3,10)

    # init arrays
    X = np.zeros([grid,grid,1])
    y = np.zeros([num_cells,6])
    
    # create cells and add to iamge
    for j in range(num_cells):
        # ellipse parameters
        cx = np.random.randint(grid)
        cy = np.random.randint(grid)
        a = np.random.randint(50,100)
        b = np.random.randint(50,100)
        theta = np.random.randint(360)
        
        # create X (image)
        X = cv2.ellipse(X, (cx,cy), (a,b) ,theta, 0, 360, 1, -1)
        
        # create label based on axes lengths (0: small, 1: medium, 2: large)
        if (a<75 and b<75):
            label = 0
        elif (a<75 or b<75):
            label = 1
        else:
            label = 2
        
        # store y (ellipse parameters + label)
        y[j] = np.array([cx,cy,a,b,theta,label])

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