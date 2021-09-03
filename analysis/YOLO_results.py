#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:35:56 2021

@author: jimmytabet
"""

#%% imports
import cv2, os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

# for proper handling of file names with numbers
from tkinter import Tcl

#%% load YOLO results
results = '/home/nel/Software/yolov5/runs/detect/exp23/labels'
results_files = [os.path.join(results, i) for i in os.listdir(results)]
# sort each file by number to match with test file list
yolo_files = Tcl().call('lsort', '-dict', results_files)
yolo_classes = ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']

#%% load test annotation results
files = list(np.load('/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/YOLO_test_files_0830.npy'))

#%% if there are missing YOLO result files (empty tile), create blank ones
if len(yolo_files) != len(files):
    print('fixing file number mismatch (YOLO did not find any cells in some tiles)')
    for i in range(len(files)):
        fname = os.path.join(results, f'{i}.txt')
        if not os.path.isfile(fname): 
            print(f'creating {fname}')
            with open(fname, 'w') as f:
                pass
    
#%% loop through test files
# store auc value for all classes to calculate mean
AUC = []

for test_class in ['prophase']:#yolo_classes:

    print(test_class)
    
    # gt_pro = number of ground truth cells in each tile (0,1,2,....)
    gt_pro = []
    # YOLO_max = maximum YOLO score for cell type in tile
    YOLO_max = []
    # center_dist = distance between max value location from YOLO and ground truth cell
    center_dist = []
    
    # go through each yolo result file and test file/tile    
    for count, (yolo_file, file) in enumerate(zip(yolo_files, files)):
        
        if count % 100 == 0:
            print(f'{count} of {len(files)}')
        
        # load test data
        with np.load(file, allow_pickle=True) as dat:
            raw = dat['raw']
            mask = dat['masks']
            stages = dat['labels']
        
        
        # find number of gt prophase cells
        pro_id = np.where(np.array(stages) == test_class)[0]+1
        gt_pro.append(len(pro_id))
    
        # load YOLO data
        '''
        txt file: prediction, centroidX, centroidY, bounding_box_height, bounding_box_width, score) 
        '''
        res = np.loadtxt(yolo_file)
        
        # add dimension if only one cell found (or if no cells found at all)
        if res.ndim == 1:
            res = res[None,:]
        
        # if no cells found at all, set classes to empty list
        if res.size == 0:
            classes=[]
        
        # otherwise... 
        else:        
            # convert YOLO prediction (integer) into class stage ('anaphase', 'prophase', etc)
            classes = np.array([yolo_classes[int(i)] for i in res[:,0]])
    
        # if a YOLO result file does not contain a cell we are looking for, set the "maximum score" to negative infinity
        # ... and the x/y coordinates to infinity (so distance is also infinity)
        if not test_class in classes:
            pro_max = -np.inf
            YOLO_x, YOLO_y = np.inf, np.inf
            
        # if there is a cell we are looking for in the YOLO results file...
        else:
            # pull out scores for all cells
            scores = np.array(res[:,-1])
            # set score for cells not in the class of interest to negative infinity
            scores[classes!=test_class] = -np.inf
            # find the maximum score for the cell we are interested in
            pro_max = scores.max()
            # convert centroid x/y from relative (YOLO output: 0-1) to location in 800x800
            centroids = raw.shape[0]*res[:,1:3]
            # get centroid for highest YOLO score
            YOLO_x, YOLO_y = centroids[np.argmax(scores)]
    
        # append maximum YOLO score for cell of interest
        YOLO_max.append(pro_max)
    
        # continue if no gt cells in tile (distance = inf)
        if len(pro_id) == 0:
            center_dist.append(np.inf)
            continue
        
        # calc gt cell centroid(s)
        dists = []
        # loop through every ground truth cell to find closest distance
        for mask_id in pro_id:
    
            # get moments for cell to calculate center of mass
            M = cv2.moments(1*(mask==mask_id), binaryImage=True)
       
            # cX,cY = gt centroid
            if M["m00"] != 0:
              cX = M["m10"] / M["m00"]
              cY = M["m01"] / M["m00"]
            else:
              cX, cY = 0, 0 
        
            # calculate distance between gt centroid and YOLO output
            dist = np.linalg.norm([cX-YOLO_x, cY-YOLO_y])
            dists.append(dist)
                
        # keep shortest distance
        center_dist.append(min(dists))
    
    ##%% convert results from all tiles to arrays
    gt_pro = np.array(gt_pro)
    YOLO_max = np.array(YOLO_max)
    center_dist = np.array(center_dist)
    
    ##%% calculate TPR/FPR using given distance threshold
    dist_thresh = 100
    TPR_all = []
    FPR_all = []
    
    # precision_all = []
    # recall_all = []
    
    # increase threshold value between 0 and 1
    for pro_thresh in (np.logspace(0,np.log10(2),1000)-1):#np.linspace(0,1,1001):
            
        # YOLO_pro turns YOLO_max into boolean if cell in tile is above threshold
        YOLO_pro = YOLO_max > pro_thresh
        
        fp = 0
        fn = 0
        tp = 0
        tn = 0
        # oor = 0
        
        # YOLO = bool if given cell found from YOLO
        # gt = (0,1,2,...) = number of ground truth cells in tile
        # dis = distance between ground truth cell and YOLO cell
        for YOLO, gt, dis in zip(YOLO_pro, gt_pro, center_dist):
            # if no ground truth cells...
            if gt == 0:
                # if no YOLO cells found, true negative
                if not YOLO:
                    tn += 1
                # if YOLO cell found (but no gt), false positive
                else:
                    fp += 1
                    
            # if there is ground truth cell...
            elif gt > 0:
                # if no YOLO cell found, false negative
                if not YOLO:
                    fn += 1
                # if YOLO cell found AND distance to gt cell is below threshold, true positive
                elif dis < dist_thresh:
                    tp += 1
                # else (YOLO cell found but it is greater than distance threshold), false positive + false negative
                else:
                    # oor += 1
                    fp += 1
                    fn += 1
                    
            else:
                raise ValueError('gt value is negative')
                    
        # print('thresh:', pro_thresh)
        # print('tp:', tp)
        # print('tn:', tn)
        # print('fp:', fp)
        # print('fn:', fn)
        # print('OOR:', oor)
        # print('tpr:', tp/(tp+fn))
        # print('fpr:', fp/(fp+tn))
        
        # print('precision:', tp/(tp+fp))
        # print('recall:', tp/(tp+fn))
        
        # print()
        
        # if fp/(fp+tn) < 10**-3:
        #     print(pro_thresh)
        #     break
        
        # append results across all threshold values        
        TPR_all.append(tp/(tp+fn))
        FPR_all.append(fp/(fp+tn))

        # precision_all.append(tp/(tp+fp))
        # recall_all.append(tp/(tp+fn))
        
    ##%% short confusion
    # print('       actual')
    # print('        +  -')
    # print('pred +',tp,fp)
    # print('     -',fn,tn)  
     
    ##%% plot ROC curve for class
    plt.plot((FPR_all), (TPR_all), label=f'YOLO (thresh = 0.25) {test_class}: AUC = {metrics.auc(FPR_all, TPR_all).round(3)}', alpha = 0.5)
    # plt.plot(np.log(FPR_all), np.log(TPR_all), label=f'{test_class}: AUC = {metrics.auc(FPR_all, TPR_all).round(3)}')
    
    # append AUC score for class
    AUC.append(metrics.auc(FPR_all, TPR_all))

#%% add title, limits, and axis info
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0,1],[0,1], ls='--', c='k')
# # plt.xlim([-6.5, .1])
# # plt.ylim([-.78, .03])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
# plt.title(f'Prophase ROC, AUC = {metrics.auc(FPR_all, TPR_all).round(3)}')
# plt.title(f'ROC Curve, Mean AUC = {np.mean(AUC).round(3)}')