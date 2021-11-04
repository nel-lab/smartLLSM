#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 15:32:09 2021

@author: jimmytabet
"""

#%% setup
import numpy as np
from utils import *

BS = 16
S = 4
B = 3
C = 3
P = 5

'''
at each cell location, output = 
C1, C2, C3, x1, y1, a1, b1, t1, R1, x2, y2, a2, b2, t2, R2
'''

X = []
y = []

# optional, store y_pred result
# y_pred = y_true randomly offset by 10%
yp = []

for i in range(BS):
    X_temp, y_temp = read_data(f'/home/nel/Desktop/YOLOv1_ellipse/train_data/{i}.npz')
    X.append(X_temp)
    y.append(y_temp)
    
    # optional y_pred result
    _, yp_temp = read_data(f'/home/nel/Desktop/YOLOv1_ellipse/train_data/{i}.npz', ypred=True)
    yp.append(yp_temp)
    
    plt.figure()
    show_results(X_temp, y_temp)

X = np.array(X)
y_true = np.array(y)

# y_pred results
y_pred = np.array(yp)
# add random noise between [-0.05,0.05]
y_pred += 0.05*(2*np.random.rand(*y_pred.shape)-1)
# randomly offset by 10%
y_pred *= 1 + 0.1*(2*np.random.rand(*y_pred.shape)-1)

# totally random y_pred
# y_pred = np.random.rand(*y_true.shape)

# y_true = np.arange((16*4*4*21)).reshape((16, 4, 4, 21))/100
# y_pred = np.arange((16*4*4*21)).reshape((16, 4, 4, 21))/50

#%% total yolo loss
l_total = yolo_loss(y_true, y_pred)
print(f'yolo loss:  \t {l_total.round(5)}')

#%% test losses
# print('-'*21)
# clas_true, pms_true, rs_true = process(y_true)
# clas_pred, pms_pred, rs_pred = process(y_pred)

# # obj mask
# rs_mask = rs_true[...,0] == 1

# # iou/bb mask
# loc_mask = np.argwhere(rs_mask==True)
# iou_mask = iou(pms_true[rs_mask], pms_pred[rs_mask], loc_mask[:,-2:])
# rs_iou_mask = np.column_stack([loc_mask, iou_mask])
# pms_pred_masked = pms_pred[tuple(rs_iou_mask.T)]
# # take first bounding box as ground truth
# pms_true_masked = pms_true[rs_mask][:,0,:]

# # response mask
# rs_pred_masked = rs_pred[tuple(rs_iou_mask.T)] 

# # response mask no object
# rs_pred[tuple(rs_iou_mask.T)] = 0

# #  center loss test - DONE
# l1 = center_loss(pms_true_masked[:,:2], pms_pred_masked[:,:2])
# print(f'center loss:\t {l1.round(5)}')

# # params loss test - DONE
# l2 = params_loss(pms_true_masked[:,2:], pms_pred_masked[:,2:])
# print(f'params loss:\t {l2.round(5)}')

# # obj loss test - DONE
# l3 = obj_loss(rs_pred_masked)
# print(f'obj loss:   \t {l3.round(5)}')

# # no obj loss test - DONE
# l4 = no_obj_loss(rs_pred)
# print(f'no obj loss:\t {l4.round(5)}')

# # class loss test - DONE
# l5 = class_loss(clas_true[rs_mask], clas_pred[rs_mask])
# print(f'class loss: \t {class_loss(clas_true[rs_mask], clas_pred[rs_mask]).round(5)}')

# assert l1+l2+l3+l4+l5 == l_total
