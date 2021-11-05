#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 16:00:24 2021

@author: jimmytabet
"""

#%% imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K

#%% global variables
IMG_SIZE = 800
S = 7
B = 2
C = 3
P = 5

# label dictionary
LABEL_DICT = {0: 'small',
              1: 'medium',
              2: 'large'}

# threshold for bounding box detection response
DETECTION_THRESH = 0

#%% read and show data functions
# read_data function
def read_data(data_path, ypred = False):
    # load image and label
    with np.load(data_path) as f:
        image = f['X']
        label = f['y']
    
    # get image size and resize to input shape
    image_h, image_w = image.shape[:2]
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # ensure image has 3 dimensions
    if image.ndim != 3:
        image = image[...,np.newaxis]
        
    # convert image to 0-1
    '''
    below code does not work with binary images, try again for actual images
    '''
    # image_min = image.min(axis=(1,2))[:,None,None]
    # image_max = image.max(axis=(1,2))[:,None,None]    
    # image = (image-image_min)/image_range
    # image -= image_min
    # image /= image_range

    image = (image-image.min())/(image.max()-image.min())
    
    # create label_matrix for training YOLO
    label_matrix = np.zeros([S, S, C + B*(P+1)])
   
    for l in label:
        # convert label to integers
        l = l.astype(int)
        # extract ellipse parameters and target class from label
        x, y, a, b, theta, target_cls = l
        
        # normalize parameters to unit height/width/rot
        x /= image_w
        y /= image_h
        a /= image_w
        b /= image_h
        theta = theta % 180 # theta = [0-180]
        theta /= 180        # theta = [0-1]
        
        # convert x and y to grid location
        loc = [S * x, S * y]
        # grid location of label_matrix
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        # relative unit location inside grid
        y = loc[1] - loc_i
        x = loc[0] - loc_j
        
        # check if response has been recorded at grid location for number of bounding boxes
        if label_matrix[loc_i, loc_j, C+P] == 0:
            
            # one-hot label
            label_matrix[loc_i, loc_j, target_cls] = 1
            
            # store ellipse params
            label_matrix[loc_i, loc_j, C:C+P] = [x, y, a, b, theta]
            
            # response
            label_matrix[loc_i, loc_j, C+P] = 1
            
            # if storing ypred, add the same results for other bounding boxes (will offset randomly later)
            if ypred:
                for i in range(1,B):
                    start_idx = C+i*(P+1)
                    end_idx = start_idx + P+1
                    label_matrix[loc_i, loc_j, start_idx:end_idx] = [x, y, a, b, theta, 1]

    return image, label_matrix

# show_results fuction
def show_results(image, label, thresh = DETECTION_THRESH):
    
    S = label.shape[0]
    
    # create copy of image for drawing ellipse
    im = np.copy(image).astype(image.dtype)
        
    # get image shape to revert label outputs
    image_h, image_w = im.shape[:2]
    
    # loop over all grid locations and draw ellipse/label
    for loc_i in range(S):
        for loc_j in range(S):
            
            grid_outputs = label[loc_i, loc_j, :]
            
            # reshape label_matrix to Bx(C+P+1) (num_bb x num_class + num_params + response)
            bb_outputs = label[loc_i, loc_j, C:].reshape(B, P+1)
            
            # extract bounding box results if response above thresh
            if bb_outputs[:,-1].max() > thresh:
                # chose bounding box with max response
                bb_output_response = bb_outputs[np.argmax(bb_outputs[:,-1])]
            
                # extract ellipse parameters
                x,y,a,b,theta = bb_output_response[:-1]

                # convert x,y,a,b,theta back into image coordinates                
                x += loc_j
                x /= S                
                x *= image_w                
                
                y += loc_i
                y /= S
                y *= image_h                
                
                a *= image_w
                b *= image_h
                theta *= 180    
                
                x = x.astype(int)
                y = y.astype(int)
                a = a.astype(int)
                b = b.astype(int)
                theta = theta.astype(int)
                
                # draw ellipse outline in gray
                '''
                fill doesn't work for some images (thickness = -1)?
                use thickness > 0 or use plots of major and minor axis instead
                '''
                im = cv2.ellipse(im, (x,y), (a,b) ,theta, 0, 360, .5, 3)
                # plt.plot([x, x+a*np.cos(np.radians(theta))], [y, y+a*np.sin(np.radians(theta))], color='r')
                # plt.plot([x, x+b*np.cos(np.radians(theta+90))], [y, y+b*np.sin(np.radians(theta+90))], color='b')
                        
                # extract label and add to image
                output = LABEL_DICT[np.argmax(grid_outputs[:C])]
                plt.text(x,y,output, ha='center', size='x-small')
    
    # show annotated image
    plt.imshow(im, cmap='gray')
    plt.axis('off')

#%% loss function + helpers
'''
ASSUMPTIONS
    pass images with the same height/width
        affects IoU calc: global_params function image_h, image_w
    only one object can be detected in each grid
        can make output [BS, S, S, B*(C+P+1)]
'''

'''
at each cell location, output = 
C1, C2, C3, x1, y1, a1, b1, t1, R1, x2, y2, a2, b2, t2, R2, ...
'''

def process(output):
    '''
    output: [BS, S, S, C+B*(P+1)]
    
    classes: [BS, S, S, C]
    params: [BS, S, S, B, P]
    response: [BS, S, S, B]
    '''
    
    classes = output[...,:3]
    params_all = output[...,C:]
    params_all = K.reshape(params_all, (-1,S,S,B,P+1))
    params = params_all[...,:-1]
    response = params_all[...,-1]
    
    return classes, params, response

def global_params(params, loc_i, loc_j, image_w=IMG_SIZE, image_h=IMG_SIZE):
    '''
    params: [P,]
        x, y, a, b, t
    
    global_params: [P,] (with image coords)
    '''
    
    # ensure all paramters are numpy to use in cv2.ellipse
    x,y,a,b,theta = params.numpy()
    loc_i = loc_i.numpy()
    loc_j = loc_j.numpy()
    
    # convert x,y,a,b,theta back into image coordinates                
    x += loc_j
    x /= S                
    x *= image_w                
    
    y += loc_i
    y /= S
    y *= image_h                
    
    a *= image_w
    b *= image_h
    theta *= 180    
    
    # need to be ints for cv2.ellipse
    x = int(x)
    y = int(y)
    
    a = int(a)
    b = int(b)
    theta = int(theta)
    
    # sanity check, a/b cannot be negative
    a = max(a, 0)
    b = max(b,0)
    
    global_params = [x,y,a,b,theta]
    
    return global_params

def iou(true_params_mask, pred_params_mask, loc_idx, viz=False):
    '''
    true/pred_mask: [number_objects, B, P]
    loc_idx: [number_objects, 2]
        specifies grid location of object to draw ellipses in image coords

    bb_idx: [number_objects,]
        index of bounding box responsible for prediction (highest IoU)
    '''

    number_objects = K.shape(true_params_mask)[0]

    ious = np.zeros([number_objects, B])

    for obj in range(number_objects):
        # true params for object
        true_params = true_params_mask[obj,0,:]
        
        # loc_i, loc_j for grid location
        loc_i, loc_j = loc_idx[obj]
        
        # loop over each proposed bounding box
        for bb in range(B):
                  
            # extract true paramters in image coordinates
            true_x, true_y, true_a, true_b, true_theta = global_params(true_params, loc_i, loc_j)
                        
            # draw ellipse
            ellipse_true = cv2.ellipse(np.zeros((IMG_SIZE,IMG_SIZE)), (true_x,true_y), (true_a,true_b), true_theta, 0, 360, 1, -1)
        
            # pred params for object and bounding box
            pred_params = pred_params_mask[obj,bb]
            # extract pred paramters in image coordinates
            pred_x, pred_y, pred_a, pred_b, pred_theta = global_params(pred_params, loc_i, loc_j)
            ellipse_pred = cv2.ellipse(np.zeros((IMG_SIZE,IMG_SIZE)), (pred_x,pred_y), (pred_a,pred_b), pred_theta, 0, 360, 1, -1)
            
            # draw combined
            combined = ellipse_pred + ellipse_true
            # calc intersection
            intersection = K.sum((combined == 2).astype(int))
            # calc union
            union = K.sum((combined > 0).astype(int))
                 
            # visualize to debug
            if viz:
                plt.cla()
                plt.imshow(combined, cmap='gray')
                plt.title(f'Obj: {obj}, B: {bb}, IOU: {intersection/union}')
                plt.pause(.5)
            
            # store iou value
            ious[obj,bb] = intersection/union
        
    bb_idx = K.argmax(ious, axis=-1)                
    
    return bb_idx

def center_loss(y_true, y_pred):
    '''
    y_true/pred: [BS, S, S, C + B*(P+1)]

    true/pred_mask: [number_objects, 2] (x/y coord)
    
    loss = lambda_coord*SSE:
        lambda_coord*sum((true-pred)**2), if object in cell and bb responsible for detection
    '''
    
    lambda_coord = 5
    
    # extract parameters from output
    _, true_params, true_response = process(y_true)
    _, pred_params, _ = process(y_pred)
        
    # mask if object in cell (when B=0 in true response): [BS, S, S]
    # True if object in cell, False if no object
    obj_mask = true_response[...,0] == 1
        
    # mask which bounding box is responsible for prediction based on IoU score
    
    # grid index of object: [number_object, 3]
    # 3 columns: [truech number, grid x, grid y]
    loc_idx = tf.where(obj_mask)
        
    # index of bounding box responsible for prediction (max IoU)
    # only need last two columns of loc_idx (grid x/y) for converting to global params
    bb_idx = iou(true_params[obj_mask], pred_params[obj_mask], loc_idx[:, -2:])
    
    # concat bb_idx with loc_idx
    loc_and_bb_idx = tf.concat([loc_idx, bb_idx[...,None]], 1)
    
    # get params of responsible bounding box for center/params loss
    # tuple and transpose convert idx to sequence of arrays describing: [batch number, grid x, grid y, bb_number]
    pred_params_masked = tf.gather_nd(pred_params, loc_and_bb_idx)
    
    # ground truth params (first bounding box is ground truth)
    true_params_masked = true_params[obj_mask][:,0,:]
    
    true_center_mask = true_params_masked[:,:2]
    pred_center_mask = pred_params_masked[:,:2]
    
    # take SSE    
    loss = K.sum((true_center_mask - pred_center_mask)**2)
    loss *= lambda_coord
    
    # batch size
    batch_size = K.shape(y_true)[0]
    # convert to same dtype as loss
    batch_size = tf.cast(batch_size, loss.dtype)
        
    # average loss over batch size
    loss /= batch_size
    
    return loss

def params_loss(y_true, y_pred):
    '''
    y_true/pred: [BS, S, S, C + B*(P+1)]
    
    true/pred_mask: [number_objects, 3] (a/b/theta)
    
    loss = lambda_coord*SSE:
        lambda_coord*sum((true-pred)**2), if object in cell and bb responsible for detection
    '''
        
    lambda_coord = 5
    
    # extract parameters from output
    _, true_params, true_response = process(y_true)
    _, pred_params, _ = process(y_pred)
        
    # mask if object in cell (when B=0 in true response): [BS, S, S]
    # True if object in cell, False if no object
    obj_mask = true_response[...,0] == 1
        
    # mask which bounding box is responsible for prediction based on IoU score
    
    # grid index of object: [number_object, 3]
    # 3 columns: [truech number, grid x, grid y]
    loc_idx = tf.where(obj_mask)
        
    # index of bounding box responsible for prediction (max IoU)
    # only need last two columns of loc_idx (grid x/y) for converting to global params
    bb_idx = iou(true_params[obj_mask], pred_params[obj_mask], loc_idx[:, -2:])
    
    # concat bb_idx with loc_idx
    loc_and_bb_idx = tf.concat([loc_idx, bb_idx[...,None]], 1)
    
    # get params of responsible bounding box for center/params loss
    # tuple and transpose convert idx to sequence of arrays describing: [batch number, grid x, grid y, bb_number]
    pred_params_masked = tf.gather_nd(pred_params, loc_and_bb_idx)
    
    # ground truth params (first bounding box is ground truth)
    true_params_masked = true_params[obj_mask][:,0,:]
    
    true_param_mask = true_params_masked[:,2:]
    pred_param_mask = pred_params_masked[:,2:]
        
    # take SSE, with square root of a/b/theta as per yolo loss equation
    loss = K.sum((true_param_mask**0.5 - pred_param_mask**0.5)**2)
    
    loss *= lambda_coord
    
    # batch size
    batch_size = K.shape(y_true)[0]
    # convert to same dtype as loss
    batch_size = tf.cast(batch_size, loss.dtype)
        
    # average loss over batch size
    loss /= batch_size
    
    return loss

def obj_loss(y_true, y_pred):
    '''
    y_true/pred: [BS, S, S, C + B*(P+1)]
    
    pred_mask: [number_objects,]
        masked for object and bb responsible for detection
        true_mask = 1
        
    loss = SSE:
        sum((true-pred)**2), if object in cell and bb responsible for detection
        true = 1
    '''
    
    # extract parameters from output
    _, true_params, true_response = process(y_true)
    _, pred_params, pred_response = process(y_pred)
        
    # mask if object in cell (when B=0 in true response): [BS, S, S]
    # True if object in cell, False if no object
    obj_mask = true_response[...,0] == 1
        
    # mask which bounding box is responsible for prediction based on IoU score
    
    # grid index of object: [number_object, 3]
    # 3 columns: [truech number, grid x, grid y]
    loc_idx = tf.where(obj_mask)
        
    # index of bounding box responsible for prediction (max IoU)
    # only need last two columns of loc_idx (grid x/y) for converting to global params
    bb_idx = iou(true_params[obj_mask], pred_params[obj_mask], loc_idx[:, -2:])
    
    # concat bb_idx with loc_idx
    loc_and_bb_idx = tf.concat([loc_idx, bb_idx[...,None]], 1)
    
    # mask response for object and bounding box responsible for prediction
    pred_response_masked = tf.gather_nd(pred_response, loc_and_bb_idx)
        
    # SSE
    loss = K.sum((1 - pred_response_masked)**2)

    # batch size
    batch_size = K.shape(y_true)[0]
    # convert to same dtype as loss
    batch_size = tf.cast(batch_size, loss.dtype)
        
    # average loss over batch size
    loss /= batch_size
    
    return loss

def no_obj_loss(y_true, y_pred):
    '''
    y_true/pred: [BS, S, S, C + B*(P+1)]
    
    pred_mask: [BS, S, S, B]
        true_mask = 0
    loc_and_bb_idx: [number_objects, 4]
        4: index into [BS, S, S, B] for boxes with object and bb responsible for detection
        !! will subtract out these values (squared) from no_obj_loss,
        !! acts as method to mask response for NO object and bounding box NOT responsible for prediction
        
    loss = lambda_noobj*SSE:
        lambda_noobj*sum((true-pred)**2), if NO object in cell and bb NOT responsible for detection
        true = 0, simplifies to lambda_noobj*sum(pred**2)
    '''
    
    lambda_noobj = 0.5
    
    # extract parameters from output
    _, true_params, true_response = process(y_true)
    _, pred_params, pred_response = process(y_pred)
        
    # mask if object in cell (when B=0 in true response): [BS, S, S]
    # True if object in cell, False if no object
    obj_mask = true_response[...,0] == 1
        
    # mask which bounding box is responsible for prediction based on IoU score
    
    # grid index of object: [number_object, 3]
    # 3 columns: [truech number, grid x, grid y]
    loc_idx = tf.where(obj_mask)
        
    # index of bounding box responsible for prediction (max IoU)
    # only need last two columns of loc_idx (grid x/y) for converting to global params
    bb_idx = iou(true_params[obj_mask], pred_params[obj_mask], loc_idx[:, -2:])
    
    # concat bb_idx with loc_idx
    loc_and_bb_idx = tf.concat([loc_idx, bb_idx[...,None]], 1)
        
    # take SSE
    loss = K.sum(pred_response**2)
    
    # subtract out squared responses for cells with object and bb responsible for detection
    loss -= K.sum(tf.gather_nd(pred_response, loc_and_bb_idx)**2)
    
    loss *= lambda_noobj
    
    # batch size
    batch_size = K.shape(y_true)[0]
    # convert to same dtype as loss
    batch_size = tf.cast(batch_size, loss.dtype)
        
    # average loss over batch size
    loss /= batch_size
    
    return loss
    
def class_loss(y_true, y_pred):
    '''
    y_true/pred: [BS, S, S, C + B*(P+1)]

    true/pred_mask: [number_objects, C]
    
    loss = SSE:
        sum((true-pred)**2), if object in cell
    '''

    # extract parameters from output
    true_class, _, true_response = process(y_true)
    pred_class, _, _ = process(y_pred)
        
    # mask if object in cell (when B=0 in true response): [BS, S, S]
    # True if object in cell, False if no object
    obj_mask = true_response[...,0] == 1
    
    true_class_mask = tf.boolean_mask(true_class, obj_mask)
    pred_class_mask = tf.boolean_mask(pred_class, obj_mask)

    # take SSE    
    loss = K.sum((true_class_mask - pred_class_mask)**2)
    
    # batch size
    batch_size = K.shape(y_true)[0]
    # convert to same dtype as loss
    batch_size = tf.cast(batch_size, loss.dtype)
        
    # average loss over batch size
    loss /= batch_size
    
    return loss

def yolo_loss(y_true, y_pred):
    '''
    y_true/pred: [BS, S, S, C + B*(P+1)],
    transformed into:
        classes: [BS, S, S, C]
        params: [BS, S, S, B, P]
        response: [BS, S, S, B]

    loss = sum of individual losses/BS
    '''
                    
    # calc losses
    loss = [
        # center loss (mask for object, bb, x/y (first two coloumns))
        center_loss(y_true, y_pred),
        
        # params loss (mask for object, bb, a/b/theta (remaining three coloumns))
        params_loss(y_true, y_pred),
        
        # object loss (mask for object, bb (true = 1))
        obj_loss(y_true, y_pred),
        
        # no object loss (will mask for NO object, bb NOT responsible (true = 0))
        no_obj_loss(y_true, y_pred),
        
        # class loss (mask for object)
        class_loss(y_true, y_pred)
        ]
        
    # sum individual losses
    loss = tf.reduce_sum(loss)
    
    return loss