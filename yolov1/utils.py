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

#%% global variables
IMG_SIZE = 800
S = 4
B = 1
C = 3
P = 5

# BS = 5

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
            # if ypred:
            #     for i in range(1,B):
            #         start_idx = C+i*(P+1)
            #         end_idx = start_idx + P+1
            #         label_matrix[loc_i, loc_j, start_idx:end_idx] = [x, y, a, b, theta, 1]
            
            # if storing ypred, alter values
            if ypred:
                # one-hot label
                label_matrix[loc_i, loc_j, :C] = [ypred, ypred, 1]
                
                # store ellipse params
                label_matrix[loc_i, loc_j, C:C+P] = [x+ypred, y+ypred, a+ypred, b+ypred, theta+ypred]
                
                # response
                label_matrix[loc_i, loc_j, C+P] = 1-ypred
                
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
    
    BS = output.shape[0]
    S = output.shape[1]
    
    classes = output[..., :C]
        
    params = np.zeros([BS, S, S, B, P])
    response = np.zeros([BS, S, S, B])
    
    for bb in range(B):
        bb_idx_start = C + bb*(P+1)
        bb_idx_end = bb_idx_start + P
                
        params[...,bb,:] = output[..., bb_idx_start : bb_idx_end]
        response[...,bb] = output[..., bb_idx_end]
    
    return classes, params, response

def global_params(params, loc_i, loc_j, image_w=IMG_SIZE, image_h=IMG_SIZE):
    '''
    params: [P,]
        x, y, a, b, t
    
    global_params: [P,] (with image coords)
    '''
    
    x,y,a,b,theta = params
    
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
    x = x.astype(int)
    y = y.astype(int)
    
    a = a.astype(int)
    b = b.astype(int)
    theta = theta.astype(int)
    
    # sanity check, a/b cannot be negative
    a = max(a, 0)
    b = max(b,0)
    
    global_params = np.array([x,y,a,b,theta])
    
    return global_params

def iou(true_params_mask, pred_params_mask, loc_idx, viz=False):
    '''
    true/pred_mask: [number_objects, B, P]
    loc_idx: [number_objects, 2]
        specifies grid location of object to draw ellipses in image coords

    bb_idx: [number_objects,]
        index of bounding box responsible for prediction (highest IoU)
    '''

    number_objects = true_params_mask.shape[0]
    B = true_params_mask.shape[1]

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
            intersection = np.sum(combined == 2)
            # calc union
            union = np.sum(combined > 0)
                 
            # visualize to debug
            if viz:
                plt.cla()
                plt.imshow(combined, cmap='gray')
                plt.title(f'Obj: {obj}, B: {bb}, IOU: {intersection/union}')
                plt.pause(.5)
            
            # store iou value
            ious[obj,bb] = intersection/union

    bb_idx = np.argmax(ious, axis=-1)                

    return bb_idx

def center_loss(true_center_mask, pred_center_mask, lambda_coord = 5):
    '''
    true/pred_mask: [number_objects, 2] (x/y coord)
    
    loss = lambda_coord*SSE:
        lambda_coord*sum((true-pred)**2), if object in cell and bb responsible for detection
    '''
    
    # take SSE    
    loss = np.sum((true_center_mask - pred_center_mask)**2)
    
    return lambda_coord*loss

def params_loss(true_param_mask, pred_param_mask, lambda_coord = 5):
    '''
    true/pred_mask: [number_objects, 3] (a/b/theta)
    
    loss = lambda_coord*SSE:
        lambda_coord*sum((true-pred)**2), if object in cell and bb responsible for detection
    '''
    
    # take square root of a/b (per yolo loss equation)
    # true params
    # true_param_mask[:,:2] = true_param_mask[:,:2]**0.5
    # pred params
    # pred_param_mask[:,:2] = pred_param_mask[:,:2]**0.5
    
    # take SSE    
    loss = np.sum((true_param_mask**0.5 - pred_param_mask**0.5)**2)
    
    return lambda_coord*loss

def obj_loss(pred_response_mask):
    '''
    pred_mask: [number_objects,]
        masked for object and bb responsible for detection
        true_mask = 1
        
    loss = SSE:
        sum((true-pred)**2), if object in cell and bb responsible for detection
        true = 1
    '''
    
    # SSE
    loss = np.sum((1 - pred_response_mask)**2)

    return loss

def no_obj_loss(pred_response_mask_noobj, lambda_noobj = 0.5):
    '''
    pred_mask: [BS, S, S, B]
        masked for NO object and bb NOT responsible for detection (those response just set to 0)
        true_mask = 0
        
    loss = lambda_noobj*SSE:
        lambda_noobj*sum((true-pred)**2), if NO object in cell and bb NOT responsible for detection
        true = 0, simplifies to lambda_noobj*sum(pred**2)
    '''
    
    # take SSE
    loss = np.sum(pred_response_mask_noobj**2)
    
    return lambda_noobj*loss

def class_loss(true_class_mask, pred_class_mask):
    '''
    true/pred_mask: [number_objects, C]
    
    loss = SSE:
        sum((true-pred)**2), if object in cell
    '''

    # take SSE    
    loss = np.sum((true_class_mask - pred_class_mask)**2)
    
    return loss

def yolo_loss(y_true, y_pred):
    '''
    y_true/pred: [BS, S, S, C + B*(P+1)],
    transformed into:
        classes: [BS, S, S, C]
        params: [BS, S, S, B, P]
        response: [BS, S, S, B]

    loss = sum of individual losses
    '''
    
    # extract parameters from output
    true_class, true_params, true_response = process(y_true)
    pred_class, pred_params, pred_response = process(y_pred)
        
    # mask if object in cell (when B=0 in true response): [BS, S, S]
    # True if object in cell, False if no object
    obj_mask = true_response[...,0] == 1
    
    # mask which bounding box is responsible for prediction based on IoU score
    
    # grid index of object: [number_object, 3]
    # 3 columns: [batch number, grid x, grid y]
    loc_idx = np.argwhere(obj_mask==True)
        
    # index of bounding box responsible for prediction (max IoU)
    # only need last two columns of loc_idx (grid x/y) for converting to global params
    bb_idx = iou(true_params[obj_mask], pred_params[obj_mask], loc_idx[:, -2:])
            
    # concat bb_idx with loc_idx
    loc_and_bb_idx = np.column_stack([loc_idx, bb_idx])
    
    # get params of responsible bounding box for center/params loss
    # tuple and transpose convert idx to sequence of arrays describing: [batch number, grid x, grid y, bb_number]
    pred_params_masked = pred_params[tuple(loc_and_bb_idx.T)]
    
    # ground truth params (first bounding box is ground truth)
    true_params_masked = true_params[obj_mask][:,0,:]
    
    # mask response for object and bounding box responsible for prediction
    pred_response_masked = pred_response[tuple(loc_and_bb_idx.T)]
    
    # mask response for NO object and bounding box NOT responsible for prediction
    # done by setting object/bounding box responses = 0
    # pred_response is now properly masked
    pred_response[tuple(loc_and_bb_idx.T)] = 0
    
    # calc losses
    loss = [
        # center loss (mask for object, bb, x/y (first two coloumns))
        center_loss(true_params_masked[:,:2], pred_params_masked[:,:2]),
        
        # params loss (mask for object, bb, a/b/theta (remaining three coloumns))
        params_loss(true_params_masked[:,2:], pred_params_masked[:,2:]),
        
        # object loss (mask for object, bb (true = 1))
        obj_loss(pred_response_masked),
        
        # no object loss (mask for NO object, bb NOT responsible (true = 0))
        no_obj_loss(pred_response),
        
        # class loss (mask for object)
        class_loss(true_class[obj_mask], pred_class[obj_mask])
        ]
    
    return loss
    
    # loss = np.sum(loss)
    
    # # batch size
    # batch_size = y_true.shape[0]
        
    # # average loss over batch size
    # loss /= batch_size
        
    # return loss
    
#%% show_loss_results fuction
def show_loss_results(image, label, pred, thresh = DETECTION_THRESH):
    
    loss = yolo_loss(label, pred)
    
    loss = [l.round(3) for l in loss]
    
    # squeeze arrays after getting loss (ie remove batch size)
    image = image.squeeze()
    label = label.squeeze()
    pred = pred.squeeze()
    
    S = label.shape[1]
    
    # create copy of image for drawing ellipse
    im = np.copy(image).astype(image.dtype)
        
    # get image shape to revert label outputs
    image_h, image_w = im.shape[:2]
    
    # loop over all grid locations and draw ellipse/label - FOR LABEL
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
 
                x = x.astype(int)
                y = y.astype(int)
                
                # plot center/label
                output = LABEL_DICT[np.argmax(grid_outputs[:C])]
                plt.scatter(x, y, label = f'true: {output}', color = 'r', alpha = 0.3)

                # extract bb params to print in title
                p = [i.round(3) for i in [a,b,theta]]
                
    # loop over all grid locations and draw ellipse/label - FOR PRED
    
    # flag if object is found above thresh
    pred_above_thresh = False
    
    # loop
    for loc_i in range(S):
        for loc_j in range(S):
            
            grid_outputs = pred[loc_i, loc_j, :]
            
            # reshape pred_matrix to Bx(C+P+1) (num_bb x num_class + num_params + response)
            bb_outputs = pred[loc_i, loc_j, C:].reshape(B, P+1)
            
            # extract bounding box results if response above thresh     
            if bb_outputs[:,-1].max() > thresh:
                
                # set flag to True
                pred_above_thresh = True
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
                
                # extract pred bb params to print in title
                p_pred = [i.round(3) for i in [a,b,theta]]
                
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
                        
                output = LABEL_DICT[np.argmax(grid_outputs[:C])]
                # plot center/label
                plt.scatter(x, y, label = f'pred: {output}', color = 'b', alpha = 0.3)
                
    if not pred_above_thresh:
        p_pred = 'no prediction'
    
    # show annotated image
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(f'{p}\n{p_pred}\n{loss}\n{sum(loss)}')
    plt.legend()