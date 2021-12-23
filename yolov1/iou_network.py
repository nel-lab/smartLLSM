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

# grid size
grid = 800
# number of images to generate
num_images = 30000
# offset between true and pred
offset = 0.3

# visualize dataset
viz = False

# init arrays
X = np.zeros([num_images, 10])
# X_small = np.zeros([num_images, 12])
y = np.zeros(num_images)
# y_small = np.zeros(num_images)

# loop over num_images
for i in range(num_images):

    # true ellipse parameters
    cx = np.random.randint(100, grid-100)
    cy = np.random.randint(100, grid-100)
    a = np.random.randint(50,100)
    b = np.random.randint(50,100)
    theta = np.random.randint(45)
    
    # create true image
    true_ellipse = cv2.ellipse(np.zeros([grid,grid]), (cx,cy), (a,b) ,theta, 0, 360, 1, -1)

    cx_pred = cx
    cy_pred = cy
    a_pred = a
    b_pred = b
    theta_pred = theta
    # pred ellipse parameters
    thresh_prob = 0.5
    if np.random.rand()<thresh_prob:
        cx_pred = int(cx * (1 + offset*(2*np.random.rand()-1)))
    if np.random.rand()<thresh_prob:
        cy_pred = int(cy * (1 + offset*(2*np.random.rand()-1)))
    if np.random.rand()<thresh_prob:
        a_pred = int(a * (1 + offset*(2*np.random.rand()-1)))
    if np.random.rand()<thresh_prob:
        b_pred = int(b * (1 + offset*(2*np.random.rand()-1)))
    if np.random.rand()<thresh_prob:
        theta_pred = int(theta * (1 + offset*(2*np.random.rand()-1)))
    
    # create pred image
    pred_ellipse = cv2.ellipse(np.zeros([grid,grid]), (cx_pred,cy_pred), (a_pred,b_pred) ,theta, 0, 360, 1, -1)
    
    # draw combined
    combined = pred_ellipse + true_ellipse
    # calc intersection
    intersection = np.sum(combined == 2)
    # calc union
    union = np.sum(combined > 0)

    # show image with labels
    if viz and i<10:
        plt.cla()
        plt.imshow(combined, cmap='gray')
        plt.title(f'IOU: {intersection/union}')
        plt.axis('off')
        plt.pause(.5)
    
    # store ellipse parameters
    # X[i] = [cx, cy, a, b, np.cos(np.radians(theta))**2, np.sin(np.radians(theta))**2, cx_pred, cy_pred, a_pred, b_pred, np.cos(np.radians(theta_pred))**2, np.sin(np.radians(theta_pred))**2]   
    X[i] = [cx, cy, a, b, theta, cx_pred, cy_pred, a_pred, b_pred, theta_pred]   
    # X_small[i] = [cx, cy, a, b, np.cos(np.radians(theta))**2, np.sin(np.radians(theta))**2, cx_pred, cy_pred, a_pred, b_pred, np.cos(np.radians(theta_pred))**2, np.sin(np.radians(theta_pred))**2]
    # store IoU
    y[i] = intersection/union
    # y_small[i] = intersection/union
#%%
X = X[(y>0.001) & (y<0.999)]
y = y[(y>0.001) & (y<0.999)]
#%%
y_all = np.stack([y]).flatten()
plt.hist(y_all, bins=1000)

#%%
import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(X.shape[-1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
    tf.keras.layers.Dense(1, activation='linear')
    ])


model.compile(loss = tf.keras.losses.MeanAbsoluteError(), optimizer = tf.keras.optimizers.Adam(learning_rate=0.001))

#%%
def scheduler(epoch, lr):
  if epoch < 50:
    return float(lr)
  else:
    return float(lr * tf.math.exp(-0.01))

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

import datetime
date_id = datetime.datetime.now().strftime("%m%d_%H%M")
log_dir = f'/home/nel/Desktop/tensorboard/{date_id}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(x=X, y=y, epochs = 400,
          batch_size=64,
          verbose = 1,
          validation_split=0.3,
          callbacks=[
              lr_callback, tensorboard_callback
          ])

#%%
model.save('/home/nel/Desktop/iou_model_2.h5')
#%%
pp  = model.evaluate(X,y)
#%%
plt.scatter(model.predict(X),y,marker='.')