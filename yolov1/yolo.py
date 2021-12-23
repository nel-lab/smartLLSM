#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:02:10 2021

@author: jimmytabet
"""

#%% imports
import os, datetime
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
tf.keras.backend.set_floatx(
    'float32'
)
import tensorflow.keras.backend as K

import sys
sys.path.append('/home/nel/Software/smart-micro/yolov1')
import utils_K

#%% prepare inputs
'''
input: (IMG_SIZE, IMG_SIZE, 1) image

output: S x S x (C + B * (P+1))

where
S X S is the number of grids
B is the number of bounding boxes per grid
C is the number of predictions per grid
P is number of parameters to describe bounding box (4 for box, 5 for ellipse)
    +1 for response (if object is in grid)
'''

IMG_SIZE = 800
S = 7
B = 1
C = 3
P = 5

# batch size
BATCH_SIZE = 20

#%% test read_data and show_results
plt.figure()
for i in range(5):
    # read in data
    X, y = utils_K.read_data(f'/home/nel/Desktop/YOLOv1_ellipse/train_data/{i}.npz')
    # clear axis to plot in loop
    plt.cla()
    # show results
    utils_K.show_results(X,y)
    # pause to plot in loop
    plt.pause(.5)
    
plt.close('all')

#%% Custom_Generator for train/val/test data batches
class Custom_Generator(tf.keras.utils.Sequence):
        
    # init paths and batch size
    def __init__(self, paths, batch_size):
        # shuffle paths
        # np.random.shuffle(paths)
        self.paths = paths
        self.batch_size = batch_size
      
    # define length
    def __len__(self):
        return (np.ceil(len(self.paths) / float(self.batch_size))).astype(int)
    
    # get item
    def __getitem__(self, idx):
        # make batch of paths
        batch_paths = self.paths[idx * self.batch_size : (idx+1) * self.batch_size]
        
        X = []
        y = []
        
        for path in batch_paths:
          image, label_matrix = utils_K.read_data(path)
          X.append(image)
          y.append(label_matrix)
        
        return np.array(X), np.array(y)

#%% get train/val/test paths and batch generator
train_data = '/home/nel/Desktop/YOLOv1_ellipse/train_data'
val_data = '/home/nel/Desktop/YOLOv1_ellipse/val_data'
test_data = '/home/nel/Desktop/YOLOv1_ellipse/test_data'

train_paths = [os.path.join(train_data, i) for i in sorted(os.listdir(train_data))]
val_paths = [os.path.join(val_data, i) for i in sorted(os.listdir(val_data))]
test_paths = [os.path.join(test_data, i) for i in sorted(os.listdir(test_data))]

# run generator
train_batch_generator = Custom_Generator(train_paths, BATCH_SIZE)
val_batch_generator = Custom_Generator(val_paths, BATCH_SIZE)
test_batch_generator = Custom_Generator(test_paths, BATCH_SIZE)

#%% test generator
X_train, y_train = train_batch_generator.__getitem__(0)

print(X_train.shape)
print(y_train.shape)

# utils_K.show_results(X_train[10], y_train[10])

#%% define yolo reshape layer
class Yolo_Reshape(tf.keras.layers.Layer):
    
    # init with target shape
    def __init__(self, target_shape, **kwargs):
        super(Yolo_Reshape, self).__init__()
        self.target_shape = tuple(target_shape)
    
    # update config with target shape?
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'target_shape': self.target_shape
            })
        
        return config
    
    def call(self, input):
        '''
        input: [BS, S*S*params]

        class_probs: [BS, S, S, C]
        params: [BS, S, S, B*(P+1)]

        outputs: [BS, S, S, C+B*(P+1)]
        
        TODO: update for multiple bounding boxes
        '''
        
        # reshape into [BS, S, S, C+B*(P+1)]
        input = K.reshape(input, (-1, S, S, C+B*(P+1)))
        
        # class probabilities - softmax
        class_probs = input[..., :C]
        class_probs = K.softmax(class_probs)
        
        # params - sigmoid
        params = input[..., C:]
        params = K.sigmoid(params)
        
        outputs = K.concatenate([class_probs,params])
        
        return outputs

#%% define yolo model
lrelu = tf.keras.layers.LeakyReLU(alpha=0.1)

model = tf.keras.models.Sequential([
tf.keras.layers.LayerNormalization(axis=(1,2), trainable=False, scale=False, center=False,  input_shape = (IMG_SIZE, IMG_SIZE, 1)),   
tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
tf.keras.layers.MaxPooling2D(2, 2),

tf.keras.layers.Dropout(0.25),
tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Dropout(0.25),
tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Dropout(0.25),
tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Dropout(0.25),
tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
tf.keras.layers.Conv2D(32, (1,1), activation='relu', padding='same'),
tf.keras.layers.MaxPooling2D(2,2),

tf.keras.layers.Conv2D(64, kernel_size=6, activation='relu'),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(64),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(S*S * (C + B * (P+1)), activation='linear'),#'sigmoid')
Yolo_Reshape(target_shape=(S, S, C + B * (P+1)))])

# # OLD MODEL
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
# from tensorflow.keras.layers import Conv2D, MaxPooling2D

# div_factor = 16

# model = Sequential()
# model.add(Conv2D(filters=64//div_factor, kernel_size= (7, 7), strides=(1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1), padding = 'same', activation=lrelu))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

# model.add(Conv2D(filters=192//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

# model.add(Conv2D(filters=128//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

# model.add(Conv2D(filters=512//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), strides=(2, 2), padding = 'same'))

# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), activation=lrelu))

# model.add(Flatten())
# model.add(Dense(512//div_factor))
# model.add(Dense(1024//div_factor))
# model.add(Dropout(0.5))

# model.add(Dense(S*S * (C + B * (P+1)), activation=lrelu))#'sigmoid'))
# model.add(Yolo_Reshape(target_shape=(S, S, C + B * (P+1))))

# print model summary
model.summary()

#%% learning rate scheduler
# # yolo LR scheduler
# class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
#     def __init__(self, schedule):
#         super(CustomLearningRateScheduler, self).__init__()
#         self.schedule = schedule

#     def on_epoch_begin(self, epoch, logs=None):
#         if not hasattr(self.model.optimizer, "lr"):
#             raise ValueError('Optimizer must have a "lr" attribute.')
#         # Get the current learning rate from model's optimizer.
#         lr = float(K.get_value(self.model.optimizer.learning_rate))
#         # Call schedule function to get the scheduled learning rate.
#         scheduled_lr = self.schedule(epoch, lr)
#         # Set the value back to the optimizer before this epoch starts
#         K.set_value(self.model.optimizer.lr, scheduled_lr)
#         print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))

# LR_SCHEDULE = [
#     # (epoch to start, learning rate) tuples
#     (0, 0.01),
#     (75, 0.001),
#     (105, 0.0001),
# ]

# def lr_schedule(epoch, lr):
#     """Helper function to retrieve the scheduled learning rate based on epoch."""
#     if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
#         return lr
#     for i in range(len(LR_SCHEDULE)):
#         if epoch == LR_SCHEDULE[i][0]:
#             return LR_SCHEDULE[i][1]
#     return lr

# simple decay learning rate scheduler
def scheduler(epoch, lr):
  if epoch < 20:
    return float(lr)
  else:
    return float(lr * tf.math.exp(-0.1))

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

#%% save best weights
# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint

# get date/time to match model with tensorboard
date_id = datetime.datetime.now().strftime("%m%d_%H%M")

mcp_save = ModelCheckpoint(f'/home/nel/Desktop/YOLOv1_ellipse/{date_id}.hdf5',
                           save_best_only=True)

#%% compile and train
model.compile(loss = utils_K.yolo_loss,
              # optimizer = 'adam',
              optimizer = tf.keras.optimizers.Adam(lr=0.001),
              run_eagerly = False,
              # metrics = [center_loss, params_loss, obj_loss, no_obj_loss, class_loss]
              )

log_dir = f'/home/nel/Desktop/tensorboard/{date_id}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(
          x=train_batch_generator,
          # steps_per_epoch = int(len(train_paths) // BATCH_SIZE),
          epochs = 150,
          verbose = 1,
          validation_data = val_batch_generator,
          # validation_steps = int(len(val_paths) // BATCH_SIZE),
          callbacks=[
              tensorboard_callback,
              # CustomLearningRateScheduler(lr_schedule),
              # lr_callback,
              mcp_save
          ])

#%% load best model
# model = tf.keras.models.load_model(f'/home/nel/Desktop/YOLOv1_ellipse/{date_id}.hdf5', custom_objects = {'Yolo_Reshape': Yolo_Reshape, 'yolo_loss': utils_K.yolo_loss})
model = tf.keras.models.load_model('/home/nel/Desktop/YOLOv1_ellipse/best_model/1116_1644.hdf5', custom_objects = {'Yolo_Reshape': Yolo_Reshape, 'yolo_loss': utils_K.yolo_loss})

#%% test model - test set
X_test, y_test = test_batch_generator.__getitem__(0)
y_pred = model.predict(X_test)

yp_classes, _, yp_response = utils_K.process(y_pred)
yt_classes, _, yt_response = utils_K.process(y_test)

for i in range(BATCH_SIZE):
    plt.cla()
    utils_K.show_loss_results(X_test[i:i+1], y_test[i:i+1], y_pred[i:i+1], 0.1)
    plt.ginput(-1)
    
plt.close('all')

#%% NON MAX SUPRESSION
true = y_test[7]
pred = y_pred[7]

yp_classes, yp_params, yp_response = utils_K.process(pred, numpy=True)
yt_classes, yt_params, yt_response = utils_K.process(true, numpy=True)


#%%
#%% get all val data
X_val = []
# y_val = []

for batch in range(7):
    print('*')
    X_temp, y_temp = X_test, y_test = test_batch_generator.__getitem__(batch)
    X_val.append(X_temp)
    # y_val.append(y_temp)

X_val = np.concatenate(X_val).astype(np.float32())
# y_val = np.concatenate(y_val)

#%%
import time

s = time.time()

res = model.predict(X_val)

print((time.time() - s)/X_val.shape[0])

#%% test model - train set
X_test, y_test = train_batch_generator.__getitem__(0)
y_pred = model.predict(X_test)

yp_classes, _, yp_response = utils_K.process(y_pred)
yt_classes, _, yt_response = utils_K.process(y_test)

for i in range(BATCH_SIZE):
    plt.cla()
    utils_K.show_loss_results(X_test[i:i+1], y_test[i:i+1], y_pred[i:i+1], 0.25)
    plt.ginput(-1)
    
plt.close('all')

#%% test model - multiple objects
X_mult = []
y_mult = []
for i in range(10):
    X_temp, y_temp = utils_K.read_data(f'/home/nel/Desktop/multi_object/{i}.npz')
    
    X_mult.append(X_temp)
    y_mult.append(y_temp)
    
X_mult = np.stack(X_mult)
y_mult = np.stack(y_mult)

y_pred_mult = model.predict(X_mult)

for i in range(10):
    plt.cla()
    utils_K.show_loss_results(X_mult[i:i+1], y_mult[i:i+1], y_pred_mult[i:i+1], 0.2)
    plt.ginput(-1)

plt.close('all')

#%% view filters - OLD
# # load the model
# # retrieve weights from the second hidden layer
# filters, biases = model.layers[0].get_weights()
# # normalize filter values to 0-1 so we can visualize them
# f_min, f_max = filters.min(), filters.max()
# filters = (filters - f_min) / (f_max - f_min)
# # plot first few filters
# n_filters, ix = 4, 1
# for i in range(n_filters):
# 	# get the filter
# 	f = filters[:, :, :, i]
# 	# plot each channel separately
# 	for j in range(1):
# 		# specify subplot and turn of axis
# 		ax = plt.subplot(n_filters, 3, ix)
# 		ax.set_xticks([])
# 		ax.set_yticks([])
# 		# plot filter channel in grayscale
# 		plt.imshow(f[:, :, j], cmap='gray')
# 		ix += 1