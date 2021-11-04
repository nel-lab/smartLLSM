#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:02:10 2021

@author: jimmytabet
"""

#%% imports
import os, datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from utils_K import read_data, show_results, yolo_loss, process

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
B = 2
C = 3
P = 5

# batch size
BATCH_SIZE = 5

#%% test read_data and show_results
# plt.close('all')

# plt.figure()
# for i in range(5):
#     # read in data
#     X, y = read_data(f'/home/nel/Desktop/YOLOv1_ellipse/train_data/{i}.npz')
#     # clear axis to plot in loop
#     plt.cla()
#     # show results
#     show_results(X,y)
#     # pause to plot in loop
#     plt.pause(.5)

#%% Custom_Generator for train/val/test data batches
class Custom_Generator(tf.keras.utils.Sequence):
        
    # init paths and batch size
    def __init__(self, paths, batch_size):
        # shuffle paths
        np.random.shuffle(paths)
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
          image, label_matrix = read_data(path)
          X.append(image)
          y.append(label_matrix)
        
        return np.array(X), np.array(y)

#%% get train/val/test paths and batch generator
train_data = '/home/nel/Desktop/YOLOv1_ellipse/train_data'
val_data = '/home/nel/Desktop/YOLOv1_ellipse/val_data'
test_data = '/home/nel/Desktop/YOLOv1_ellipse/test_data'

train_paths = [os.path.join(train_data, i) for i in os.listdir(train_data)]
val_paths = [os.path.join(val_data, i) for i in os.listdir(val_data)]
test_paths = [os.path.join(test_data, i) for i in os.listdir(test_data)]

# run generator
train_batch_generator = Custom_Generator(train_paths, BATCH_SIZE)
val_batch_generator = Custom_Generator(val_paths, BATCH_SIZE)
test_batch_generator = Custom_Generator(test_paths, BATCH_SIZE)

#%% test generator
X_train, y_train = train_batch_generator.__getitem__(0)

print(X_train.shape)
print(y_train.shape)

# show_results(X_train[0], y_train[0])

#%% define yolo reshape layer
'''
not sure if this is necessary, will try with a simple reshape first
'''

class Yolo_Reshape(tf.keras.layers.Layer):
    
    # init with target shape
    def __init__(self, target_shape):
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
        boxes: [BS, S, S, B*P]
        confs: [BS, S, S, B]
        outputs: [BS, S, S, C+B*P+B]
        
        TODO: update for multiple bounding boxes
        '''
        # bb_input = K.reshape(input, (K.shape(input)[0],) + tuple.reshape(B, C+P+1)
        
        # for bb in bb_input:
            # x,y,a,b,theta = bb[C:-1]
        
        class_idx = S * S * C
        bb_idx = class_idx + S * S * B * P
        
        # class probabilities
        class_probs = K.reshape(input[:, :class_idx], (K.shape(input)[0],) + tuple([S, S, C]))
        class_probs = K.softmax(class_probs)
      
        # boxes
        boxes = K.reshape(input[:, class_idx:bb_idx], (K.shape(input)[0],) + tuple([S, S, B * P]))
        boxes = K.sigmoid(boxes)
      
        # response
        confs = K.reshape(input[:, bb_idx:], (K.shape(input)[0],) + tuple([S, S, B]))
        confs = K.sigmoid(confs)
        
        outputs = K.concatenate([class_probs, boxes, confs])
        
        return outputs

#%% define yolo model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D

div_factor = 8
lrelu = LeakyReLU(alpha=0.1)

model = Sequential()
model.add(Conv2D(filters=64//div_factor*2, kernel_size= (7, 7), strides=(1, 1), input_shape = (IMG_SIZE, IMG_SIZE, 1), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=192//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=128//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=256//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=256//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=512//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=512//div_factor, kernel_size= (1, 1), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
# model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), activation=lrelu))
model.add(Conv2D(filters=1024//div_factor, kernel_size= (3, 3), activation=lrelu))

model.add(Flatten())
model.add(Dense(512//div_factor))
model.add(Dense(1024//div_factor))
model.add(Dropout(0.5))

model.add(Dense(S*S * (C + B * (P+1)), activation='sigmoid'))
model.add(Yolo_Reshape(target_shape=(S, S, C + B * (P+1))))

# print model summary
model.summary()

#%% learning rate scheduler
class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))


LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (0, 0.01),
    (75, 0.001),
    (105, 0.0001),
]


def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr

#%% save best weights
# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

#%% compile and train
model.compile(loss=yolo_loss, optimizer='adam', run_eagerly = True)

log_dir = f'/home/nel/Desktop/tensorboard/{datetime.datetime.now().strftime("%m%d_%H%M")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

model.fit(x=train_batch_generator,
          steps_per_epoch = int(len(train_paths) // BATCH_SIZE),
          epochs = 150,
          verbose = 1,
          validation_data = val_batch_generator,
          validation_steps = int(len(val_paths) // BATCH_SIZE),
          callbacks=[
              tensorboard_callback,
              CustomLearningRateScheduler(lr_schedule),
              mcp_save
          ])

#%% test model
X_test, y_test = test_batch_generator.__getitem__(0)
y_pred = model.predict(X_test)

show_results(X_test[0], y_pred[0], .73)
