#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:02:10 2021

@author: jimmytabet
"""

#%% imports
import cv2
import numpy as np
import tensorflow as tf

#%% prepare inputs
"""
The input is a (IMG_SIZE, IMG_SIZE, 1) image and the output is a (7, 7, 30) tensor.
The output is based on S x S x (B * 5 +C). 

S X S is the number of grids
B is the number of bounding boxes per grid
C is the number of predictions per grid
"""

IMG_SIZE = 800 # input image size
S = 7 # SxS: number of grids
B = 2 # number of bounding boxes per grid
C = 3 # number of classes
P = 5 # number of parameters to desribe bounding box (4 for box, 5 for ellipse)

'''
# ex
temp = np.arange(B*(P+1) + C)
output = temp[:C]
# loop over each bounding box result
for i in range(B):
    params = temp[C+i*(P+1): C+i*(P+1)+P]
    response = temp[C+i*(P+1)+P]
'''

#%% read_data function
def read_data(data_path):
    # load image and label
    with np.load(data_path) as f:
        image = f['X']
        label = f['y']
        
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

    label_matrix = np.zeros([S, S, B*(P+1) + C])
    for l in label:
        # convert label to integers
        l = l.astype(int)
        # extract ellipse parameters and label
        x = l[0]
        y = l[1]
        a = l[2]
        b = l[3]
        theta = l[4]
        target_cls = l[5]
        
        # normalize parameters
        x /= image_w
        y /= image_h
        a /= image_w
        b /= image_h
        theta /= 360
        
        loc = [S * x, S * y]
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        y = loc[1] - loc_i
        x = loc[0] - loc_j
        
        # create label_matrix for training YOLO
        # check if response has been recorded at grid location
        if label_matrix[loc_i, loc_j, C+P] == 0:
            # one-hot label
            label_matrix[loc_i, loc_j, target_cls] = 1
            # store ellipse params
            label_matrix[loc_i, loc_j, C: C+P] = [x, y, a, b, theta]
            # response
            label_matrix[loc_i, loc_j, C+P] = 1 

    return image, label_matrix

#%% test read_data
X, y = read_data('/home/nel/Desktop/YOLOv1_ellipse/data/0.npz')
plt.imshow(X)

#%%
"""## Training the model

Next, I am defining a custom generator that returns a batch of input and outputs. 
"""

from tensorflow import keras

class My_Custom_Generator(keras.utils.Sequence):
  
  def __init__(self, images, labels, batch_size):
    self.images = images
    self.labels = labels
    self.batch_size = batch_size
    
    
  def __len__(self) :
    return (np.ceil(len(self.images) / float(self.batch_size))).astype(np.int)
  
  
  def __getitem__(self, idx) :
    batch_x = self.images[idx * self.batch_size : (idx+1) * self.batch_size]
    batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

    train_image = []
    train_label = []

    for i in range(0, len(batch_x)):
      img_path = batch_x[i]
      label = batch_y[i]
      image, label_matrix = read(img_path, label)
      train_image.append(image)
      train_label.append(label_matrix)
    return np.array(train_image), np.array(train_label)

"""The code snippet below, prepares arrays with inputs and outputs. """

train_datasets = []
val_datasets = []

X_train = []
Y_train = []

X_val = []
Y_val = []

for item in train_datasets:
  item = item.replace("\n", "").split(" ")
  X_train.append(item[0])
  arr = []
  for i in range(1, len(item)):
    arr.append(item[i])
  Y_train.append(arr)

for item in val_datasets:
  item = item.replace("\n", "").split(" ")
  X_val.append(item[0])
  arr = []
  for i in range(1, len(item)):
    arr.append(item[i])
  Y_val.append(arr)

"""Next, we create instances of the generator for our training and validation sets. """

batch_size = 4
my_training_batch_generator = My_Custom_Generator(X_train, Y_train, batch_size)

my_validation_batch_generator = My_Custom_Generator(X_val, Y_val, batch_size)

x_train, y_train = my_training_batch_generator.__getitem__(0)
x_val, y_val = my_training_batch_generator.__getitem__(0)
print(x_train.shape)
print(y_train.shape)

print(x_val.shape)
print(y_val.shape)

"""### Define a custom output layer

We need to reshape the output from the model so we define a custom Keras layer for it. 
"""

from tensorflow import keras
import keras.backend as K

class Yolo_Reshape(tf.keras.layers.Layer):
  def __init__(self, target_shape):
    super(Yolo_Reshape, self).__init__()
    self.target_shape = tuple(target_shape)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
        'target_shape': self.target_shape
    })
    return config

  def call(self, input):
    # grids 7x7
    S = [self.target_shape[0], self.target_shape[1]]
    # classes
    C = 20
    # no of bounding boxes per grid
    B = 2

    idx1 = S[0] * S[1] * C
    idx2 = idx1 + S[0] * S[1] * B
    
    # class probabilities
    class_probs = K.reshape(input[:, :idx1], (K.shape(input)[0],) + tuple([S[0], S[1], C]))
    class_probs = K.softmax(class_probs)

    #confidence
    confs = K.reshape(input[:, idx1:idx2], (K.shape(input)[0],) + tuple([S[0], S[1], B]))
    confs = K.sigmoid(confs)

    # boxes
    boxes = K.reshape(input[:, idx2:], (K.shape(input)[0],) + tuple([S[0], S[1], B * 4]))
    boxes = K.sigmoid(boxes)

    outputs = K.concatenate([class_probs, confs, boxes])
    return outputs

"""### Defining the YOLO model. 

Next, we define the model as described in the original paper. 
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, LeakyReLU
from tensorflow.keras.layers import Conv2D, MaxPooling2D

lrelu = LeakyReLU(alpha=0.1)

nb_boxes=1
grid_w=7
grid_h=7
cell_w=64
cell_h=64
img_w=grid_w*cell_w
img_h=grid_h*cell_h

model = Sequential()
model.add(Conv2D(filters=64, kernel_size= (7, 7), strides=(1, 1), input_shape =(img_h, img_w, 3), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=192, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=128, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=256, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=256, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=512, kernel_size= (1, 1), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024, kernel_size= (3, 3), padding = 'same', activation=lrelu))
model.add(Conv2D(filters=1024, kernel_size= (3, 3), strides=(2, 2), padding = 'same'))

model.add(Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu))
model.add(Conv2D(filters=1024, kernel_size= (3, 3), activation=lrelu))

model.add(Flatten())
model.add(Dense(512))
model.add(Dense(1024))
model.add(Dropout(0.5))
model.add(Dense(1470, activation='sigmoid'))
model.add(Yolo_Reshape(target_shape=(7,7,30)))
model.summary()

"""### Define a custom learning rate scheduler

The paper uses different learning rates for different epochs. So we define a custom Callback function for the learning rate. 
"""

from tensorflow import keras

class CustomLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, schedule):
        super(CustomLearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
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

"""### Define the loss function

Next, we would be defining a custom loss function to be used in the model. Take a look at this blog post to understand more about the [loss function used in YOLO](https://hackernoon.com/understanding-yolo-f5a74bbc7967). 

I understood the loss function but didn't implement it on my own. I took the implementation as it is from this [Github repo](https://github.com/JY-112553/yolov1-keras-voc).
"""

def xywh2minmax(xy, wh):
    xy_min = xy - wh / 2
    xy_max = xy + wh / 2

    return xy_min, xy_max


def iou(pred_mins, pred_maxes, true_mins, true_maxes):
    intersect_mins = K.maximum(pred_mins, true_mins)
    intersect_maxes = K.minimum(pred_maxes, true_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    pred_wh = pred_maxes - pred_mins
    true_wh = true_maxes - true_mins
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores


def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0])
    conv_width_index = K.arange(0, stop=conv_dims[1])
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
    conv_width_index = K.tile(
        K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
    conv_width_index = K.flatten(K.transpose(conv_width_index))
    conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
    conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
    conv_index = K.cast(conv_index, K.dtype(feats))

    conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), K.dtype(feats))

    box_xy = (feats[..., :2] + conv_index) / conv_dims * IMG_SIZE
    box_wh = feats[..., 2:4] * IMG_SIZE

    return box_xy, box_wh


def yolo_loss(y_true, y_pred):
    label_class = y_true[..., :20]  # ? * 7 * 7 * 20
    label_box = y_true[..., 20:24]  # ? * 7 * 7 * 4
    response_mask = y_true[..., 24]  # ? * 7 * 7
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :20]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    label_xy = K.expand_dims(label_xy, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_wh = K.expand_dims(label_wh, 3)  # ? * 7 * 7 * 1 * 1 * 2
    label_xy_min, label_xy_max = xywh2minmax(label_xy, label_wh)  # ? * 7 * 7 * 1 * 1 * 2, ? * 7 * 7 * 1 * 1 * 2

    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2
    predict_xy = K.expand_dims(predict_xy, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_wh = K.expand_dims(predict_wh, 4)  # ? * 7 * 7 * 2 * 1 * 2
    predict_xy_min, predict_xy_max = xywh2minmax(predict_xy, predict_wh)  # ? * 7 * 7 * 2 * 1 * 2, ? * 7 * 7 * 2 * 1 * 2

    iou_scores = iou(predict_xy_min, predict_xy_max, label_xy_min, label_xy_max)  # ? * 7 * 7 * 2 * 1
    best_ious = K.max(iou_scores, axis=4)  # ? * 7 * 7 * 2
    best_box = K.max(best_ious, axis=3, keepdims=True)  # ? * 7 * 7 * 1

    box_mask = K.cast(best_ious >= best_box, K.dtype(best_ious))  # ? * 7 * 7 * 2

    no_object_loss = 0.5 * (1 - box_mask * response_mask) * K.square(0 - predict_trust)
    object_loss = box_mask * response_mask * K.square(1 - predict_trust)
    confidence_loss = no_object_loss + object_loss
    confidence_loss = K.sum(confidence_loss)

    class_loss = response_mask * K.square(label_class - predict_class)
    class_loss = K.sum(class_loss)

    _label_box = K.reshape(label_box, [-1, 7, 7, 1, 4])
    _predict_box = K.reshape(predict_box, [-1, 7, 7, 2, 4])

    label_xy, label_wh = yolo_head(_label_box)  # ? * 7 * 7 * 1 * 2, ? * 7 * 7 * 1 * 2
    predict_xy, predict_wh = yolo_head(_predict_box)  # ? * 7 * 7 * 2 * 2, ? * 7 * 7 * 2 * 2

    box_mask = K.expand_dims(box_mask)
    response_mask = K.expand_dims(response_mask)

    box_loss = 5 * box_mask * response_mask * K.square((label_xy - predict_xy) / IMG_SIZE)
    box_loss += 5 * box_mask * response_mask * K.square((K.sqrt(label_wh) - K.sqrt(predict_wh)) / IMG_SIZE)
    box_loss = K.sum(box_loss)

    loss = confidence_loss + class_loss + box_loss

    return loss

"""### Add a callback for saving the weights

Next, I define a callback to keep saving the best weights. 
"""

# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

"""### Compile the model

Finally, I compile the model using the custom loss function that was defined above. 
"""

from tensorflow import keras

model.compile(loss=yolo_loss ,optimizer='adam')

"""### Train the model

Now that we have everything setup, we will call `model.fit` to train the model for 135 epochs. 
"""

model.fit(x=my_training_batch_generator,
          steps_per_epoch = int(len(X_train) // batch_size),
          epochs = 135,
          verbose = 1,
          workers= 4,
          validation_data = my_validation_batch_generator,
          validation_steps = int(len(X_val) // batch_size),
           callbacks=[
              CustomLearningRateScheduler(lr_schedule),
              mcp_save
          ])

"""## Conclusion

It was a good exercise to implement YOLO V1 from scratch and understand various nuances of writing a model from scratch. This implementation won't achieve the same accuracy as what was described in the paper since we have skipped the pretraining step. 
"""