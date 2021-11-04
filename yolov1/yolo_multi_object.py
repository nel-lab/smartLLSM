#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 12:02:10 2021

@author: jimmytabet
"""

#%% imports
import os, cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K

#%% prepare inputs
'''
input: (IMG_SIZE, IMG_SIZE, 1) image

output: S x S x (B * (C + P+1))

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

# other inputs
# label dictionary
LABEL_DICT = {0: 'small',
              1: 'medium',
              2: 'large'}

# threshold for bounding box detection response
DETECTION_THRESH = 0

# threshold for class label
LABEL_THRESH = 0

# batch size
BATCH_SIZE = 5

#%% read_data function
def read_data(data_path):
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
    label_matrix = np.zeros([S, S, B*(C + P+1)])
   
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
        theta /= 360
        
        # convert x and y to grid location
        loc = [S * x, S * y]
        # grid location of label_matrix
        loc_i = int(loc[1])
        loc_j = int(loc[0])
        # relative unit location inside grid
        y = loc[1] - loc_i
        x = loc[0] - loc_j
        
        # check if response has been recorded at grid location for number of bounding boxes
        for i in range(B):
            # check for response, if empty store info
            if label_matrix[loc_i, loc_j,
                            i*(C+P+1) + (C+P)] == 0:
                
                # one-hot label
                label_matrix[loc_i, loc_j,
                             i*(C+P+1) + target_cls] = 1
                
                # store ellipse params
                label_matrix[loc_i, loc_j,
                             i*(C+P+1) + C: i*(C+P+1) + C+P] = [x, y, a, b, theta]
                
                # response
                label_matrix[loc_i, loc_j,
                             i*(C+P+1) + (C+P)] = 1
                
                break

    # if there are more objects in grid than (B) bounding boxes, print message)
    if label_matrix.reshape(S,S,B,-1)[:,:,:,-1].sum() != label.shape[0]:
        print(f'More than {B} objects in grid. Increase B parameter.')

    return image, label_matrix

#%% show_results fuction
def show_results(image, label):
    
    # create copy of image for drawing ellipse
    im = np.copy(image).astype(image.dtype)
        
    # get image shape to revert label outputs
    image_h, image_w = im.shape[:2]
    
    # loop over all grid locations and draw ellipse/label
    for loc_i in range(S):
        for loc_j in range(S):
            # reshape label_matrix to Bx(C+P+1) (num_bb x num_class + num_params + response)
            bb_outputs = label[loc_i, loc_j].reshape(B, C+P+1)
            
            # extract bounding box results that have a response above DETECTION_THRESH
            bb_outputs_w_response = bb_outputs[bb_outputs[:,-1] > DETECTION_THRESH]
            
            # loop throuh valid bounding boxes and draw ellipse/label 
            for bb in bb_outputs_w_response:
                # extract ellipse parameters
                x,y,a,b,theta = bb[C:-1]

                # convert x,y,a,b,theta back into image coordinates                
                x += loc_j
                x /= S                
                x *= image_w                
                
                y += loc_i
                y /= S
                y *= image_h                
                
                a *= image_w
                b *= image_h
                theta *= 360    
                
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
                output = LABEL_DICT[np.argmax(bb[:C])]
                plt.text(x,y,output, ha='center', size='x-small')
    
    # show annotated image
    plt.imshow(im, cmap='gray')
    plt.axis('off')

#%% test read_data and show_results
plt.close('all')

plt.figure()
for i in range(5):
    # read in data
    X, y = read_data(f'/home/nel/Desktop/YOLOv1_ellipse/train_data/{i}.npz')
    # clear axis to plot in loop
    plt.cla()
    # show results
    show_results(X,y)
    # pause to plot in loop
    plt.pause(.5)

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

show_results(X_train[0], y_train[0])

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
        confs: [BS, S, S, B]
        boxes: [BS, S, S, B*P]
        outputs: [BS, S, S, C+B+B*P]
        
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

model.add(Dense(S*S * (B*(C + P+1)), activation='sigmoid'))
model.add(Yolo_Reshape(target_shape=(S, S, B*(C + P+1))))

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

#%% loss function
'''
reference links
https://hackernoon.com/understanding-yolo-f5a74bbc7967
https://github.com/JY-112553/yolov1-keras-voc
'''

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

'''
input: BS, S, S, (C+B*(P+1))

output: BS * S * S * B * 1
'''

def iou(pred, true):
    
    BS = K.shape(pred)[0]
    
    iou_scores = np.zeros([BS, S, S, B, 1])
    
    for i in range(BS):
        for j in range(S):
            for k in range(S):
                    pred_x, pred_y, pred_a, pred_b, pred_theta = pred[i,j,k,C:-1]
                    ellipse_pred = cv2.ellipse(np.zeros((800,800)), (pred_x,pred_y), (pred_a,pred_b), pred_theta, 0, 360, 1, -1)
                    
                    true_x, true_y, true_a, true_b, true_theta = true[i,j,k,C:-1]
                    ellipse_true = cv2.ellipse(np.zeros((800,800)), (true_x,true_y), (true_a,true_b), true_theta, 0, 360, 1, -1)
                
                    combined = ellipse_pred + ellipse_true
                    intersection = np.sum(combined == 2)
                    union = np.sum(combined > 0)
        
                    plt.imshow(combined)
                    plt.title(intersection/union)
        
                    iou_scores[i,j,k,0,0] = intersection/union        
    
    return tf.convert_to_tensor(iou_scores)



def yolo_head(feats):
    # Dynamic implementation of conv dims for fully convolutional model.
    conv_dims = K.shape(feats)[1:3]  # assuming channels last # [7,7]
    # In YOLO the height index is the inner most iteration.
    conv_height_index = K.arange(0, stop=conv_dims[0]) # np.arange(7)
    conv_width_index = K.arange(0, stop=conv_dims[1]) # np.arange(7)
    conv_height_index = K.tile(conv_height_index, [conv_dims[1]]) # 0-6,0-6,0-6...x7

    # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
    # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0) # 0x7, 1x7, 2x7, ...6x7
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

    label_class = y_true[..., :C]  # ? * 7 * 7 * 20
    label_box = y_true[..., C:C+P]  # ? * 7 * 7 * 4
    response_mask = y_true[..., C+P+1]  # ? * 7 * 7
    response_mask = K.expand_dims(response_mask)  # ? * 7 * 7 * 1

    predict_class = y_pred[..., :C]  # ? * 7 * 7 * 20
    predict_trust = y_pred[..., 20:22]  # ? * 7 * 7 * 2
    predict_box = y_pred[..., 22:]  # ? * 7 * 7 * 8

    _label_box = K.reshape(label_box, [-1, S, S, 1, P])
    _predict_box = K.reshape(predict_box, [-1, S, S, B, P])

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

#%% save best weights
# defining a function to save the weights of best model
from tensorflow.keras.callbacks import ModelCheckpoint

mcp_save = ModelCheckpoint('weight.hdf5', save_best_only=True, monitor='val_loss', mode='min')

#%% compile and train
model.compile(loss=yolo_loss, optimizer='adam')

model.fit(x=train_batch_generator,
          steps_per_epoch = int(len(train_paths) // BATCH_SIZE),
          epochs = 135,
          verbose = 1,
          workers= 4,
          validation_data = val_batch_generator,
          validation_steps = int(len(val_paths) // BATCH_SIZE),
          callbacks=[
              CustomLearningRateScheduler(lr_schedule),
              mcp_save
          ])