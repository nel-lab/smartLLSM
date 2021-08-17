#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:34:23 2021

@author: jimmytabet
"""

#%% imports
# !pip install -q tensorflow-gpu==2.0
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, h5py, platform, cv2
from sklearn.model_selection import train_test_split
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

print('python:', platform.python_version())
print('tf:', tf.__version__)
print('h5py:', h5py.version.version)

#%% load data
fil = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update_no_isolate_286_8_11.npz'

with np.load(fil, allow_pickle=True) as dat:
  X_org = dat['X']
  y_org = dat['y']

print(Counter(y_org))

#%% remove junk and TBD cells and train on all classes
remove_mask = (y_org == 'junk') | (y_org == 'other') | (y_org == 'TBD') | (y_org == 'early_prophase')
X_mask = X_org[~remove_mask]
y_mask = y_org[~remove_mask]

unique = ['metaphase', 'telophase', 'anaphase', 'prometaphase']

# choose up to 800 random cells
X_all, y_all = [], []
for i in np.unique(y_mask):
  mask = np.isin(y_mask, i)  
  X = X_mask[mask]
  
  total_count = 800
  if i in unique: total_count /= len(unique)

  rand_mask = np.random.choice(len(X), min(len(X), int(total_count)), replace=False)
  X_all.append(X[rand_mask])
  y_all.append(np.repeat(i, len(X[rand_mask])))

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)
print(X_all.shape, y_all.shape)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=None, stratify=y_all)

randperm = np.random.permutation(len(y_train))
X_train = X_train[randperm]
y_train = y_train[randperm]

randperm = np.random.permutation(len(y_test))
X_test = X_test[randperm]
y_test = y_test[randperm]

for y in [y_train, y_test]:
  mask = (y=='metaphase') | (y=='telophase') | (y=='anaphase') | (y=='prometaphase')# | (y=='prophase')
  y[mask] = 'unique'
  # y[y=='prometaphase']='prophase'

print('train:', Counter(y_train))
print('test:', Counter(y_test))

#%% load data from previous models
# data = np.load('/content/drive/MyDrive/smart_micro/smart_micro/prophase_classifier_data_5_10.npz')
# X_train, y_train = data['X_train'], data['y_train']
# X_test, y_test = data['X_test'], data['y_test']

#%% convert labels to numerical and one_hot vectors
from tensorflow.keras.utils import to_categorical
label = np.unique(y_train)
y_train_num = np.array([np.argwhere(i==label) for i in y_train]).squeeze()
y_train_one_hot = to_categorical(y_train_num, num_classes=len(label), dtype = 'int')

y_test_num = np.array([np.argwhere(i==label) for i in y_test]).squeeze()
y_test_one_hot = to_categorical(y_test_num, num_classes=len(label), dtype='int')

# get cell stage from one_hot vector:
# num_back = np.argmax(y_train_one_hot, axis=1)
# back = label[num_back]

#%% set class weights as per sklearn's SVC class_weight = 'balanced'
weights = len(y_train) / (len(np.unique(y_train)) * np.bincount(y_train_num))
class_weight = {num: weight for num,weight in enumerate(weights)}

print(Counter(y_train))
print('weights', class_weight)

#%% add extra dimension to input vectors for model input
X_train = X_train[..., np.newaxis]
# X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

#%% add image preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=True,
    featurewise_std_normalization=False, samplewise_std_normalization=True,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=360, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=[0.5,1.5], shear_range=1, zoom_range=[0.9,1.1],
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=True, vertical_flip=True, rescale=None,
    preprocessing_function=None, data_format=None, validation_split=0.0, dtype=None
)
datagen.fit(X_train)

from skimage.util import montage
og = X_train[0]
a = np.concatenate([datagen.standardize(datagen.random_transform(og))[None,:] for i in range(25)], axis=0)
plt.subplot(121); plt.imshow(og, cmap='gray'); plt.axis('off'); plt.title('og')
plt.subplot(122); plt.imshow(montage(a.squeeze(), padding_width=10), cmap='gray'); plt.axis('off'); plt.title('preprocessed')

#%% define model
def get_conv(input_shape=(286, 286, 1), filename=None):

  model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape), #(None, None, 1)),#X_train.shape[1:])),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(64, kernel_size=8, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Conv2D(len(np.unique(y_train)), kernel_size=1, activation='softmax'),
  # tf.keras.layers.Conv2D(len(np.unique(y_train)), kernel_size=1),
  # tf.keras.layers.Dropout(0.5),
  # tf.keras.layers.BatchNormalization(),
  # tf.keras.layers.GlobalMaxPooling2D(),
  # tf.keras.layers.Activation('softmax')
  ], name='__temp__'.join(label))

  if filename:
    model.load_weights(filename)
  return model

#%% add dense layer for training
model = get_conv()
model.add(tf.keras.layers.Flatten())

print(model.summary())

#%% define F1 score
import tensorflow.keras.backend as K
def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

#%% compile and train model
# use loss=sparse_categorical_crossentropy for one-hot
model.compile(loss='categorical_crossentropy',
# optimizer=RMSprop(lr=0.001),
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=['accuracy',f1_metric])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(datagen.flow(X_train, y_train_one_hot, batch_size=64),
steps_per_epoch=len(X_train) // 64, 
epochs=3,
verbose=1,
validation_data=(datagen.standardize(X_test.astype(np.float32)), y_test_one_hot),
class_weight=class_weight,
shuffle=True,
# callbacks=[early_stopping],
)

#%% evaluate model
model.evaluate(datagen.standardize(X_test.astype(np.float32)), y_test_one_hot)

#%% save model
import datetime

save_dir = '/content/drive/MyDrive/smart_micro/model_res'
model_name = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name + '_edge.h5')
model.save(model_path)
print('Saved trained model at %s ' % model_path)

#%% get heatmat model
heatmodel = get_conv(input_shape=(None, None, 1), filename='/content/drive/MyDrive/smart_micro/model_res/2021-08-04-15-57-31.199175_edge.h5')

#%% save/load models and data

# model.save('/content/drive/MyDrive/smart_micro/smart_micro/annotator_filter_0419.hdf5')
# model = keras.models.load_model('/content/drive/MyDrive/smart_micro/smart_micro/prophase_classifier_5_10.hdf5', custom_objects={'f1_metric':f1_metric})
# np.savez('/content/drive/MyDrive/smart_micro/smart_micro/prophase_classifier_data_5_10.npz', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
