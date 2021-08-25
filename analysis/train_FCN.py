#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 11:34:23 2021

@author: jimmytabet
"""

#%% imports
# !pip install -q tensorflow-gpu==2.0
import os, h5py, platform, datetime
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

print('python:', platform.python_version())
print('tf:', tf.__version__)
print('h5py:', h5py.version.version)

#%% load data
fil = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_fov200_0824.npz'

with np.load(fil, allow_pickle=True) as dat:
    X_org = dat['X']
    y_org = dat['y']

print(Counter(y_org))

#%% clean data
remove_mask = (y_org == 'junk') | (y_org == 'other') | (y_org == 'TBD') | (y_org == 'early_prophase')
# include early prophase cells with prophase
# y_org[y_org == 'early_prophase'] = 'prophase'
X_mask = X_org[~remove_mask]
y_mask = y_org[~remove_mask]

# unique = ['metaphase', 'telophase', 'anaphase', 'prometaphase']
unique = []

# choose up to 800 random cells
X_all, y_all = [], []
for i in np.unique(y_mask):
    mask = np.isin(y_mask, i)    
    X = X_mask[mask]
    
    total_count = 400
    if i in unique: total_count /= len(unique)

    rand_mask = np.random.choice(len(X), min(len(X), int(total_count)), replace=False)
    X_all.append(X[rand_mask])
    y_all.append(np.repeat(i, len(X[rand_mask])))

X_all = np.concatenate(X_all)
y_all = np.concatenate(y_all)
print(X_all.shape, y_all.shape)

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=None, stratify=y_all)

randperm = np.random.permutation(len(y_train))
X_train = X_train[randperm]
y_train = y_train[randperm]

randperm = np.random.permutation(len(y_test))
X_test = X_test[randperm]
y_test = y_test[randperm]

print('og')
print('train:', Counter(y_train))
print('test:', Counter(y_test))

# for y in [y_train, y_test]:
#     mask = (y=='metaphase') | (y=='telophase') | (y=='anaphase') | (y=='prometaphase')# | (y=='prophase')
#     y[mask] = 'unique'
#     # y[y=='prometaphase']='prophase'

# print()
# print('unique')
# print('train:', Counter(y_train))
# print('test:', Counter(y_test))

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
print('classes', label)

#%% add extra dimension to input vectors for model input
X_train = X_train[..., np.newaxis]
# X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

#%% add gaussian noise
def add_noise(img):
    img = img - img.mean()
    img /= img.std()
    noise = np.random.randn(*img.shape)*.15
    img += noise
    return img

#%% add image preprocessing
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False, samplewise_center=True,
    featurewise_std_normalization=False, samplewise_std_normalization=True,
    zca_whitening=False, zca_epsilon=1e-06, rotation_range=360, width_shift_range=0.1,
    height_shift_range=0.1, brightness_range=[0.5,1.5], shear_range=1, zoom_range=[0.9,1.1],
    channel_shift_range=0.0, fill_mode='nearest', cval=0.0,
    horizontal_flip=True, vertical_flip=True, rescale=None,
    preprocessing_function=add_noise, data_format=None, validation_split=0.0, dtype=None
)
# datagen.fit(X_train)

from skimage.util import montage
og = X_train[y_train == 'prophase'][0]
a = np.concatenate([datagen.standardize(datagen.random_transform(og))[None,:] for i in range(25)], axis=0)
plt.subplot(121); plt.imshow(og, cmap='gray'); plt.axis('off'); plt.title('og')
plt.subplot(122); plt.imshow(montage(a.squeeze(), padding_width=10), cmap='gray'); plt.axis('off'); plt.title('preprocessed')

#%% define model
def get_conv(input_shape=(200, 200, 1), filename=None):

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

    tf.keras.layers.Conv2D(64, kernel_size=6, activation='relu'),
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

#%% learning rate scheduler
def scheduler(epoch, lr):
  if epoch < 30:
    return float(lr)
  else:
    return float(lr * tf.math.exp(-0.1))

lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

#%% compile and train model
# use loss=sparse_categorical_crossentropy for one-hot
model.compile(loss='categorical_crossentropy',
# optimizer=RMSprop(lr=0.001),
optimizer=tf.keras.optimizers.Adam(lr=0.001),
metrics=['accuracy',f1_metric])

early_stopping = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

log_dir = f'/home/nel/Desktop/{datetime.datetime.now().strftime("%m%d_%H%M")}'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(datagen.flow(X_train, y_train_one_hot, batch_size=64),
steps_per_epoch=len(X_train) // 64, 
epochs=50,
verbose=1,
validation_data=(np.array([datagen.standardize(i) for i in X_test.astype(float)]), y_test_one_hot),
class_weight=class_weight,
shuffle=True,
callbacks=[tensorboard_callback, lr_callback],#, early_stopping],
)

#%% evaluate model
model.evaluate(np.array([datagen.standardize(i) for i in X_test.astype(float)]), y_test_one_hot)

#%% training curves
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# plt.subplot(211)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()

# plt.subplot(212)
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()

#%% confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd

preds = model.predict(np.array([datagen.standardize(i) for i in X_test.astype(float)]))
results = np.array([label[i] for i in np.argmax(preds, axis=1)])

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, results), display_labels=label)
disp.plot(xticks_rotation='vertical')

con_matrix = pd.DataFrame(confusion_matrix(y_test, results), index = [i+'_true' for i in label], columns = [i+'_pred' for i in label])
print(con_matrix)

#%% ROC curve
col = np.argwhere(label=='prophase').squeeze()

fpr = {}
tpr = {}
roc_auc = {}
for i in range(len(label)):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], preds[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# LOG LOG
plt.plot(np.log(fpr[int(col)]), np.log(tpr[int(col)]))
plt.title('log-log ROC Prophase, AUC = '+str(roc_auc[int(col)].round(3)))
plt.xlabel('log(False Positive Rate)')
plt.ylabel('log(True Positive Rate)')
# plt.ylim([-.72, 0.05])
# plt.xlim([-6.5,0.1])
# # plt.savefig('log_roc_pro.pdf', dpi=300, bbox_inches="tight")

# REG
# plt.plot((fpr[int(col)]), (tpr[int(col)]))
# plt.title('ROC Prophase, AUC = '+str(roc_auc[int(col)].round(3)))
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.ylim([-.72, 0.05])
# plt.xlim([-6.5,0.1])
# # plt.savefig('roc_pro.pdf', dpi=300, bbox_inches="tight")

#%% save model/data
save_dir = f'/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/FCN_models/{datetime.date.today()}'
model_name = '_'.join(label)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
    
model_path = os.path.join(save_dir, model_name + '.h5')
model.save(model_path)
print(f'Trained model saved here: {model_path}')

#%% save/load models and data
np.savez(os.path.join(save_dir, model_name) + '_data', X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)