#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:59:31 2021

@author: nel
"""
import numpy as np
from skimage.util import montage
import tensorflow as tf
#%%
dat = np.load('/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/FCN_models/2021-08-31/anaphase_blank_blurry_edge_interphase_metaphase_prometaphase_prophase_telophase_data.npz')
X_train, y_train, X_test, y_test = dat['X_train'], dat['y_train'], dat['X_test'], dat['y_test']
#%%
plt.imshow(montage(X_train[y_train=='prophase'].squeeze()),cmap='gray',vmax=1000)


#%%
#%% load model

nn_path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/FCN_models/2021-08-31/anaphase_blank_blurry_edge_interphase_metaphase_prometaphase_prophase_telophase.h5'
label = nn_path.split('.')[-2].split('/')[-1].split('_')

def get_conv(input_shape=(200, 200, 1), filename=None):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape),
    # tf.keras.layers.LayerNormalization(axis=(1,2), trainable=False),   
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'), #(None, None, 1)),#X_train.shape[1:])),
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

    tf.keras.layers.Conv2D(len(label), kernel_size=1, activation='softmax'),
    # tf.keras.layers.Conv2D(len(np.unique(y_train)), kernel_size=1),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.GlobalMaxPooling2D(),
    # tf.keras.layers.Activation('softmax')
    ])

    if filename:
        model.load_weights(filename)
    return model

model = get_conv(filename=nn_path)
model.add(tf.keras.layers.LayerNormalization,)
model.add(tf.keras.layers.Flatten())



#%%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import pandas as pd


# X_test_norm = (X_test-X_test.mean(axis=(1,2))[:,None,None])/X_test.std(axis=(1,2))[:,None,None]
X_test_norm = (X_test-X_test.mean())/X_test.std()
preds = model.predict(np.array([(i) for i in X_test.astype(float)]))
results = np.array([label[i] for i in np.argmax(preds, axis=1)])

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, results), display_labels=label)
disp.plot(xticks_rotation='vertical')

con_matrix = pd.DataFrame(confusion_matrix(y_test, results), index = [i+'_true' for i in label], columns = [i+'_pred' for i in label])
print(con_matrix)