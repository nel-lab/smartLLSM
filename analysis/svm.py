#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:22:47 2021

@author: jimmytabet
"""

#%% LOAD DATA
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

path = '/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update_prophase_slices.npz'
# path = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update_prophase_slices.npz'
dat = np.load(path, allow_pickle=True)
X_org, y_org, ID, slice_ID = dat['X'], dat['y'], dat['ID'], dat['slice_ID']

#%% STD AND SPLIT DATA
X_std = np.stack([x/np.max(x) for x in X_org])
X_std = X_std.astype(np.float32)
# plt.imshow(montage(X_std[:49], padding_width=10), cmap='gray')

X_train, X_test, y_train, y_test = train_test_split(X_std, y_org, test_size=0.3, random_state=0)

#%% AUGMENT DATA
import cv2
from skimage.transform import rotate, AffineTransform, warp
import random
from skimage.util import random_noise

def anticlockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

def v_flip(image):
    return np.flipud(image)

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)

def warp_shift(image): 
    transform = AffineTransform(translation=(np.random.randint(5),np.random.randint(5)))  #chose x,y values according to your convinience    
    warp_image = warp(image, transform, mode="wrap")    
    return warp_image

transformations = {'rotate anticlockwise': anticlockwise_rotation,
                   'rotate clockwise': clockwise_rotation,
                   'horizontal flip': h_flip, 
                   'vertical flip': v_flip,
                   'warp shift': warp_shift,
                   'adding noise': add_noise,
                   'blurring image':blur_image
                 }

X_train_aug = []
y_train_aug = []
stages = np.unique(y_train)
for stage in stages:
    if stage == 'prophase':
        images_to_generate=1000
    else:
        images_to_generate=round(1000/(len(stages)-1))
   
    images=X_train[np.argwhere(y_train==stage).ravel()]
   
    i=0
    while i<images_to_generate:    
        original_image=random.choice(images)
        transformed_image=original_image.copy()
       
        n = 0       #variable to iterate till number of transformation to apply
        transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image
        
        while n <= transformation_count:
            key = random.choice(list(transformations)) #randomly choosing method to call
            transformed_image = transformations[key](transformed_image)
            n = n + 1
        
        X_train_aug.append(transformed_image.astype(np.float32))
        y_train_aug.append(stage)
        i += 1

X_train_aug = np.array(X_train_aug).astype(np.float32)
y_train_aug = np.array(y_train_aug)

# only add certain class to dataset
# new_cells = []
# new_cells.append(transformed_image.astype(np.float32))
# new_cells = np.array(new_cells)
# X_std = np.concatenate([X_std, new_cells])
# y = np.append(y, ['prophase' for i in new_cells])

#%% TRAIN (std datatset w/ class weights best so far)
model = SVC(class_weight = 'balanced', probability = True)
model.fit(X_train_aug.reshape([len(X_train_aug),-1]), y_train_aug)

#%% TEST
results = model.predict(X_test.reshape([len(X_test), -1]))
print('          accuracy:', metrics.accuracy_score(y_test, results))
print(' balanced accuracy:', metrics.balanced_accuracy_score(y_test, results))
print('                f1:', metrics.f1_score(y_test, results, average = 'weighted'))
print('   prophase recall:', metrics.recall_score(y_test, results, labels=['prophase'], average = 'weighted'))
print('prophase precision:', metrics.precision_score(y_test, results, labels=['prophase'], average = 'weighted'))

#%% SHOW PROPHASE TP
pro = np.intersect1d(np.where(y_test == 'prophase'), np.where(results == 'prophase'))
plt.imshow(montage(X_std[pro], padding_width=10), cmap='gray')

#%% CONFUSION MATRIX/CLASSIFICATION REPORT
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
classes = model.classes_

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, results), display_labels=classes)
disp.plot()

con_matrix = pd.DataFrame(confusion_matrix(y_test, results), index = [str(i)+'_true' for i in classes], columns = [str(i)+'_pred' for i in classes])
print(con_matrix)

print(classification_report(y_test, results))

#%% TRAIN ONE HOT
y_one = y_train_aug=='prophase'
# tuned_parameters = {'C': [0.1, 1, 10, 100], 'class_weight': ['balanced'], 'probability': [True]}
# clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1', verbose=1)
# clf.fit(X_train_aug.reshape([len(X_train_aug),-1]), y_one)
# print(clf.best_params_)
# model = clf.best_estimator_

model = SVC(probability=True)
model.fit(X_train_aug.reshape([len(X_train_aug),-1]), y_one)

#%% TEST ONE HOT
y_test = y_test=='prophase'
results = model.predict(X_test.reshape([len(X_test), -1]))
print('          accuracy:', metrics.accuracy_score(y_test, results))
print(' balanced accuracy:', metrics.balanced_accuracy_score(y_test, results))
print('                f1:', metrics.f1_score(y_test, results))
print('   prophase recall:', metrics.recall_score(y_test, results))
print('prophase precision:', metrics.precision_score(y_test, results))

#%% prophase with highest probability.
prob = model.predict_proba(X_test.reshape([len(X_test), -1]))
prophase_col = 0    # for one hot
# prophase_col = np.argwhere(classes=='prophase')[0][0]
highest = prob[:,prophase_col].argsort()

plt.imshow(montage(X_test[highest[:49]].reshape([-1,191,191]), padding_width=10), cmap='gray')

#%%---------------------------------OTHER-------------------------------------
#%% random forest
# from sklearn.ensemble import RandomForestClassifier

# X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X.reshape(X_std.shape[0], -1), y, test_size=0.3, random_state=0)

# model = RandomForestClassifier()
# model.fit(X_train_rf, y_train_rf)
# results_rf = model.predict(X_test_rf)
# print('         accuracy:', metrics.accuracy_score(y_test_rf, results_reg_nw))
# print('balanced accuracy:', metrics.balanced_accuracy_score(y_test_rf, results_reg_nw))
# print('               f1:', metrics.f1_score(y_test_reg_nw, results_rf, average = 'weighted'))

#%% remove 'unknown' and 'interphase' labels
# remove = np.where((y=='unknown') | (y=='interphase'))
# X_clean = np.delete(X, remove, axis=0)
# y_clean = np.delete(y, remove, axis=0)

#%% balance dataset with 100 examples of each class
# from collections import Counter
# import random

# keep = []
# for k,v in Counter(y).items():
#   ids = np.argwhere(y==k).ravel()
#   if v > 100:
#     keep.append(random.sample(list(ids),100))
#   else:
#     keep.append(ids)

# keep = np.array([i for j in keep for i in j])
# X = X[keep]
# y = y[keep]
# print(Counter(y))

#%% fft
# X_fft=[]
# for x in X:
#     f = np.fft.fft2(x)
#     fshift = np.fft.fftshift(f)
#     magnitude_spectrum = 20*np.log(np.abs(fshift))
#     X_fft.append(cv2.resize(magnitude_spectrum, X.shape[1:]))

# X_fft= np.array(X_fft)

#%% grid search
# tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100, 1000], 'probability': [True]}
# clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_weighted', verbose=1)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# model = clf.best_estimator_

#%% show prophase/not prophase cells
# prophase = X_test[results == 1].reshape(-1, 191, 191)
# not_prophase = X_test[results == 0].reshape(-1, 191, 191)

# p_mont = montage(prophase[:400], rescale_intensity=True)
# notp_mont = montage(not_prophase[:400], rescale_intensity=True)

# plt.imshow(p_mont, cmap='gray')
# plt.xticks([])
# plt.yticks([])
# plt.title('Prophase: '+str(prophase.shape[0])+' of '+str(new_cells.shape[0]))
# # plt.savefig('/Users/jimmytabet/Desktop/prophase_fft', dpi=1000, bbox_inches='tight')
# plt.cla()

# plt.imshow(notp_mont, cmap='gray')
# plt.xticks([])
# plt.yticks([])
# plt.title('Not Prophase: '+str(not_prophase.shape[0])+' of '+str(new_cells.shape[0]))
# # plt.savefig('/Users/jimmytabet/Desktop/not prophase_fft', dpi=1000, bbox_inches='tight')
# plt.close('all')