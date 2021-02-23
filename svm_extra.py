#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:22:47 2021

@author: jimmytabet
"""

#%% LOAD ALL ANNOTATED DATA
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt
import cv2
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

dat = np.load('/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated.npz', allow_pickle=True)
X = dat['X']
y = dat['y']

#%% optionally transform annotated data
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise

X_temp = np.stack([cv2.resize(i.astype(np.float32), (250,250)) for i in X])

X_transform=[]
tf = AffineTransform(shear=-0.03)
for x in X_temp:
    x -= x.mean()
    x /= x.std()
    x = transform.warp(x, tf, order=1, preserve_range=True, mode='wrap')
    x = transform.rescale(x, 1.1)
    x = rotate(x, np.random.random()*360)  
    x = rotate(x, np.random.choice([0,90,180,270]))
    x = np.flipud(x)
    x = np.fliplr(x)
    # x = random_noise(x, var=0.001**2)
    X_transform.append(x)

X_transform = np.array(X_transform)

#%% resize for SVM - temp_data = X OR X_transform
temp_data = X_transform
X = np.stack([cv2.resize(i, (191,191)) for i in temp_data])

#%% show transformed data
plt.imshow(montage(dat['X'][:49], padding_width = 10), cmap='gray')
# for i in [123,1765,654]:
#     plt.imshow(X[i], cmap='gray')
#     plt.pause(1)

#%% show std data
X_std = np.stack([x/np.max(x) for x in dat['X']])
plt.imshow(montage(X_std[:49], padding_width=10), cmap='gray')

#%% train on std dataset (BEST SO FAR)
X_train, X_test, y_train, y_test = train_test_split(X_std.reshape(X_std.shape[0], -1), y, test_size=0.3, random_state=0)
# tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100, 1000], 'probability': [True]}

# clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_weighted', verbose=1)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# model = clf.best_estimator_
model = SVC(class_weight = 'balanced', verbose = 1)
model.fit(X_train, y_train)
results = model.predict(X_test)
print('         accuracy:', metrics.accuracy_score(y_test, results))
print('balanced accuracy:', metrics.balanced_accuracy_score(y_test, results))
print('               f1:', metrics.f1_score(y_test, results, average = 'weighted'))

#%% train on reg dataset w/weights
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(dat['X'].reshape(dat['X'].shape[0], -1), y, test_size=0.3, random_state=0)
# tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100, 1000], 'probability': [True]}

# clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_weighted', verbose=1)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# model = clf.best_estimator_
model = SVC(class_weight = 'balanced')
model.fit(X_train_reg, y_train_reg)
results_reg = model.predict(X_test_reg)
print('         accuracy:', metrics.accuracy_score(y_test_reg, results_reg))
print('balanced accuracy:', metrics.balanced_accuracy_score(y_test_reg, results_reg))
print('               f1:', metrics.f1_score(y_test_reg, results_reg, average = 'weighted'))

#%% train on reg dataset no weights
X_train_reg_nw, X_test_reg_nw, y_train_reg_nw, y_test_reg_nw = train_test_split(dat['X'].reshape(dat['X'].shape[0], -1), y, test_size=0.3, random_state=0)
# tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100, 1000], 'probability': [True]}

# clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_weighted', verbose=1)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# model = clf.best_estimator_
model = SVC()
model.fit(X_train_reg_nw, y_train_reg_nw)
results_reg_nw = model.predict(X_test_reg_nw)
print('         accuracy:', metrics.accuracy_score(y_test_reg_nw, results_reg_nw))
print('balanced accuracy:', metrics.balanced_accuracy_score(y_test_reg_nw, results_reg_nw))
print('               f1:', metrics.f1_score(y_test_reg_nw, results_reg_nw, average = 'weighted'))

#%% show TP for prophase
pro = np.intersect1d(np.where(y_test == 'prophase'), np.where(results == 'prophase'))
plt.imshow(montage(X_std[pro], padding_width=10), cmap='gray')

#%% confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
classes = model.classes_

print(pd.DataFrame(confusion_matrix(y_test, results), index = classes, columns = classes))

disp = ConfusionMatrixDisplay(confusion_matrix(y_test, results), display_labels=classes)
disp.plot()

#%% train on one hot w/prophase
y = dat['y']=='prophase'
X_train, X_test, y_train, y_test = train_test_split(X.reshape(X.shape[0], -1), y, test_size=0.3, random_state=0)
tuned_parameters = {'C': [0.1, 1, 10], 'probability': [True], 'class_weight': ['balanced']}

clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1', verbose=1)
clf.fit(X_train, y_train)
print(clf.best_params_)
model = clf.best_estimator_
prob = model.predict_proba(X_test)
results = model.predict(X_test)
print('         accuracy:', metrics.accuracy_score(y_test, results))
print('balanced accuracy:', metrics.balanced_accuracy_score(y_test, results))
print('               f1:', metrics.f1_score(y_test, results))

#%% prophase with highest probability... (unfinished)
prophase_col = np.argwhere(classes=='prophase')[0][0]
highest = prob[:,prophase_col].argsort()

#%% random forest...
from sklearn.ensemble import RandomForestClassifier

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X.reshape(X_std.shape[0], -1), y, test_size=0.3, random_state=0)
# tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100, 1000], 'probability': [True]}

# clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_weighted', verbose=1)
# clf.fit(X_train, y_train)
# print(clf.best_params_)
# model = clf.best_estimator_
model = RandomForestClassifier()
model.fit(X_train_rf, y_train_rf)
results_rf = model.predict(X_test_rf)
print('         accuracy:', metrics.accuracy_score(y_test_rf, results_reg_nw))
print('balanced accuracy:', metrics.balanced_accuracy_score(y_test_rf, results_reg_nw))
print('               f1:', metrics.f1_score(y_test_reg_nw, results_rf, average = 'weighted'))