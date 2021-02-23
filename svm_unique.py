#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 17:59:40 2021

@author: jimmytabet
"""

#%%
import numpy as np

from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

dat = np.load('/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated.npz', allow_pickle=True)
X = dat['X']
y = dat['y']

#%%
unique = ['prophase', 'metaphase','telophase','anaphase']

idx_unique = []
for idx in range(len(y)):
    if y[idx] in unique:
        idx_unique.append(idx)
    else:
        pass
    
X_unique = X[idx_unique]
y_unique = y[idx_unique]
print(X_unique.shape, y_unique.shape)

X_unique = np.array([x/np.max(x) for x in X_unique])

# from skimage.util import montage
# import matplotlib.pyplot as plt
# mont = montage(X_unique)
# plt.imshow(mont, cmap = 'gray')

#%%
X_train, X_test, y_train, y_test = train_test_split(X_unique.reshape(X_unique.shape[0], -1), y_unique, test_size=0.3, random_state=0)
tuned_parameters = {'C': [0.0001, 0.001, 0.01, 0.1, 1]}
clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1_weighted', verbose = 10)
clf.fit(X_train, y_train)
print(clf.best_params_)
model = clf.best_estimator_
#model=SVC(probability=True).fit(X_train,y_train)
#prob = model.predict_proba(X_test)
results = model.predict(X_test)
classes = model.classes_
print('         accuracy:', metrics.accuracy_score(y_test, results))
print('balanced accuracy:', metrics.balanced_accuracy_score(y_test, results))
print('               f1:', metrics.f1_score(y_test, results, average = 'weighted'))