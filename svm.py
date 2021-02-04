#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 21:07:47 2021

@author: jimmytabet
"""

#%% imports and data
import numpy as np
from skimage.util import montage
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

data = np.load('annotated.npz', allow_pickle=True)
X = data['X']
y = data['y']
fls = data['fls']

new_cells=np.load('/Users/jimmytabet/NEL/Projects/Smart Micro/datasets/cell_dataset_isolate_9224.npy')
new_cells = new_cells.reshape(new_cells.shape[0],-1)

#%% optionally transform annotated data
from skimage import transform
from skimage.transform import rotate, AffineTransform
from skimage.util import random_noise

X_temp = np.stack([cv2.resize(i.astype(np.float32), (250,250)) for i in X])

X_transform=[]
tf = AffineTransform(shear=-0.3)
for x in X_temp:
    x -= x.mean()
    x /= x.std()
    x = transform.warp(x, tf, order=1, preserve_range=True, mode='wrap')
    x = transform.rescale(x, 1.1)
    x = rotate(x, angle=np.random.random()*360)  
    x = np.flipud(x)
    x = np.fliplr(x)
    # x = random_noise(x, var=0.001**2)
    X_transform.append(x)

X_transform = np.array(X_transform)

#%% resize for SVM
X = np.stack([cv2.resize(i, (191,191)) for i in X])

#%% optionally fft
X_fft=[]
for x in X:
    f = np.fft.fft2(x)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    X_fft.append(cv2.resize(magnitude_spectrum, X.shape[1:]))

X_fft= np.array(X_fft)

#%% reshape for SVM
X = X_fft.reshape(X_fft.shape[0],-1)

#%% SVM and GridSearch
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
tuned_parameters = {'kernel': ['rbf', 'linear', 'poly', 'sigmoid'], 'C': [0.1, 1, 10, 100, 1000]}

clf = GridSearchCV(SVC(), tuned_parameters, scoring='f1')
clf.fit(X_train, y_train)
print(clf.best_params_)
model = clf.best_estimator_
print('accuracy:', metrics.accuracy_score(y_test, model.predict(X_test)))
print('f1:', metrics.f1_score(y_test, model.predict(X_test)))

#%% run through new_cells
results = model.predict(new_cells)

#%% show prophase/not prophase cells
prophase = new_cells[results == 1]
prophase = prophase.reshape(prophase.shape[0], 191, 191)

not_prophase = new_cells[results == 0]
not_prophase = not_prophase.reshape(not_prophase.shape[0], 191, 191)

p_mont = montage(prophase[:400], rescale_intensity=True)
notp_mont = montage(not_prophase[:400], rescale_intensity=True)

plt.imshow(p_mont, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Prophase: '+str(prophase.shape[0])+' of '+str(new_cells.shape[0]))
plt.savefig('/Users/jimmytabet/Desktop/prophase_fft', dpi=1000, bbox_inches='tight')
plt.cla()
plt.imshow(notp_mont, cmap='gray')
plt.xticks([])
plt.yticks([])
plt.title('Not Prophase: '+str(not_prophase.shape[0])+' of '+str(new_cells.shape[0]))
plt.savefig('/Users/jimmytabet/Desktop/not prophase_fft', dpi=1000, bbox_inches='tight')
plt.close('all')