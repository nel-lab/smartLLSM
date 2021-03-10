#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 00:58:26 2021

@author: jimmytabet
"""

#%% COMBINE OLD AND NEW DATASETS
import numpy as np

old = '/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update.npz'
new = '/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/prophase_cell_slices.npz'

data_old = np.load(old)
data_new = np.load(new)

X_old, y_old = data_old['X'], data_old['y']
X_new, y_new = data_new['X'], data_new['y']

print(X_old.shape, y_old.shape)
print(X_new.shape, y_new.shape)

X_all = np.concatenate([X_old, X_new])
y_all = np.concatenate([y_old, y_new])

print(X_all.shape, y_all.shape)

#%% REMOVE DUPLICATES
X_all_unique, indices = np.unique(X_all, return_index=True, axis=0)
y_all_unique = y_all[indices]

#%% PRINT RESULTS
from collections import Counter

stages_dict = dict(Counter(y_all_unique))
stages_dict = dict(sorted(stages_dict.items(), key = lambda item: item[1], reverse = True))

no_cells = len(y_all_unique)

for k,v in stages_dict.items():
    if len(k) < 7:
        print(f'{k}: \t\t{100*v/no_cells:-3.0f}% ({v})')
    else:
        print(f'{k}: \t{100*v/no_cells:-3.0f}% ({v})')
        
print('-------------------------')
print(f'             100% ({no_cells})')

#%% SAVE
np.savez('/Users/jimmytabet/NEL/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update_prophase_slices.npz', X=X_all_unique, y=y_all_unique)