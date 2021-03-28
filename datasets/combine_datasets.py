#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 00:58:26 2021

@author: jimmytabet
"""

#%% COMBINE OLD AND NEW DATASETS
import numpy as np
import pandas as pd
from collections import Counter

old = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update_no_isolate_286.npz'
new = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/prophase_cell_slices_no_isolate_286.npz'

data_old = np.load(old)
data_new = np.load(new)

X_old, y_old = data_old['X'], data_old['y']
X_new, y_new = data_new['X'], data_new['y']
ID_new, slice_ID_new = data_new['ID'], data_new['slice_ID']

ID_old = []
slice_ID_old = []
for i in X_old:
    ID_old.append(np.nan)
    slice_ID_old.append(np.nan)
    
ID_old = np.array(ID_old)
slice_ID_old = np.array(slice_ID_old)

print(X_old.shape, y_old.shape)
print(X_new.shape, y_new.shape)

X_all = np.concatenate([X_new, X_old])
y_all = np.concatenate([y_new, y_old])
ID_all = np.concatenate([ID_new, ID_old])
slice_ID_all = np.concatenate([slice_ID_new, slice_ID_old])

print(X_all.shape, y_all.shape, ID_all.shape, slice_ID_all.shape)
print(np.unique(y_all), pd.unique(ID_all), pd.unique(slice_ID_all))

#%% REMOVE DUPLICATES (will keep first unique entry)
X_all_unique, indices = np.unique(X_all, return_index=True, axis=0)
y_all_unique = y_all[indices]
ID_all_unique = ID_all[indices]
slice_ID_all_unique = slice_ID_all[indices]

#%% CHECK THAT IDs WERE SAVED CORRECTLY
error = False
for uniq, orig in zip([ID_all_unique, slice_ID_all_unique], [ID_all, slice_ID_all]):
    uniq_no_nan = uniq[~np.isnan(uniq)]
    orig_no_nan = orig[~np.isnan(orig)]
    same_len = len(uniq_no_nan) == len(X_new)
    same_num = Counter(uniq_no_nan) == Counter(orig_no_nan)
    if not (same_len and same_num):
        error = True

#%% PRINT RESULTS

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
if not error:
    np.savez('/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/all_annotated_update_prophase_slices_no_isolate_286.npz',
             X=X_all_unique, y=y_all_unique, ID=ID_all_unique, slice_ID=slice_ID_all_unique)