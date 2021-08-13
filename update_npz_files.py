#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:40:40 2021

@author: jimmytabet
"""
    
#%%
import zipfile
import io
import numpy as np

# break on UserWarning for duplicate array
import warnings
warnings.simplefilter('error')

# initial file
filename = '/home/nel/Desktop/Scan_Iter_0000_0000_annotated.npz'

# new data
new_data = {'annotator': 'JST',
            'rand_dat': np.random.rand(3,4)}

# add new data to initial file
with zipfile.ZipFile(filename, 'a') as zipf:
    for k,v in new_data.items():
        bio = io.BytesIO()
        np.save(bio, v)
        zipf.writestr(f'{k}.npy', data=bio.getbuffer().tobytes())