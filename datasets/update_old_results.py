#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 12:40:40 2021

@author: jimmytabet
"""

#%%
import numpy as np
import glob, os, time

start = time.time()

folder = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
files = sorted(glob.glob(os.path.join(folder,'**','*.npz'), recursive=True))

for count, i in enumerate(files):
    initials = 'MC'
    
    # if 'data_1' in i:
    #     initials = 'KH'
    # elif 'data_2' in i:
    #     initials = 'MC'
    # else:
    #     raise ValueError('not data 1 or 2')
        
    dat = np.load(i, allow_pickle=True)
    
    filedic = dict(dat)
    if 'confirmed' in filedic:
        continue
    
    labels_dict = dat['labels_dict'].item()
    stage = [labels_dict[j] for j in dat['labels']]
    
    filedic['labels'] = stage
    filedic[f'labels_{initials}'] = stage
    filedic['confirmed'] = []
    
    if '_updated' in i:
        np.savez(''.join(i.split('_updated')), **filedic)
        os.remove(i)
    else:
        np.savez(i, **filedic)
        
    if count%100 == 0:
        print(f'{count}\tof\t{len(files)}')
    
print(f'{len(files)} fixed in {time.time()-start} seconds')

#%% old method - zipfile
# import zipfile
# import io
# import numpy as np

# # break on UserWarning for duplicate array
# import warnings
# warnings.simplefilter('error')

# # initial file
# filename = '/home/nel/Desktop/Scan_Iter_0000_0000_annotated.npz'

# # new data
# new_data = {'annotator': 'JT',
#             'rand_dat': np.random.rand(3,4)}

# # add new data to initial file
# with zipfile.ZipFile(filename, 'a') as zipf:
#     for k,v in new_data.items():
#         bio = io.BytesIO()
#         np.save(bio, v)
#         zipf.writestr(f'{k}.npy', data=bio.getbuffer().tobytes())