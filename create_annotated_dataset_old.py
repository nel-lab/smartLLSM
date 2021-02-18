#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 23:02:59 2021

@author: jimmytabet
"""

#%% get annotated dataset
import glob
import numpy as np
import matplotlib.pyplot as plt
import os

X = []
y = []
fls = glob.glob('annotated examples/*.tif')
for fl in fls:
    im = plt.imread(fl)
    if "prophase" in fl.lower():
        label = 1
    else:
        label = 0
    X.append(im)
    y.append(label)
    
# np.savez('/home/nel-lab/Desktop/Jimmy/annotated', X=X, y=y, fls=fls)

#%% show each class
def row_col(num):
    sqr = np.sqrt(np.ceil(np.sqrt(num))**2)
    row, col = sqr,sqr
    diff1 = row*col-num
    if diff1 >= sqr and sqr>1:
    
        return int(row), int(col-1)
    
    return int(row), int(col)

#%%
prophase, anaphase, telophase, metaphase = [],[],[],[]
for f in os.listdir('annotated examples'):
    try:
        im = plt.imread('annotated examples/'+f)
    except:
        continue
    label = f.split('_')[-1].split('.')[0].lower()
    if label == 'prophase': prophase.append(im)
    if label == 'anaphase': anaphase.append(im)
    if label == 'telophase': telophase.append(im)
    if label == 'metaphase': metaphase.append(im)

all_stages = [prophase, anaphase, telophase, metaphase]

#%%
name = ['prophase', 'anaphase', 'telophase', 'metaphase']
for n,stage in enumerate(all_stages):
    row,col = row_col(len(stage))
    fig, ax = plt.subplots(row,col)
    fig.set_tight_layout(True)
    fig.suptitle(name[n])
    for i,im in enumerate(stage):
        a = fig.axes[i]
        a.imshow(im, cmap = 'gray')
        a.set_xticks([])
        a.set_yticks([])
    
    for a in fig.axes[i+1:]:
        a.remove()
        
    DPI = fig.get_dpi()
    fig.set_size_inches(1920.0/float(DPI),1080.0/float(DPI))
    fig.savefig('annotated/'+name[n], dpi=DPI, bbox_inches='tight')