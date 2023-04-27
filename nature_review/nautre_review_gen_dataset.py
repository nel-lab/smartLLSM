#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:03:36 2023

@author: jimmytabet
"""

import os, random
from collections import Counter
import matplotlib.pyplot as plt

stages = ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']
folder = '/home/nel/Software/yolov5/smart_micro_datasetsv2/labels/test'
filelist = os.listdir(folder)
# shuffle list
random.shuffle(filelist)

labels = []

for file in filelist[:100]:
    with open(os.path.join(folder,file)) as f:
        lines = f.readlines()
        # get first entry of each line, the label index
        index = [i.split(' ')[0] for i in lines]
        # convert to label and append to list
        labels.extend([stages[int(i)] for i in index])
          
dist = Counter(labels)
print(dist)

names = list(dist.keys())
values = list(dist.values())

# a = []
# for i,j in zip(names,values):
    # a.append(f'{i}\n{j}')

plt.bar(range(len(dist)), values, tick_label=names)
plt.show()