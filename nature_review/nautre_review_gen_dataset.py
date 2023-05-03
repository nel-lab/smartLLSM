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
important = ['anaphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']

folder = '/home/nel/Software/yolov5/smart_micro_datasetsv2/labels/train'
filelist = os.listdir(folder)

'''
Total Training Cells
Counter({'interphase': 26284,
         'blurry': 5787,
         'metaphase': 631,
         'prometaphase': 293,
         'telophase': 542,
         'prophase': 258,
         'anaphase': 84})
'''

# shuffle list
random.shuffle(filelist)

labels = []
files = []

LIMIT = 200

final_dict = Counter(labels)

for file in filelist:
    with open(os.path.join(folder,file)) as f:
        lines = f.readlines()
        # get first entry of each line, the label index
        index = [i.split(' ')[0] for i in lines]
        # convert to label
        phase = [stages[int(i)] for i in index]

        # check if important cell or mitotic phase limits are reached
        # flag to ADD image to list
        ADD = False
        for k,v in Counter(phase).items():
            # if phase is not important or limit is reached, pass (ADD = False)
            if k not in important or final_dict[k]+v > LIMIT:
                pass
            # otherwise add to labels (ADD = True)
            else:
                ADD = True
                
        if ADD:
            labels.extend(phase)
            files.append(file)
        
        # update final_dict
        final_dict = Counter(labels)

print(Counter(labels))

# plot training cells distribution
dist = dict(final_dict.most_common())
names = list(dist.keys())
values = list(dist.values())
a = plt.bar(range(len(dist)), values, tick_label=names)
plt.bar_label(a)
plt.title(f'{LIMIT} Cells per Mitotic Phase')
plt.show()