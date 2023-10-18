#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  1 17:34:24 2023

@author: jimmytabet
"""

'''
UPDATE PATHS to fit your system
1. edit yolov5/utils/metrics.py script
    add 'np.savez('/home/nel/Desktop/pr_data.npz', px=px, py=py, names=names)' to end of plot_pr_curve function (around line 320)
2. python val.py --weights "/home/nel/Software/yolov5/runs/train/exp20/weights/best.pt" --data "smart_micro.yaml" --batch-size 16 --imgsz 800
3. run this script
'''

import numpy as np
import pandas as pd

dat = np.load('/home/nel/Desktop/pr_data.npz', allow_pickle=True)
px = dat['px']
py = dat['py']
names_dict = dat['names'].item()
names = [names_dict[i] for i in range(len(names_dict))]

# add all_classes data
all_classes = py.mean(1)[..., None]
py = np.append(py,all_classes, axis=1)
names.append('all_classes')

df = pd.DataFrame(py, index=px, columns=names)
df.index.names = ['x']
df.to_csv('/home/nel/Desktop/F1G_data.csv')