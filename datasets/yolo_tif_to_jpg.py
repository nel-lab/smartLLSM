#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 16:36:01 2021

@author: jimmytabet
"""

import glob, os
import matplotlib

base_folder = '/home/nel/Software/yolov5/smart_micro_datasetsv2/data_3'

files = sorted(glob.glob(os.path.join('/home/nel/Desktop/Smart Micro/ShannonEntropy_2Dimgs/data_3', '**/*.tif')))

for count, file in enumerate(files):
    if count%100==0:
        print(count)

    raw = matplotlib.pyplot.imread(file)
    matplotlib.image.imsave(os.path.join(base_folder, f'{count}.jpg'), raw, cmap='gray', dpi=300)