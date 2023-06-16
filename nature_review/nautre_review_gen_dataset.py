#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 09:03:36 2023

@author: jimmytabet
"""

import os, random, pickle
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

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

LIMIT = 1000000

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

# plot training cells distribution
dist = dict(final_dict.most_common())
names = list(dist.keys())
values = list(dist.values())
a = plt.bar(range(len(dist)), values, tick_label=names)
plt.bar_label(a)
plt.title(f'{LIMIT} Cells per Mitotic Phase')
plt.show()

#%% pickle files
with open(f'/home/nel/Desktop/{LIMIT}_training_files.pickle', 'wb') as f:
    pickle.dump(files, f)

#%% save bar plot
plt.savefig(f'/home/nel/Desktop/{LIMIT}_training.pdf', dpi=300, transparent=True)

#%% copy training files for yolov5 training
import os, pickle, shutil

LIMIT = 1000000
with open(f'/home/nel/Desktop/{LIMIT}_training_files.pickle', 'rb') as f:
    files = pickle.load(f)

# copy test files
for fold in ['images', 'labels']:
    og_path = os.path.join('/home/nel/Software/yolov5/smart_micro_datasetsv2/', fold, 'test')
    new_path = os.path.join(f'/home/nel/Software/yolov5/smart_micro_datasets{LIMIT}/', fold, 'test')
    shutil.copytree(og_path, new_path)

# copy images/labels
for fold,ext in zip(['images', 'labels'], ['.jpg','.txt']):
    # make train folder if not already made
    if not os.path.exists(f'/home/nel/Software/yolov5/smart_micro_datasets{LIMIT}/{fold}/train'): os.makedirs(f'/home/nel/Software/yolov5/smart_micro_datasets{LIMIT}/{fold}/train')
    # copy files
    for f in files:
        og_path = os.path.join('/home/nel/Software/yolov5/smart_micro_datasetsv2/', fold, 'train', f.split('.')[0]+ext)
        new_path = os.path.join(f'/home/nel/Software/yolov5/smart_micro_datasets{LIMIT}/', fold, 'train', f.split('.')[0]+ext)
        shutil.copyfile(og_path, new_path)
        
#%% terminal code to train
'''
python train.py --weights "yolov5s.pt" --data "smart_micro.yaml" --hyp "/home/nel/Software/yolov5/runs/train/exp20/hyp.yaml" --epochs 300 --batch-size 16 --imgsz 800 --patience 30
'''

#%% plot multiple bar charts together
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

og = {'interphase': 26284,
      'blurry': 5787,
      'telophase': 542,
      'metaphase': 631,
      'prophase': 258,
      'prometaphase': 293,
      'anaphase': 84}
      
t100 = {'interphase': 2608,
       'blurry': 707,
       'telophase': 105,
       'metaphase': 105,
       'prophase': 101,
       'prometaphase': 102,
       'anaphase': 84}

t50 = {'interphase': 1344,
       'blurry': 381,
       'telophase': 53,
       'metaphase': 51,
       'prophase': 50,
       'prometaphase': 50,
       'anaphase': 50}

t10 = {'interphase': 266,
       'blurry': 60,
       'telophase': 10,
       'metaphase': 10,
       'prophase': 10,
       'prometaphase': 10,
       'anaphase': 10}
       

data = [og, t100, t50, t10]
plot_labels = ['og','100 mitotic cells','50 mitotic cells','10 mitotic cells']
ind = np.arange(len(og.keys()))
width = 1/len(data)-0.05

for i, (dic,plot_label) in enumerate(zip(data,plot_labels)):
    a = plt.bar(ind+width*i, dic.values(), width, label = plot_label)
    plt.bar_label(a)

plt.xticks(ind+width*i/2, og.keys())
plt.legend()

#%% save cumulative bar plot
plt.savefig('/home/nel/Desktop/all_training.pdf', dpi=300, transparent=True)

