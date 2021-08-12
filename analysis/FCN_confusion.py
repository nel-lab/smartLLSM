#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:35:56 2021

@author: jimmytabet
"""

#%% imports
import glob, cv2, os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

import datetime

def get_conv(input_shape=(286, 286, 1), filename=None):

  model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', input_shape=input_shape), #(None, None, 1)),#X_train.shape[1:])),
  tf.keras.layers.MaxPooling2D(2, 2),

  tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
  tf.keras.layers.MaxPooling2D(2,2),

  tf.keras.layers.Conv2D(64, kernel_size=8, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.BatchNormalization(),

  tf.keras.layers.Conv2D(5, kernel_size=1, activation='softmax'),
  # tf.keras.layers.Conv2D(len(np.unique(y_train)), kernel_size=1),
  # tf.keras.layers.Dropout(0.5),
  # tf.keras.layers.BatchNormalization(),
  # tf.keras.layers.GlobalMaxPooling2D(),
  # tf.keras.layers.Activation('softmax')
  ])

  if filename:
    model.load_weights(filename)
  return model

#%% inputs

# input_size = (286, 286)
# half_size = input_size[0]//2

tile_path = '/home/nel-lab/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
# files = sorted(glob.glob(os.path.join(tile_path, '**','*.npz'), recursive=True))
# files = [fil for fil in files if not 'og_backup' in fil]

# nn_path = '/home/nel-lab/Desktop/Jimmy/Smart Micro/FCN/2021-06-16-18-58-19.214988.h5'
nn_path = '/home/nel/Desktop/Smart Micro/FCN/2021-08-04-15-57-31.199175_edge.h5'

heatmodel = get_conv(input_shape=(None, None, 1), filename=nn_path)

#%% loop
gt_pro = []
FCN_max = []
center_dist = []

no_files = 0

for count, file in enumerate(files):
    
    # only use files that were not part of training set
    if not int(datetime.datetime.fromtimestamp(os.path.getmtime(file)).strftime('%m%d'))>617:
        continue
    
    no_files += 1
    
    dat = np.load(file, allow_pickle=True)
    
    raw = dat['raw']
    mask = dat['masks']
    labels = dat['labels']
    labels_dict = dat['labels_dict'].item()
    
    stages = [labels_dict[lab] for lab in labels]
    
    pro_id = np.where(np.array(stages) == 'prophase')[0]+1
    gt_pro.append(len(pro_id))
    
    raw_FCN = raw - raw.mean()
    raw_FCN /= raw.std()
    raw_FCN = raw_FCN[np.newaxis,...,np.newaxis]

    res = heatmodel.predict(raw_FCN)
  
    # NEW STUFF--------
    bord = 4
    res_new = tf.transpose(res, perm=[0, 3, 1, 2]).numpy()
    res = []
    for i in res_new.squeeze():
        res.append(cv2.copyMakeBorder(i, bord, bord, bord, bord, 0))
        
    res = np.array(res)
    res = res[np.newaxis,...]
    # NEW STUFF--------
    
  
    # pro = res[:,:,:,3].squeeze()
    pro = res[:,3,:,:].squeeze()
  
    # if pro.max()>0.5:
    #   pass
    #   #run=False
    # else:
    #   continue  
    
    FCN_pro_max = pro.max()
    FCN_max.append(FCN_pro_max)
    
    # skip if no prophase cells in tile (distance = inf)
    if len(pro_id) == 0:
        center_dist.append(np.inf)
        continue
    

    ms_h = 0#np.ceil(pro.shape[0]/2)
    ms_w = 0#np.ceil(pro.shape[0]/2)
  
    top_left = cv2.minMaxLoc(pro)[-1]
    sh_y, sh_x = top_left
    
    # if sh_y == 0 or sh_y == pro.shape[0]-1 or sh_x == 0 or sh_x == pro.shape[0]-1:
    if sh_y == 0+bord or sh_y == pro.shape[0]-bord-1 or sh_x == 0+bord or sh_x == pro.shape[0]-bord-1:
    # if sh_y == pro.shape[0]-1 or sh_x == pro.shape[0]-1:
      # continue
      sh_y_n = sh_y
      sh_x_n = sh_x
    
    else:
      # peak registration
      log_xm1_y = np.log(pro[sh_x - 1, sh_y])
      log_xp1_y = np.log(pro[sh_x + 1, sh_y])
      log_x_ym1 = np.log(pro[sh_x, sh_y - 1])
      log_x_yp1 = np.log(pro[sh_x, sh_y + 1])
      four_log_xy = 4 * np.log(pro[sh_x, sh_y])
  
      sh_x_n = (sh_x - ms_h + (log_xm1_y - log_xp1_y)/(2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
      sh_y_n = (sh_y - ms_w + (log_x_ym1 - log_x_yp1)/ (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
    
    # factor = raw_FCN.shape[1]/pro.shape[0]
    factor = raw_FCN.shape[1]/(pro.shape[0]-bord)
    FCN_x = factor*sh_y_n
    FCN_y = factor*sh_x_n
    
    # calc gt prophase centroid(s)
    dists = []
    for mask_id in pro_id:

        # get moments for cell to calculate center of mass
        M = cv2.moments(1*(mask==mask_id), binaryImage=True)
    
        if M["m00"] != 0:
          cX = M["m10"] / M["m00"]
          cY = M["m01"] / M["m00"]
        else:
          cX, cY = 0, 0 
    
        dist = np.linalg.norm([cX-FCN_x, cY-FCN_y])
        dists.append(dist)
        
    # keep shorter distance
    center_dist.append(min(dists))
    
    if count % 100 == 0:
        print(f'{count} of {len(files)}')

print(no_files)

#%% arrays
gt_pro = np.array(gt_pro)
FCN_max = np.array(FCN_max)
center_dist = np.array(center_dist)

#%% confusion
dist_thresh = 100
TPR_all = []
FPR_all = []
for pro_thresh in np.linspace(0,1,1001):
    # pro_thresh = .25
    
    FCN_pro = FCN_max > pro_thresh
    # FCN_pro = np.array([1,0,0,1,1,0])
    # gt_pro =  np.array([1,0,1,1,0,2])
    # center_dist = np.array([50,np.inf, 70, 150, np.inf, 200])
    
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    # oor = 0
    
    for FCN, gt, dis in zip(FCN_pro, gt_pro, center_dist):
        if gt == 0:
            if FCN == 0:
                tn += 1
            else:
                fp += 1
                
        else:
            if FCN == 0:
                fn += 1
            elif dis < dist_thresh:
                tp += 1
            else:
                # oor += 1
                fp += 1
                fn += 1
                
    print('thresh:', pro_thresh)
    print('tp:', tp)
    print('tn:', tn)
    print('fp:', fp)
    print('fn:', fn)
    # print('OOR:', oor)
    
    TPR_all.append(tp/(tp+fn))
    FPR_all.append(fp/(fp+tn))
    
#%% short confusion
print('       actual')
print('        +  -')
print('pred +',tp,fp)
print('     -',fn,tn)  
 
#%% ROC curve
plt.plot((FPR_all), (TPR_all))
plt.xlim([0,1])
plt.ylim([0,1])
plt.plot([0,1],[0,1], ls='--', c='k')
# plt.plot(np.log(FPR_all), np.log(TPR_all))
# plt.xlim([-6.5, .1])
# plt.ylim([-.78, .03])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'ROC, AUC = {metrics.auc(FPR_all, TPR_all).round(3)}')

#%% raw tif files
import glob,os
path = '/home/nel/Desktop/Smart Micro/ShannonEntropy_2Dimgs'
files = sorted(glob.glob(os.path.join(path, '**','*.tif'), recursive=True))
files = [fil for fil in files if not 'annotated_examples' in fil]

#%% visualize results
plt.close('all')


label = ['blurry','edge','interphase','prophase','unique']
pro_col = label.index('prophase')

pro_thresh = 0.7

completed = 0

num_to_show = 1

num_to_break = 8

for count, file in enumerate(files):
    is_bord=False
    
    # if count % 100 == 0:
    #     print(f'{count} of {len(files)}')
    
    # only use files that were not part of training set
    # if not int(datetime.datetime.fromtimestamp(os.path.getmtime(file)).strftime('%m%d'))>617:
    #     continue
        
    # dat = np.load(file, allow_pickle=True)
    
    # raw = dat['raw'].copy()
    # raw = cv2.copyMakeBorder(raw,143,143,143,143,0)
    # mask = dat['masks']
    # labels = dat['labels']
    # labels_dict = dat['labels_dict'].item()
    
    # stages = [labels_dict[lab] for lab in labels]
    
    # pro_id = np.where(np.array(stages) == 'prophase')[0]+1
    
    raw = plt.imread(file)
    
    raw_FCN = raw - raw.mean()
    raw_FCN /= raw.std()
    raw_FCN = raw_FCN[np.newaxis,...,np.newaxis]

    res = heatmodel.predict(raw_FCN)

    bord = 4
    res_new = tf.transpose(res, perm=[0, 3, 1, 2]).numpy()
    res = []
    for i in res_new.squeeze():
        res.append(cv2.copyMakeBorder(i, bord, bord, bord, 0, 0))
        
    res = np.array(res)
    res = res[np.newaxis,...]
  
    pro = res[:,pro_col,:,:].squeeze()
  
    if pro.max()>0.7:
        pass
    # elif len(pro_id) > 0:
    #     pass
        # print('missed one!', file)
    else:
        continue
    
    ms_h = 0#np.ceil(pro.shape[0]/2)
    ms_w = 0#np.ceil(pro.shape[0]/2)
  
    # pro = cv2.resize(pro,(514,514))
    # pro = cv2.copyMakeBorder(pro,bord,bord,bord,bord,0)
    top_left = cv2.minMaxLoc(pro)[-1]
    sh_y, sh_x = top_left
  
    if sh_y == 0+bord or sh_y == pro.shape[0]-bord-1 or sh_x == 0+bord or sh_x == pro.shape[0]-bord-1:
    # if sh_y == pro.shape[0]-1 or sh_x == pro.shape[0]-1:
      # continue
      sh_y_n = sh_y
      sh_x_n = sh_x
      is_bord = True
      
    else:
      # peak registration
        log_xm1_y = np.log(pro[sh_x - 1, sh_y])
        log_xp1_y = np.log(pro[sh_x + 1, sh_y])
        log_x_ym1 = np.log(pro[sh_x, sh_y - 1])
        log_x_yp1 = np.log(pro[sh_x, sh_y + 1])
        four_log_xy = 4 * np.log(pro[sh_x, sh_y])
    
        sh_x_n = (sh_x - ms_h + (log_xm1_y - log_xp1_y)/(2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = (sh_y - ms_w + (log_x_ym1 - log_x_yp1)/ (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
        
    factor = raw_FCN.shape[1]/(pro.shape[0]-bord)

    completed += 1
    # if completed == num_to_break:
    if True:
        fig2 = plt.figure()#figsize=(20,20))
    
        ax = fig2.add_subplot(num_to_show,6,6*(num_to_show-1)+1)
        ax.imshow(raw.squeeze(),cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('original')
        ax.scatter(factor*sh_y_n, factor*sh_x_n, c='b', label='MC')
        # max_loc = np.unravel_index(pro.argmax(), pro.shape)
        ax.scatter(factor*sh_y, factor*sh_x, c='r', label='OG')
        # ax.legend()
          
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((factor*sh_y_n-286/2, factor*sh_x_n-286/2), 286, 286, fill=False, edgecolor='g'))
        
        # res = tf.transpose(res, perm=[0, 3, 1, 2]).numpy()
        for num, i in enumerate(res.squeeze()):
          ax = fig2.add_subplot(num_to_show,6,6*6*(num_to_show-1)+2+num)
          ax.imshow(i, cmap='gray')
          ax.set_xticks([])
          ax.set_yticks([])
          ax.set_title(label[num]+' heatmap')
          if num == pro_col:
            # max_loc = np.unravel_index(i.argmax(), i.shape)
            ax.scatter(sh_y_n, sh_x_n, c='b', label='MC')
            ax.scatter(sh_y, sh_x, c='r', label='OG')
            # ax.legend()
        
        if is_bord: print('border')
        print((sh_y_n, sh_x_n))
        print(sh_y, sh_x)
        print()
        
        break
    

    
    if completed >= num_to_break:
        break
    































#%% visualize results - NEW
plt.close('all')


label = ['blurry','edge','interphase','prophase','unique']
pro_col = label.index('prophase')

pro_thresh = 0.7

completed = 0

num_to_show = 1

num_to_break = 8

for count, file in enumerate(files):
    is_bord=False
    
    if count % 100 == 0:
        print(f'{count} of {len(files)}')
    
    # only use files that were not part of training set
    if not int(datetime.datetime.fromtimestamp(os.path.getmtime(file)).strftime('%m%d'))>617:
        continue
        
    dat = np.load(file, allow_pickle=True)
    
    raw = dat['raw'].copy()
    raw = cv2.copyMakeBorder(raw,143,143,143,143,0)
    mask = dat['masks']
    labels = dat['labels']
    labels_dict = dat['labels_dict'].item()
    
    stages = [labels_dict[lab] for lab in labels]
    
    pro_id = np.where(np.array(stages) == 'prophase')[0]+1
    
    # raw = plt.imread(file)
    
    raw_FCN = raw - raw.mean()
    raw_FCN /= raw.std()
    raw_FCN = raw_FCN[np.newaxis,...,np.newaxis]

    # add border, batch sliding window
    raw_FCN_bigger = cv2.copyMakeBorder(raw_FCN.squeeze(), 143, 143, 143, 143, 0)
    raw_FCN_bigger = raw_FCN_bigger[np.newaxis, ..., np.newaxis]
    print(raw_FCN_bigger.shape)
    
    batch=[]
    for i1,i in enumerate(range(143,raw_FCN_bigger.shape[1]-143,16)):
        for j1,j in enumerate(range(143, raw_FCN_bigger.shape[1]-143, 16)):
            print((i1,j1))
            batch.append(raw_FCN_bigger[:1,i-143:i+143,j-143:j+143,:1])
    batch=np.concatenate(batch, axis=0)
    
    res = heatmodel.predict(batch)














    res = heatmodel.predict(raw_FCN)

    
    if pro.max()>0.7:
        pass
    elif len(pro_id) > 0:
        pass
        # print('missed one!', file)
    else:
        continue
    
    ms_h = 0#np.ceil(pro.shape[0]/2)
    ms_w = 0#np.ceil(pro.shape[0]/2)
  
    # pro = cv2.resize(pro,(514,514))
    # pro = cv2.copyMakeBorder(pro,bord,bord,bord,bord,0)
    top_left = cv2.minMaxLoc(pro)[-1]
    sh_y, sh_x = top_left
  
    if sh_y == 0+bord or sh_y == pro.shape[0]-bord-1 or sh_x == 0+bord or sh_x == pro.shape[0]-bord-1:
    # if sh_y == pro.shape[0]-1 or sh_x == pro.shape[0]-1:
      # continue
      sh_y_n = sh_y
      sh_x_n = sh_x
      is_bord = True
      
    else:
      # peak registration
        log_xm1_y = np.log(pro[sh_x - 1, sh_y])
        log_xp1_y = np.log(pro[sh_x + 1, sh_y])
        log_x_ym1 = np.log(pro[sh_x, sh_y - 1])
        log_x_yp1 = np.log(pro[sh_x, sh_y + 1])
        four_log_xy = 4 * np.log(pro[sh_x, sh_y])
    
        sh_x_n = (sh_x - ms_h + (log_xm1_y - log_xp1_y)/(2 * log_xm1_y - four_log_xy + 2 * log_xp1_y))
        sh_y_n = (sh_y - ms_w + (log_x_ym1 - log_x_yp1)/ (2 * log_x_ym1 - four_log_xy + 2 * log_x_yp1))
        
    factor = raw_FCN.shape[1]/(pro.shape[0]-bord)

    completed += 1
    # if completed == num_to_break:
    if True:
        fig2 = plt.figure()#figsize=(20,20))
    
        ax = fig2.add_subplot(num_to_show,6,6*(num_to_show-1)+1)
        ax.imshow(raw.squeeze(),cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('original')
        ax.scatter(factor*sh_y_n, factor*sh_x_n, c='b', label='MC')
        # max_loc = np.unravel_index(pro.argmax(), pro.shape)
        ax.scatter(factor*sh_y, factor*sh_x, c='r', label='OG')
        # ax.legend()
          
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((factor*sh_y_n-286/2, factor*sh_x_n-286/2), 286, 286, fill=False, edgecolor='g'))
        
        # res = tf.transpose(res, perm=[0, 3, 1, 2]).numpy()
        for num, i in enumerate(res.squeeze()):
          ax = fig2.add_subplot(num_to_show,6,6*6*(num_to_show-1)+2+num)
          ax.imshow(i, cmap='gray')
          ax.set_xticks([])
          ax.set_yticks([])
          ax.set_title(label[num]+' heatmap')
          if num == pro_col:
            # max_loc = np.unravel_index(i.argmax(), i.shape)
            ax.scatter(sh_y_n, sh_x_n, c='b', label='MC')
            ax.scatter(sh_y, sh_x, c='r', label='OG')
            # ax.legend()
        
        if is_bord: print('border')
        print((sh_y_n, sh_x_n))
        print(sh_y, sh_x)
        print()
        
        break
    

    
    if completed >= num_to_break:
        break
    













#%% pass sliding window in batches
raw_FCN_bigger = cv2.copyMakeBorder(raw_FCN.squeeze(), 143, 143, 143, 143, 0)
raw_FCN_bigger = raw_FCN_bigger[np.newaxis, ..., np.newaxis]
print(raw_FCN_bigger.shape)

batch=[]
for i1,i in enumerate(range(143,raw_FCN_bigger.shape[1]-143,16)):
    for j1,j in enumerate(range(143, raw_FCN_bigger.shape[1]-143, 16)):
        print((i1,j1))
        batch.append(raw_FCN_bigger[:1,i-143:i+143,j-143:j+143,:1])
batch=np.concatenate(batch, axis=0)

res = heatmodel.predict(batch)

#%% plot results
plt.subplot(1,6,1)
plt.imshow(raw, cmap='gray')
plt.title('raw')
plt.axis('off')

for i in range(5):
    plt.subplot(1,6,i+2)
    plt.imshow(res[:,:,:,i].reshape(50,50), cmap='gray')
    plt.title(label[i]+' heatmap')
    plt.axis('off')

#%%
for i in range(5):
    plt.subplot(1,6,i+1)
    plt.imshow(cv2.copyMakeBorder(np.reshape(res[:,:,:,i],(33,33)), 16, 16, 16, 16, 0), cmap='gray')
plt.subplot(1,6,6);plt.imshow(raw, cmap='gray')