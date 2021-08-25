#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 13:35:56 2021

@author: jimmytabet
"""

#%% imports
import glob, cv2, os, time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence TensorFlow error message about not being optimized...

#%% load model

nn_path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/FCN_models/2021-08-25/anaphase_blank_blurry_edge_interphase_metaphase_prometaphase_prophase_telophase.h5'
label = nn_path.split('.')[-2].split('/')[-1].split('_')

def get_conv(input_shape=(200, 200, 1), filename=None):

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

    tf.keras.layers.Conv2D(64, kernel_size=6, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Conv2D(len(label), kernel_size=1, activation='softmax'),
    # tf.keras.layers.Conv2D(len(np.unique(y_train)), kernel_size=1),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.GlobalMaxPooling2D(),
    # tf.keras.layers.Activation('softmax')
    ])

    if filename:
        model.load_weights(filename)
    return model

heatmodel = get_conv(input_shape=(None, None, 1), filename=nn_path)

# generate viz of model architecture
# tf.keras.utils.plot_model(heatmodel, show_shapes=True, show_layer_names=False, rankdir='TB')

#%% load test annotation results
files = list(np.load('/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/datasets/FCN_test_files_0824.npy'))

#%% all annotated files - for dist hist
# path = '/home/nel/NEL-LAB Dropbox/NEL/Datasets/smart_micro/Cellpose_tiles/annotation_results'
# files = sorted(glob.glob(os.path.join(path,'**','*.npz'), recursive=True))

#%% loop
train_shape = 200
half_shape = train_shape//2
pro_col = label.index('prophase')

gt_pro = []
FCN_max = []
center_dist = []
reg_dists = []

for count, file in enumerate(files):
    
    if count % 100 == 0:
        print(f'{count} of {len(files)}')
    
    # load data
    with np.load(file, allow_pickle=True) as dat:
        raw = dat['raw']
        mask = dat['masks']
        stages = dat['labels']
    
    # find gt prophase cells
    pro_id = np.where(np.array(stages) == 'prophase')[0]+1
    gt_pro.append(len(pro_id))
    
    # preprocess raw image
    raw_FCN = raw - raw.mean()
    raw_FCN /= raw.std()
    raw_FCN = raw_FCN[np.newaxis,...,np.newaxis]
    
    # add border
    raw_FCN_bigger = cv2.copyMakeBorder(raw_FCN.squeeze(), half_shape, half_shape, half_shape, half_shape, 0)
    raw_FCN_bigger = raw_FCN_bigger[np.newaxis, ..., np.newaxis]
    
    # # batch sliding window
    # batch=[]
    # for i1,i in enumerate(range(100,raw_FCN_bigger.shape[1]-100,20)):
    #     for j1,j in enumerate(range(100, raw_FCN_bigger.shape[1]-100, 20)):
    #         # print((i1,j1))
    #         batch.append(raw_FCN_bigger[:1,i-100:i+100,j-100:j+100,:1])
    # batch=np.concatenate(batch, axis=0)
    
    # forward pass
    res = heatmodel.predict(raw_FCN_bigger).squeeze()

    pro = res[:,:,pro_col]
    
    FCN_pro_max = pro.max()
    FCN_max.append(FCN_pro_max)
    
    # skip if no prophase cells in tile (distance = inf)
    if len(pro_id) == 0:
        center_dist.append(np.inf)
        continue

    # exponential interpolation
    ms_h = 0
    ms_w = 0
    
    top_left = cv2.minMaxLoc(pro)[-1]
    sh_y, sh_x = top_left
    
    if sh_y == 0 or sh_y == pro.shape[0] or sh_x == 0 or sh_x == pro.shape[0]:
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
    
    # heatmap to raw image factor
    factor = raw.shape[1]/pro.shape[1]
    FCN_x = factor*sh_y_n
    FCN_y = factor*sh_x_n
    
    # calc gt prophase centroid(s)
    dists = []
    reg_dist = []
    corr_dist = []
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
        
        reg_d = np.linalg.norm([cX-factor*sh_y, cY-factor*sh_x])
        reg_dist.append(reg_d)
        
    # keep shorter distance
    center_dist.append(min(dists))
    reg_dists.append(min(reg_dist))

#%% dist hist
reg = np.array(reg_dists)
corr = np.array(center_dist)[~np.isinf(center_dist)]
plt.hist(reg, bins = 70, range = (0,100), label = f'regular, mean={reg.mean().round(1)}')
plt.hist(corr, bins = 70, range = (0,100), label = f'corrected, mean={corr.mean().round(1)}')
plt.legend()

#%% arrays
gt_pro = np.array(gt_pro)
FCN_max = np.array(FCN_max)
center_dist = np.array(center_dist)

#%% confusion
dist_thresh = 100
TPR_all = []
FPR_all = []
for pro_thresh in np.linspace(0,1,1001):
    
    FCN_pro = FCN_max > pro_thresh
    
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
    # print('tp:', tp)
    # print('tn:', tn)
    # print('fp:', fp)
    # print('fn:', fn)
    # print('OOR:', oor)
    print(fp/(fp+tn), tp/(tp+fn))
    
    # if fp/(fp+tn) < 10**-3:
    #     print(pro_thresh)
    #     break
    
    TPR_all.append(tp/(tp+fn))
    FPR_all.append(fp/(fp+tn))
    
#%% short confusion
print('       actual')
print('        +  -')
print('pred +',tp,fp)
print('     -',fn,tn)  
 
#%% ROC curve
# plt.plot((FPR_all), (TPR_all))
# plt.xlim([0,1])
# plt.ylim([0,1])
# plt.plot([0,1],[0,1], ls='--', c='k')
plt.plot(np.log(FPR_all), np.log(TPR_all))
# plt.xlim([-6.5, .1])
# plt.ylim([-.78, .03])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title(f'Prophase ROC, AUC = {metrics.auc(FPR_all, TPR_all).round(3)}')

#%% raw tif files
path = '/home/nel/Desktop/Smart Micro/ShannonEntropy_2Dimgs'
files = sorted(glob.glob(os.path.join(path, '**','*.tif'), recursive=True))
files = [fil for fil in files if not 'annotated_examples' in fil]

#%% visualize results
# clear results on run
plt.close('all')

# model info
train_shape = 200
half_shape = train_shape//2
pro_col = label.index('prophase')
pro_thresh = 0.99

# plot bool
plot = True

# init number of files to run/break on
completed = 0
num_to_break = 15

for count, file in enumerate(files):
    is_bord=False
    
    if count % 100 == 0:
        print(f'{count} of {len(files)}')
    
    # load data
    # raw = plt.imread(file)
    with np.load(file, allow_pickle=True) as dat:
        raw = dat['raw']
        stages = dat['labels']
        
    pro_id = np.where(np.array(stages) == 'prophase')[0]+1
    
    # start pipeline timing
    start = time.time()
    
    # preprocess raw image
    raw_FCN = raw - raw.mean()
    raw_FCN /= raw.std()
    raw_FCN = raw_FCN[np.newaxis,...,np.newaxis]

    # add border
    raw_FCN_bigger = cv2.copyMakeBorder(raw_FCN.squeeze(), half_shape, half_shape, half_shape, half_shape, 0)
    raw_FCN_bigger = raw_FCN_bigger[np.newaxis, ..., np.newaxis]
    
    # # batch sliding window
    # batch=[]
    # for i1,i in enumerate(range(100,raw_FCN_bigger.shape[1]-100,20)):
    #     for j1,j in enumerate(range(100, raw_FCN_bigger.shape[1]-100, 20)):
    #         # print((i1,j1))
    #         batch.append(raw_FCN_bigger[:1,i-100:i+100,j-100:j+100,:1])
    # batch=np.concatenate(batch, axis=0)
    
    # forward pass
    res = heatmodel.predict(raw_FCN_bigger).squeeze()

    # isolate prophase
    pro = res[:,:,pro_col]
    print('max prophase score:', pro.max())
    
    # plot if above thresh
    if pro.max()>pro_thresh:
        pass
    elif len(pro_id) > 0:
        pass
        # print('missed one!', file)
    else:
        continue
        
    # exponential interpolation
    ms_h = 0#np.ceil(pro.shape[0]/2)
    ms_w = 0#np.ceil(pro.shape[0]/2)
  
    top_left = cv2.minMaxLoc(pro)[-1]
    sh_y, sh_x = top_left
  
    if sh_y == 0 or sh_y == pro.shape[0] or sh_x == 0 or sh_x == pro.shape[0]:
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
    
    # print pipeline time
    print()
    print('pipeline time:', time.time()-start)
    
    # print centroid info
    if is_bord:
        print('HEATMAP MAX ON BORDER')
    print('corrected centroid:', (sh_y_n, sh_x_n))
    print('original centroid:', sh_y, sh_x)
            
    print()
    
    # heatmap to raw image factor
    factor = raw.shape[1]/pro.shape[1]
    
    if plot:
        fig = plt.figure()
        
        # plot raw
        ax = fig.add_subplot(4,3,1)
        ax.imshow(raw.squeeze(),cmap='gray',vmin=np.percentile(raw,1),vmax=np.percentile(raw,99))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('original:'+str(pro.max()))
        ax.scatter(factor*sh_y_n, factor*sh_x_n, c='b', label='EI')
        ax.scatter(factor*sh_y, factor*sh_x, c='r', label='OG')
        ax.legend()
        
        # add rectangle for viz
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((factor*sh_y_n-half_shape, factor*sh_x_n-half_shape),
                               train_shape, train_shape, fill=False, edgecolor='g'))
        
        # plot heatmaps
        res = tf.transpose(res, perm=[2, 0, 1]).numpy()
        for num, i in enumerate(res.squeeze()):
          ax = fig.add_subplot(4,3,2+num)
          ax.imshow(i, cmap='gray')
          
          ax.set_xticks([])
          ax.set_yticks([])
          ax.set_title(label[num]+' heatmap:'+str(i[np.unravel_index(np.argmax(pro), i.shape)]))
          if num == pro_col:
            ax.scatter(sh_y_n, sh_x_n, c='b', label='EI')
            ax.scatter(sh_y, sh_x, c='r', label='OG')
            ax.legend()
    
    # break if enough files have been run
    completed += 1
    if completed >= num_to_break:
        break