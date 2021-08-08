# -*- coding: utf-8 -*-
"""SmartMicroFullPipelineWatchFolder.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19OGIyhmSPBb_yok93ScdLmM3upfho9hO

# IMPORTS
"""

#!pip install -q cellpose
#!pip install -q tensorflow

import os, glob, platform, time, cv2, csv
import numpy as np
import skimage.io

import torch
from cellpose import models
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # silence PyTorch warnings about changed behavior...

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # silence Tensorflow error message about not being optimized...
# release GPU for Pytorch
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# OLD VERSION:
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.compat.v1.Session(config=config)

import matplotlib.pyplot as plt

# disable Cellpose's ~very~ verbose outputs
import logging.config
logging.config.dictConfig({'version': 1, 'disable_existing_loggers': True})

print('Python version:', platform.python_version())
print('Tensorflow version:', tf.__version__)
print('-----------------------------------------------')

"""# USER INPUT, SETUP"""

### USER SETUP ###
folder_to_watch = '/home/nel-lab/Desktop/Jimmy/Smart Micro/full_pipeline/watch_folder'

current_date = time.strftime('%m_%d_%y')
results_csv = os.path.join(folder_to_watch, f'results_{current_date}.csv')

path_to_nn = '/home/nel-lab/Desktop/Jimmy/Smart Micro/full_pipeline/prophase_classifier_5_10.hdf5'

# number of files to analyze at a time
BATCH_SIZE = 1
# delay in seconds between checking folder
delay = 5

# visualize results
visualize_results = True
### USER SETUP ###


### INIT PIPELINE ###
# create completed folders if not already
os.makedirs(os.path.join(folder_to_watch, 'completed', current_date), exist_ok=True)

# set folder for finished images
finished_folder = os.path.join(folder_to_watch, 'completed', current_date)

# CELLPOSE MODEL
use_GPU = models.use_gpu()
cellpose_model = models.Cellpose(gpu=use_GPU, model_type='cyto')
channels = [0,0]

# NEURAL NETWORK MODEL
def nn_temp(a,b):return True
nn_model = tf.keras.models.load_model(path_to_nn, custom_objects={'f1_metric':nn_temp})
output_classes = np.array(nn_model.name.split('__temp__'))
input_size = nn_model.input_shape[1:3]
half_size = nn_model.input_shape[1]//2
filter_class = 'prophase'
filter_col = int(np.where(filter_class == output_classes)[0])
thresh = 0.7
### INIT PIPELINE ###

### RUN_PIPELINE FUNCTION ###
def run_pipeline(files, cellpose_model, channels,
                 nn_model, output_classes, input_size, half_size, filter_col, thresh,
                 results_csv, finished_folder, viz=False):
    
    # BUILD IMAGES LIST
    '''
    # open tiff stack using PIL if 'AttributeError: is_indexed' --> ~5x SLOWER than skimage.io
    # create array for tif_stack function
    from PIL import Image
    def tif_stack(tif_path):
        dataset = Image.open(tif_path)
        slices = dataset.n_frames
        h,w = dataset.size
        tif_array = np.zeros([slices, h, w])
        for i in range(slices):
           dataset.seek(i)
           tif_array[i,:,:] = np.array(dataset)
    
        return tif_array
    '''
   
    imgs = []
    file_names = []
    file_indexes = []
    for f in files:
        temp_file = skimage.io.imread(f)
        # if tiff stack, add each image indivdually
        if temp_file.ndim > 2:
            for idx, temp_im in enumerate(temp_file):
                imgs.append(temp_im)
                file_names.append(f)
                file_indexes.append(idx)
        # if single tiff image, add
        else:
            imgs.append(temp_file)
            file_names.append(f)
            # blank index
            file_indexes.append('')

    print(f'analyzing {len(files)} file(s), {len(imgs)} image(s)')

    # RUN CELLPOSE
    # imgs = [skimage.io.imread(f) for f in files]
    masks = cellpose_model.eval(imgs, diameter=150, flow_threshold=None, channels=channels)[0]    

    # release GPU for tensorflow
    torch.cuda.empty_cache()
    
    print('Cellpose segmentation complete')
    
    # SEGMENT OUT CELLS FOR ANALYSIS
    X_all = []
    centers = []
    file_ref = []

    for file_num, (raw, mask) in enumerate(zip(imgs, masks)):
        # get number of cells in tile
        num_masks = mask.max()

        # make border for edge cells
        raw_border = cv2.copyMakeBorder(raw, input_size[0], input_size[0], input_size[0], input_size[0],
                                        borderType = cv2.BORDER_CONSTANT, value = int(raw[mask==0].mean()))
    
        # loop through each mask in tile to get input cell image
        for mask_id in range(1,num_masks+1):
        
            file_ref.append(file_num)

            # get moments for cell to calculate center of mass
            M = cv2.moments(1*(mask==mask_id), binaryImage=True)

            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            centers.append((cX, cY))

            # bounding box
            r1 = cY-half_size + input_size[0]
            r2 = cY+half_size + input_size[0]
            c1 = cX-half_size + input_size[0]
            c2 = cX+half_size + input_size[0]
      
            # save cropped cell
            X_all.append(raw_border[r1:r2,c1:c2])

    X_all = np.array(X_all, dtype = float)
    
    # PREDICT CELL STAGE
    # normalize image
    X_all -= X_all.mean(axis=(1,2))[:,None,None]
    X_all /= X_all.std(axis=(1,2))[:,None,None]
    
    # reshape for neural netowrk
    # X_all = X_all.reshape(-1,*nn_model.input_shape[1:]) # reshape is slower based on tests...
    X_all.resize((X_all.shape[0],*nn_model.input_shape[1:]))
    
    # predict on batch of cells
    preds = nn_model.predict(X_all)
    
    print('Tensorflow classification complete')
    
    # # print output/save to file/etc... FOR ANY CLASS
    # cell_stage = output_classes[np.argmax(preds, axis=1)]
    # for idx, stage in enumerate(cell_stage):
    #   print(f'{np.char.upper(stage.squeeze())} CELL FOUND HERE: {centers[idx]}')
    
    # make masks for prophase cells
    pro_mask = preds[:,filter_col] > thresh
    file_mask = np.array(file_ref)[pro_mask]
    # file name mask
    f_name_mask = np.array(file_names)[file_mask]
    file_name_mask = [os.path.basename(f) for f in f_name_mask]
    # file index mask
    file_index_mask = np.array(file_indexes)[file_mask]
    # cell centroid mask
    cell_centroid_mask = np.array(centers)[pro_mask]
    
    all_info = np.column_stack([file_name_mask, file_index_mask, cell_centroid_mask])
        
    # write prophase centroid to results csv
    with open(results_csv, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(all_info)
  
    # view results
    if viz and all_info.size:
        n = int(np.ceil(np.sqrt(len(f_name_mask))))
        fig = plt.figure()
        count = 1
        for fn, idx, (cx, cy) in zip(f_name_mask, file_index_mask, cell_centroid_mask):
            im = skimage.io.imread(fn)
            if idx:
                im = im[int(idx)]
            ax = fig.add_subplot(n,n,count)
            ax.set_title(f'{os.path.basename(fn)} {idx}')
            ax.axis('off')
            ax.imshow(im.squeeze(), cmap='gray')
            ax.scatter(cx,cy, c='r', marker='*')
            count += 1
        # pause to show image while pipeline runs
        plt.pause(0.1)

    # move files to completed folder
    [os.replace(fil, os.path.join(folder_to_watch, 'completed', current_date, os.path.basename(fil))) for fil in files]
### RUN_PIPELINE FUNCTION ###

"""# WATCH FOLDER"""

while True:
    # look for files
    files_all = sorted(glob.glob(os.path.join(folder_to_watch,'**','*.tif'), recursive=True))
    files_all = [file for file in files_all if not 'completed' in file]
    
    files_analyzed = files_all[:BATCH_SIZE]
    
    if len(files_all) >= BATCH_SIZE:
        print('----------------------------')
        start = time.time()
        run_pipeline(files_analyzed, cellpose_model, channels,
                     nn_model, output_classes, input_size, half_size, filter_col, thresh,
                     results_csv, finished_folder, viz=visualize_results)
        print(f'pipeline timetime: {time.time()-start}')
        print('----------------------------')
              
    else:
        print('waiting for files...')
        time.sleep(delay)