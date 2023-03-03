import os, glob
import skimage.io
import random
from cellpose import models

drive_dir = '/home/nel/Desktop/Smart Micro/ShannonEntropy_2Dimgs'
files_all = sorted(glob.glob(os.path.join(drive_dir,'**/*.tif'), recursive = True))

#%%
model = models.Cellpose(gpu=True, model_type='cyto', )
channels = [0,0]

#%%
%%timeit
for _ in range(100):
    imgs = skimage.io.imread(random.choice(files_all))
    masks = model.eval(imgs, diameter=150, flow_threshold=None, channels=channels)[0]