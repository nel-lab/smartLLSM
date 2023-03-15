# smartLLSM
Developed and tested on Linux Ubuntu 18.04. Tested on Mac OS BigSur (see installation notes below for help with Mac install).

GPU recommended, but not required.

Installation should only take a few minutes (<10). Consider using [mamba](https://mamba.readthedocs.io/en/latest/), a faster drop-in conda replacement, if conda environment creation is taking too long.

Demo our software in Google Colab (see links below). Demo runtime is around three minutes. The annotator demo can be stopped at any time by pressing 'q' when prompted to annotate a cell.

Demo files (.zip) and model weights (.pt/.hdf5) can be downloaded [here](https://www.dropbox.com/s/2xwfox4e8mg0jhv/smartLLSM%20Demo%20Data.zip?dl=0&file_subpath=%2FsmartLLSM+Demo+Data).

## YOLO Pipeline

YOLO Pipeline Demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11YKXvFOAEAKNaunR0JqvFjqtqKtBgaHX?usp=sharing), download demo files [here](https://www.dropbox.com/s/2xwfox4e8mg0jhv/smartLLSM%20Demo%20Data.zip?dl=0&file_subpath=%2FsmartLLSM+Demo+Data%2FYOLO_demo.zip).

Download trained YOLO weights [here](https://www.dropbox.com/s/2xwfox4e8mg0jhv/smartLLSM%20Demo%20Data.zip?dl=0&file_subpath=%2FsmartLLSM+Demo+Data%2FYOLO_weights.pt).

### Installation

1. Install following packages, for example using a dedicated conda environment:

```
conda create -n smartLLSM_yolo_pipeline python spyder pandas opencv tqdm matplotlib seaborn scikit-image
conda activate smartLLSM_yolo_pipeline
```

2. Install correct PyTorch version (at least pytorch and torchvision packages). See [here](https://pytorch.org/get-started/locally/) for guidance.

It can be tricky to install compatible pytorch and torchvision versions with your current CUDA install. In our case, using a Linux machine running Ubuntu 18.04 with an NVIDIA GeForce RTX 2080 Ti GPU, CUDA version 11.1 (as well as on MacOS BigSur), this simple command worked best:

```
conda install pytorch torchvision
```

> Note for Mac OS installation: 
> 
> 1. if you are getting an opencv/cv2 error, try:
>
> ```conda remove opencv```
>
> ```pip install opencv-python```
>
> 2. if you are getting an OMP error ("...multiple copies of the OpenMP runtime...") OR the kernel keeps restarting in Spyder, try importing torch FIRST (ie move ```import torch``` to line 10) and run via Terminal (```python full_pipeline_YOLO.py```)
> 
> 3. if you are getting the following message:
>
> ```AttributeError: 'Upsample' object has no attribute 'recompute_scale_factor'```
>
> try adding the following lines of code AFTER setting up the model (line 48):
>
> ```
> for m in nn_model.modules():
>     if isinstance(m, torch.nn.Upsample):
>         m.recompute_scale_factor = None
> ```

If you are running into local installation issues, you can always demo our pipeline in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11YKXvFOAEAKNaunR0JqvFjqtqKtBgaHX?usp=sharing)

### Use

```full_pipeline_YOLO.py``` watches a user-designated folder (```folder_to_watch```), waiting for tif files from microscope to populate folder before sending out for analysis. ```stage_of_interest``` dictate which cell phases YOLO should be on the lookout for (YOLO was trained to recognize the following phases: ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']). The pipeline stores results of its analysis to a csv (results_{DATE}.csv) within the watch folder. This csv contains cell-by-cell results with the following format (columns):
> [file name, slice (if using a tiff stack), x cell center coordinate, y cell center coordinate, YOLO confidence score, cell phase/label, x cell center coordinate relative to center of image, y cell center coordinate relative to center of image]

Other features of the pipeline:
* ```store_all``` - tallies the number of cells in each phase for each file, results stored in all_cells_found_{DATE}.csv within watch folder
* ```set_thresh``` - saves ALL cells detected by YOLO of a certain class and saves images to a folder. images are named based on their YOLO confidence score. This allows users to visually determine a threhsold they would like to use for detecting cells/initiating imaging.

## Annotator GUI

Annotator GUI Demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15XbMJAP2yDOS5VtXgFkwGQYSIqarJQp0?usp=share_link), download demo files [here](https://www.dropbox.com/s/2xwfox4e8mg0jhv/smartLLSM%20Demo%20Data.zip?dl=0&file_subpath=%2FsmartLLSM+Demo+Data%2Fannotation_demo.zip).

### Installation

1. Install following packages, for example using a dedicated conda environment (edit tensorflow install depending on if you are using a GPU):

```
conda create -n smartLLSM_annotator tensorflow-estimator tensorflow[-gpu] h5py spyder opencv scipy
conda activate smartLLSM_annotator
```

> Note for Mac OS installation: 
> 
> 1. if you are getting an opencv/cv2 error, try:
>
> ```conda remove opencv```
>
> ```pip install opencv-python```

### Use

```annotation_gui.py``` provides an efficient script for manually annotating cells. Users can set their own dictionary of keys to label cell phases (```labels_dict```). There are also keys to display the label dictionary (```show_label```), return to a previously labeled cell in the same tile (```back_key```), and exit the GUI (```exit_key```).

In addition, a simple classifier can be used to filter out tiles that do not include mitotic cells (```nn_filter```). This bootstraps the annotation process by only prompting users to label tiles that include rarer mitotic cells. Our trained classifier weights can be found [here](https://www.dropbox.com/s/2xwfox4e8mg0jhv/smartLLSM%20Demo%20Data.zip?dl=0&file_subpath=%2FsmartLLSM+Demo+Data%2Fannotator_filter.hdf5). The annotator is best run in the Spyder IDE but can also be run via Terminal or Jupyter Notebook. 

In order to generate tiles for annotation, raw tifs were first processed through [Cellpose](https://github.com/MouseLand/cellpose) to segment out individual cells, then saved as npy files (file.npy). Final annotated tifs/tiles are saved as an npz (file_annotated.npz) with the following variables:

[

'raw': raw image

'masks': masked cells from Cellpose (0 = background, 1 = cell 1, 2 = cell 2, etc)

'labels': list of annotated cell labels ordered per Cellpose masks

'labels_dict': dictionary of labels used (ie. 0: prophase, 1: interphase, 2: prophase, etc)

'labels_XX': n/a, used in case where multiple users annotate the same tile before an expert/majority/etc decides on final labels

'confirmed': n/a, used in case where multiple users annotate the same tile before an expert/majority/etc decides on final labels

]

## Notes
For those interested in training custom YOLO models from annotated cell data, a helper script (```yolo_create_dataset.py```) is provided for converting annotated cells to YOLO training data format (for more info, see [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)).

Our complete, manually annotated cell dataset used to train/evaluate YOLO is available upon request. This complete dataset (along with the provided trained model weights) is required to reproduce our model evaluation results as stated in our paper utilizing the [val.py script](https://github.com/ultralytics/yolov5/blob/master/val.py) from the YOLO repository.

## Developers
* Jimmy Tabet, UNC
