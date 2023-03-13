# smartLLSM

## YOLO Pipeline

YOLO Pipeline Demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11YKXvFOAEAKNaunR0JqvFjqtqKtBgaHX?usp=sharing), download demo files [here](https://drive.google.com/drive/folders/1kGUbSdDD1E2v7K3tW5L7n0x-9O6WvIlo?usp=share_link).

Download trained YOLO weights [here](https://drive.google.com/file/d/1vQa8d6ZsTgpnkU2ERViVr2CnBNcf_QtS/view?usp=share_link).

### Installation

1. Install following packages, for example using a dedicated conda environment:

```
conda create -n smartLLSM_yolo_pipeline python spyder pandas opencv tqdm matplotlib seaborn scikit-image
conda activate smartLLSM_yolo_pipeline
```

2. Install correct PyTorch version (at least pytorch and torchvision packages). See [here](https://pytorch.org/get-started/locally/) for guidance.

It can be tricky to install compatible pytorch and torchvision versions with your current CUDA install. In our case, using a Linux machine running Ubuntu 18.04 with an NVIDIA GeForce RTX 2080 Ti GPU, CUDA version 11.1, this simple command worked best:

```
conda install pytorch torchvision
```

If you are running into local installation issues, you can always demo our pipeline in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11YKXvFOAEAKNaunR0JqvFjqtqKtBgaHX?usp=sharing)

### Use

```full_pipeline_YOLO.py``` watches a user-designated folder (```folder_to_watch```), waiting for tif files from microscope to populate folder before sending out for analysis. ```stage_of_interest``` dictate which cell phases YOLO should be on the lookout for (YOLO was trained to recognize the following phases: ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']). The pipeline stores results of its analysis to a csv (results_{DATE}.csv) within the watch folder. This csv contains cell-by-cell results with the following format (columns):
> [file name, slice (if using a tiff stack), x cell center coordinate, y cell center coordinate, YOLO confidence score, cell phase/label, x cell center coordinate relative to center of image, y cell center coordinate relative to center of image]

Other features of the pipeline:
* ```store_all``` - tallies the number of cells in each phase for each file, results stored in all_cells_found_{DATE}.csv within watch folder
* ```set_thresh``` - saves ALL cells detected by YOLO of a certain class and saves images to a folder. images are named based on their YOLO confidence score. This allows users to visually determine a threhsold they would like to use for detecting cells/initiating imaging.

## Annotator GUI

Annotator GUI Demo:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/15XbMJAP2yDOS5VtXgFkwGQYSIqarJQp0?usp=share_link), download demo files [here](https://drive.google.com/drive/folders/1kGUbSdDD1E2v7K3tW5L7n0x-9O6WvIlo?usp=share_link).

### Installation

1. Install following packages, for example using a dedicated conda environment:

```
conda create -n smartLLSM_annotator tensorflow-estimator tensorflow-gpu h5py spyder opencv scipy
conda activate smartLLSM_annotator
```

### Use

```annotation_gui.py``` provides an efficient script for manually annotating cells. Users can set their own dictionary of keys to label cell phases (```labels_dict```). In addition, a binary classifier can be used to filter out tiles that do not include mitotic cells (```nn_filter```). This bootstraps the annotation process by only prompting users to label tiles that include rarer mitotic cells. Our trained binary classifier weights can be found [here](https://drive.google.com/file/d/1hqYLRnyHW1QH0B_a7PDga4dZmvhS4jgH/view?usp=share_link). The annotator is best run in the Spyder IDE but can also be run via Terminal or Jupyter Notebook.

In order to generate tiles for annotation, raw tifs were first processed through [Cellpose](https://github.com/MouseLand/cellpose) to segment out individual cells.

## Notes
For those interested in training custom YOLO models from annotated cell data, a helper script (```yolo_create_dataset.py```) is provided for converting annotated cells to YOLO training data format (for more info, see [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)).

Manually annotated cell data available upon request.

## Developers
* Jimmy Tabet, UNC
