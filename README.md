# smartLLSM

# YOLO Pipeline Installation Instructions

```
conda create -n sm_yolo_pipeline python spyder pandas opencv tqdm matplotlib seaborn scikit-image
conda activate sm_yolo_pipeline
```
Install correct PyTorch version from [here](https://pytorch.org/get-started/locally/), for example:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

```full_pipeline_YOLO.py``` watches a user-designated folder (```folder_to_watch```), waiting for tif files from microscope to populate folder before sending out for analysis. ```stage_of_interest``` dictate which cell phases YOLO should be on the lookout for (YOLO was trained to recognize the following phases: ['anaphase', 'blurry', 'interphase', 'metaphase', 'prometaphase', 'prophase', 'telophase']). Our trained YOLO weights can be found [here](DROPBOX!!!). The pipeline stores results of its analysis to a csv (results_{DATE}.csv) within the watch folder. This csv contains cell-by-cell results with the following format (columns):
> [file name, slice (if using a tiff stack), x cell center coordinate, y cell center coordinate, YOLO confidence score, cell phase/label, x cell center coordinate relative to center of image, y cell center coordinate relative to center of image]

Other features of the pipeline:
* ```store_all``` - tallies the number of cells in each phase for each file, results stored in all_cells_found_{DATE}.csv within watch folder
* ```set_thresh``` - saves ALL cells detected by YOLO of a certain class and saves images to a folder. images are named based on their YOLO confidence score. This allows users to visually determine a threhsold they would like to use for detecting cells/initiating imaging.

A demo of the YOLO pipeline can be found on [Google Colab](https://colab.research.google.com/drive/11YKXvFOAEAKNaunR0JqvFjqtqKtBgaHX?usp=sharing). Download demo tifs [here](DROPBOX!!!)

# Annotator GUI Installation instructions

```conda create -n sm_annotator tensorflow-estimator tensorflow-gpu h5py spyder opencv scipy```

```annotation_gui.py``` provides an efficient script for manually annotating cells. Users can set their own dictionary of keys to label cell phases (```labels_dict```). In addition, a binary classifier can be used to filter out tiles that do not include mitotic cells (```nn_filter```). This bootstraps the annotation process by only prompting users to label tiles that include rarer mitotic cells. Our trained binary classifier weights can be found [here](DROPBOX).

Demo tiles that are ready for annotation can be found [here](DROPBOX!!!). In order to generate tiles for annotation, tifs were first processed through [Cellpose](https://github.com/MouseLand/cellpose) to segment out individual cells.

# Notes
For those interested in training custom YOLO models from annotated cell data, a helper script (```yolo_create_dataset.py```) is provided for converting annotated cells to YOLO training data format (for more info, see [here](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#12-create-labels-1)

Manually annotated cell data available upon request
