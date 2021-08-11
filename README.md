# smart-micro
Repo for smart microscopy project

# Pipeline Install instructions

For GPU performance with cellpose: pytorch is needed 

For GPU performance with tensorflow: cudatoolkit and cuda dnn are needed 

Installation on windows has been a bit troublesome.  For example making sure that the python environment is reading the correct cuda runtime dlls (e.g. cudnn64_7.dll) has been a problem.  You may need to search the computer and remove other versions of cudnn64_7.dll

```
conda create -n sm_gpu tensorflow-estimator=2.0.0 tensorflow-gpu=2.0.0 scikit-image matplotlib opencv spyder
conda activate  sm_gpu
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install cellpose
conda install cudatoolkit=10.1
conda install cudnn=7.6.5
```

#
Currently cells are segmented using cellpose.  Then classified.

# Running
```
python full_pipeline_cellpose.py
```
This will start the classifier pipeline, which will watch the "watch folder" for files to process.  Once a file has been processed it will be moved to a "completed" folder, and a new row will appear in the .csv spreadsheet.

Parameters are specified in `full_pipeline_cellpose.py` 

