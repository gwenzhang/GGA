# General Geometry-aware Weakly Supervised 3D Object Detection

This repo is the official implementation of ECCV24 paper [General Geometry-aware (GGA) Weakly Supervised 3D Object Detection](). Our GGA exhibits promising generalization capabilites, allowing it to be easily extend to various novel scenarios and classes. GGA achieves state-of-the-art performance on 2D bbox-supervised Monocular 3D object Detection. GGA is built on the codebase of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/dev-1.0).

## News
-[24-07-04] Our GGA is accepted by ECCV'24 🎉🎉🎉, if you find it helpful, please give it a star.  
-[24-07-18] Code of KITTI is released.

## Overview  


## TODO  
- [x] Release the code of KITTI.  
- [ ] Release the arXiv version.  
- [ ] Release the pseudo labels.  
- [ ] Release more detail results.  


### Main Results  

## Quick Start  
### Installation  
```shell
conda create --name gga python=3.8 -y  
conda activate gga  
conda install pytorch torchvision -c pytorch  
pip install openmim  
mim install mmcv-full  
mim install mmdet  
mim install mmsegmentation  
git clone https://github.com/gwenzhang/GGA.git  
cd GGA  
pip install -e .  
```


### Data Preparation  
#### KITTI  
```
mmdetection3d
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
```
* Generate the data infos by running the following command (it may take several hours):  
```python 
cd GGA  
python ./tools/create_data_gga.py kitti --root_path ./data/kitti --out_dir ./data/kitti  
# Create dataset info file, and lidar pseudo database
```
* The format of the generated data is as follows:  
```
mmdetection3d
├── data
│   ├── kitti
│   │   ├── ImageSets
│   │   ├── testing
│   │   ├── training
│   │   ├── kitti_gt_database_GGA
│   │   ├── kitti_infos_train_GGA.pkl
│   │   ├── kitti_infos_val_GGA.pkl
│   │   ├── kitti_infos_trainval_GGA.pkl
│   │   ├── kitti_infos_test.pkl
│   │   ├── kitti_dbinfos_train_GGA.pkl
│   ├── kitti_GGA_split_file
```

### Training (Generate Pseudo 3D Labels)  



