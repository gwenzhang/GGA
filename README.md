# General Geometry-aware Weakly Supervised 3D Object Detection

This repo is the official implementation of ECCV24 paper [General Geometry-aware (GGA) Weakly Supervised 3D Object Detection](https://arxiv.org/pdf/2407.13748). Our GGA exhibits promising generalization capabilites, allowing it to be easily extend to various novel scenarios and classes. GGA achieves state-of-the-art performance on 2D bbox-supervised Monocular 3D object Detection. GGA is built on the codebase of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/dev-1.0).

## 🔥News
-[24-07-04] Our GGA is accepted by ECCV'24 🎉🎉🎉, if you find it helpful, please give it a star.  
-[24-07-18] Code of KITTI is released.

## 👀Overview  
- [📘 TODO](https://github.com/gwenzhang/GGA#📘todo)  
- [🚀 Main Results](https://github.com/gwenzhang/GGA#🏆Mmin-results)  
- [🛠️ Quick Start](https://github.com/gwenzhang/GGA#🚀quick-start)  
- [📘 Citation](https://github.com/gwenzhang/GGA#citation)  
- [🚀 Acknowledgments](https://github.com/gwenzhang/GGA#acknowledgments)  

## 📘TODO  
- [x] Release the code of KITTI.  
- [x] Release the [arxiv](https://arxiv.org/pdf/2407.13748) version.  
- [ ] Release the pseudo labels.  
- [ ] Release more detail results.  

#### Notice     
We are currently updating this repository due to a code reorganization. There may be some issues. Please feel free to report any problems in the issues section.   

### 🏆Main Results

#### Outdoor Monocular 3D Object Detection (on KITTI test)  

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="3">AP<sub>BEV</th>
      <th colspan="3">AP<sub>3D</th>
    </tr>
    <tr>
      <th>Model</th>
      <th>Easy</th>
      <th>Mod.</th>
      <th>Hard</th>
      <th>Easy</th>
      <th>Mod.</th>
      <th>Hard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PGD+GGA</td>
      <td>17.42</td>
      <td>10.21</td>
      <td>8.09</td>
      <td>10.42</td>
      <td>6.08</td>
      <td>4.65</td>
    </tr>
  </tbody>
</table>

#### Outdoor Monocular 3D Object Detection (on KITTI validation)  

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="3">AP<sub>BEV</th>
      <th colspan="3">AP<sub>3D</th>
    </tr>
    <tr>
      <th>Model</th>
      <th>Easy</th>
      <th>Mod.</th>
      <th>Hard</th>
      <th>Easy</th>
      <th>Mod.</th>
      <th>Hard</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MonoDETR+GGA</td>
      <td>30.07</td>
      <td>21.49</td>
      <td>18.23</td>
      <td>21.18</td>
      <td>14.96</td>
      <td>10.89</td>
    </tr>
  </tbody>
</table>

#### Indoor Point Cloud 3D Object Detection (on SUN-RGBD)  
<table style="width:100%; text-align:center;">
  <thead>
    <tr>
      <th>Model</th>
      <th>bathtub</th>
      <th>bed</th>
      <th>bkshelf</th>
      <th>chair</th>
      <th>desk</th>
      <th>dresser</th>
      <th>nstand</th>
      <th>sofa</th>
      <th>table</th>
      <th>toilet</th>
      <th>mAP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>FCAF3D+GGA</td>
      <td>55.4</td>
      <td>69.9</td>
      <td>22.4</td>
      <td>59.1</td>
      <td>22.5</td>
      <td>31.3</td>
      <td>59.3</td>
      <td>58.9</td>
      <td>34.8</td>
      <td>71.4</td>
      <td>48.5</td>
    </tr>
  </tbody>
</table>

## 🚀Quick Start  
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

### Training GGA  
```
./tools/dist_train.sh configs/gga/gga_kitti_config.py 8  
```

### Generate Pseudo 3D Labels  
```
./tools/dist_pseudo.sh configs/gga/gga_kitti_matching_config.py {checkpoints} 8  --eval mAP  
python create_data_gga_retrain_mono.py kitti --root_path ./data/kitti --out_dir ./data/kitti  
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
│   │   ├── kitti_infos_trainval_GGA_mono3d.coco.json
│   │   ├── kitti_infos_test_mono3d.coco.json
│   ├── kitti_GGA_split_file
```

### Retraining  
 ```
./tools/dist_train.sh configs/gga/gga_pgd.py 8  
```

### Testing (Generate submission files)
 ```
./tools/dist_test.sh configs/gga/gga_pgd.py {checkpoint_dir} 8  --format-only --eval-options 'pklfile_prefix=./gga_results' 'submission_prefix=./gga_results' 
```

## Citation
Please consider citing our work as follows if it is helpful.
```
@article{zhang2024general,
  title={General Geometry-aware Weakly Supervised 3D Object Detection},
  author={Zhang, Guowen and Fan, Junsong and Chen, Liyi and Zhang, Zhaoxiang and Lei, Zhen and Zhang, Lei},
  booktitle={European Conference on Computer Vision},
  organization={Springer}
  year={2024}
}
```

## Acknowledgments
GGA is based on MMDetection3D.  
We also thank the FGR, MonoDETR, PDG, CenterPoint and FCAF3D authors for their efforts.




