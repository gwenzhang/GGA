import pickle
import mmcv


# path1 = '/home/guowen_zhang/GGA/mmdetection3d/data/kitti/kitti_infos_train_GGA.pkl'

# path2 = '/home/guowen_zhang/GGA/mmdetection3d/data/kitti/kitti_infos_val_GGA.pkl'

# # with open(path1, 'rb') as file:
# #     kitti_infos_train = pickle.load(file)
# kitti_infos_train = mmcv.load(path1)

# # with open(path2, 'rb') as file:
# #     kitti_infos_val = pickle.load(file)
# kitti_infos_val = mmcv.load(path2)

# kitti_infos_trainval = kitti_infos_train + kitti_infos_val

# mmcv.dump(kitti_infos_trainval, '/home/guowen_zhang/GGA/mmdetection3d/data/kitti/kitti_infos_trainval_GGA.pkl')

# debug = 1



path1 = '/home/guowen_zhang/GGA/mmdetection3d/data/kitti/kitti_infos_trainval_GGA_mono3d.coco.json'

path2 = '/home/guowen_zhang/GGA/mmdetection3d/data/kitti/kitti_infos_trainval_GGA_pseudo_mono3d.coco.json'


kitti_infos_train = mmcv.load(path1)
kitti_infos_val = mmcv.load(path2)

debug = 1