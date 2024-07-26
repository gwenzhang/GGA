from collections import OrderedDict
from pathlib import Path

import mmcv
import numpy as np
import torch
from nuscenes.utils.geometry_utils import view_points

from mmdet3d.core.bbox import box_np_ops, points_cam2img
from .kitti_data_utils import WaymoInfoGatherer, get_kitti_image_info
from .nuscenes_converter import post_process_coords
# from segment_anything import SamPredictor, sam_model_registry
import cv2
# from tools.RANSAC import voxel_generator, get_ground_gpu
# from tools.visual_for_pts import pc_show_cluster, pc_show_points
# from collections import Counter
# from tools.visual_for_pts import show_points_and_bboxes_in_cluster
# from tools.some_function import calculate_simliar

from collections import Counter

from tools.data_converter.utils_gga import project_pts_on_img, calculate_ground, points_in_frustm_indices, region_grow

import os
from multiprocessing import Pool
import multiprocessing as mp
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis
from mmdet3d.core.bbox import box_np_ops as box_np_ops

kitti_categories = ('Pedestrian', 'Cyclist', 'Car')

def create_kitti_info_file(data_path,
                           pkl_prefix='kitti',
                           with_plane=False,
                           save_path=None,
                           relative_path=True):
    """Create info file of KITTI dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'kitti'.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    # for train set #
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        with_plane=with_plane,
        image_ids=train_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)
    
    # mp.set_start_method("spawn")
    pool = Pool(processes=60)
    for info in kitti_infos_train:
        pool.apply_async(_calculate_rga, (data_path, info, relative_path))
        # _calculate_rga(data_path, info, relative_path)
    pool.close()
    pool.join()

    # for debug
    # for info in kitti_infos_train:
    #     _calculate_rga(data_path, info, relative_path)

    train_split_path = './data/kitti/ImageSets/train.txt'
    with open(train_split_path, 'r') as file:
        content = file.read()
        content = content.split('\n')
    save_root_path = './data/kitti_GGA_split_file'
    GGA_train_infos = []
    for index, id in enumerate(content):
        cur_name =  'GGA_kitti_scene_{}.pkl'.format(int(id))
        cur_path =  os.path.join(save_root_path, cur_name)
        cur_info = mmcv.load(cur_path)
        GGA_train_infos.append(cur_info)
    filename = save_path / f'{pkl_prefix}_infos_train_GGA.pkl'
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(GGA_train_infos, filename)

    # for validation set #

    kitti_infos_val = get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        with_plane=with_plane,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_val, relative_path)

    pool = Pool(processes=60)
    for info in kitti_infos_val:
        pool.apply_async(_calculate_rga, (data_path, info, relative_path))
        # _calculate_rga(data_path, info, relative_path)
    pool.close()
    pool.join()

    val_split_path = './data/kitti/ImageSets/val.txt'
    with open(val_split_path, 'r') as file:
        content = file.read()
        content = content.split('\n')
    save_root_path = './data/kitti_GGA_split_file'
    GGA_val_infos = []
    for index, id in enumerate(content):
        cur_name =  'GGA_kitti_scene_{}.pkl'.format(int(id))
        cur_path =  os.path.join(save_root_path, cur_name)
        cur_info = mmcv.load(cur_path)
        GGA_val_infos.append(cur_info)
    filename = save_path / f'{pkl_prefix}_infos_val_GGA.pkl'
    print(f'Kitti info train file is saved to {filename}')
    mmcv.dump(GGA_val_infos, filename)
    # for trainval #
    filename = save_path / f'{pkl_prefix}_infos_trainval_GGA.pkl'
    print(f'Kitti info trainval file is saved to {filename}')
    mmcv.dump(GGA_train_infos + GGA_val_infos, filename)

    # for test set #
    kitti_infos_test = get_kitti_image_info(
        data_path,
        training=False,
        label_info=False,
        velodyne=True,
        calib=True,
        with_plane=False,
        image_ids=test_img_ids,
        relative_path=relative_path)
    filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    print(f'Kitti info test file is saved to {filename}')
    mmcv.dump(kitti_infos_test, filename)


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


"""
Data Preparation for General Geometry-aware Weakly Supervised 3D Object Detection
"""

def boundary_range(final_coords, img_size):

    _coords = np.array(final_coords)
    in_boundary_flags = ((_coords[0] > 3.0)
                      & (_coords[1] > 3.0)
                      & (_coords[2] < img_size[0] - 3.0)
                      & (_coords[3] < img_size[1] - 3.0))

    return in_boundary_flags

def _calculate_rga(data_path,
                    info,
                    relative_path,
                    remove_outside=True,
                    num_features=4):

    # if info['image']['image_idx'] == 11:
    #     print('FIND')
    # else:
    #     return
    save_path = './data/kitti_GGA_split_file'

    cur_name =  'GGA_kitti_scene_{}.pkl'.format(info['image']['image_idx'])
    filename = os.path.join(save_path, cur_name)    
    
    # if os.path.exists(filename):
    #     return

    pc_info = info['point_cloud']
    image_info = info['image']
    calib = info['calib']
    if relative_path:
        v_path = str(Path(data_path) / pc_info['velodyne_path'])
    else:
        v_path = pc_info['velodyne_path']
    points_v = np.fromfile(
        v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
    rect = calib['R0_rect']
    Trv2c = calib['Tr_velo_to_cam']
    P2 = calib['P2']
    # remove ground
    points_lidar = points_v[..., :3]
    points_shape = list(points_lidar.shape[0:-1])
    points_lidar = np.concatenate([points_lidar, np.ones(points_shape + [1])], axis=-1)
    points_cam = points_lidar @ (rect @ Trv2c).T
    mask_ground_all, _ = calculate_ground(points_cam[..., :3], 0.2)
    ground_plane_height = points_lidar[(1 - mask_ground_all).astype(np.bool_)][:, 2].mean()

    annos = info['annos']
    num_obj = len([n for n in annos['name'] if n != 'DontCare'])
    dims = annos['dimensions'][:num_obj]
    loc = annos['location'][:num_obj]
    rots = annos['rotation_y'][:num_obj]
    name = annos['name'][:num_obj]
    occluded = annos['occluded'][:num_obj]
    gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
    gt_boxes_lidar = box_np_ops.box_camera_to_lidar(gt_boxes_camera, rect, Trv2c)
    num_points_in_gt = annos['num_points_in_gt']

    img_shape = image_info['image_shape']
    img_size = tuple((img_shape[1]-1, img_shape[0]-1))
    img_boundary = np.array([0, 0, img_size[0], img_size[1]])
    mask2d = []
    box2d = []
    depth_mask = []
    bdry_masks = []
    mask_boundary = []
    sign_boundary = []

    """
    Generate the 2D boxes label
    depth_mask: True: All the object is in front of the camera.
    bdry_mask: True: This side of the 2D box is on the boundary of the image.
    box2d: The four edges of 2D bounding boxes. (xmin, ymin, xmax, ymax).
    mask2d: Whather the object is in the image.
    """

    for index, box3d in enumerate(gt_boxes_camera):
        box3d = box3d[np.newaxis, :]
        corners_3d = box_np_ops.center_to_corner_box3d(
            box3d[:, :3],
            box3d[:, 3:6],
            box3d[:, 6], [0.5, 1.0, 0.5],
            axis=1
        )
        corners_3d = corners_3d[0].T
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]
        camera_intrinsic = P2
        corner_coords = view_points(corners_3d, camera_intrinsic, True).T[:, :2].tolist()
        final_coords = post_process_coords(corner_coords, img_size)

        if final_coords is None:
            mask2d.append(False)
            depth_mask.append(False)
            mask_boundary.append(False)
            final_coords = np.array(-np.ones([1, 4]))
            box2d.append(final_coords)
            bdry_masks.append(np.ones(4).astype(np.bool_))
        else:
            mask2d.append(True)
            depth_mask.append(in_front.shape[0]==8)
            # sign_boundary.append(boundary_range(final_coords, img_size))
            final_coords = np.array(final_coords)[np.newaxis, :]
            box2d.append(final_coords)

            # for the boundary mask
            bdry_mask = final_coords[0] == img_boundary
            bdry_masks.append(bdry_mask)
            mask_boundary.append(np.all(~bdry_mask))

    gt_boxes_img = np.concatenate(box2d)
    annos['GGA_boxes_img'] = gt_boxes_img
    # annos['mask2d'] = np.array(mask2d)
    # annos['mask_depth'] = np.array(depth_mask)
    # annos['sign_boundary'] = np.array(mask_boundary)
    
    annos['GGA_mask_depth'] = np.array(depth_mask)
    annos['GGA_mask2d'] = np.array(mask2d)
    annos['GGA_mask_boundary'] = np.array(mask_boundary)
    annos['GGA_bdry_masks'] = np.stack(bdry_masks)

    img_path = image_info['image_path']
    img_path = data_path + '/' + img_path
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lidar2img = P2 @ rect @ Trv2c
    _, _, object_filter_all = project_pts_on_img(points_lidar, image, lidar2img)

    """
    Sort the objects by depth
    """
    isvalid = []
    medis = []
    box2d_pts_indices = []
    boxes_img = gt_boxes_img.copy()
    for index, bbox2d in enumerate(boxes_img):
        bpi = points_in_frustm_indices(points_lidar, rect, Trv2c, P2, bbox2d).squeeze()
        pts_cam = points_cam[bpi] # use the point depth in Camera coordinate
        if (bpi.sum() == 0) or (num_points_in_gt[index]==0):
            medi = 1000
            isvalid.append(False)
        else:
            medi = np.median(pts_cam[:, 2])
            isvalid.append(True)
        box2d_pts_indices.append(bpi)
        medis.append(medi)
    
    obj_ord = np.argsort(np.array(medis))
    
    """
    Generate In-Box-Points
    """
    points_cluster = []
    mask_object = np.ones((points_lidar.shape[0]))
    for element in np.nditer(obj_ord):
        if isvalid[element] == False:
            points_cluster.append(np.array([]))
            continue

        result = np.zeros((7, 2))
        count = 0
        mask_seg_list = []

        thresh_seg_max = 7
        if name[element] == 'Car':
            ratio = 0.96
        else:
            ratio = 0.85
        curr_box = boxes_img[element]
        for j in range(thresh_seg_max):
            thresh = (j + 1) * 0.1
            object_filter = points_in_frustm_indices(points_lidar, rect, Trv2c, P2, curr_box).squeeze()
            filter_z = points_cam[:, 2] > 0
            mask_search = mask_ground_all * object_filter_all * mask_object * filter_z
            mask_origin = mask_ground_all * object_filter * mask_object * filter_z
            mask_seg = region_grow(points_cam.copy(), mask_search, mask_origin, thresh, ratio)
            
            if mask_seg.sum() == 0:
                continue

            if j >= 1:
                mask_seg_old = mask_seg_list[-1]
                if mask_seg_old.sum() != (mask_seg * mask_seg_old).sum():
                    count += 1
            result[count, 0] = j
            result[count, 1] = mask_seg.sum()
            mask_seg_list.append(mask_seg)
        best_j = result[np.argmax(result[:, 1]), 0]
        
        try:
            mask_seg_best = mask_seg_list[int(best_j)]
            mask_object *= (1 - mask_seg_best)
            pc = points_lidar[mask_seg_best == 1].copy()
            if annos['GGA_mask_boundary'][element] == True:
                points_cluster.append(pc)
        except IndexError:
            points_cluster.append(np.array([]))
            continue

        if annos['GGA_mask_boundary'][element] == False:
            mask_origin_new = mask_seg_best
            mask_search_new = mask_ground_all
            thresh_new      = (best_j + 1) * 0.1
            mask_seg_for_truncate = region_grow(points_cam.copy(), mask_search_new, mask_origin_new, thresh_new, ratio=None)
            pc_truncate = points_lidar[mask_seg_for_truncate == 1].copy()
            if pc_truncate.shape[0] > 6000:
                points_cluster.append(pc)
            else:
                points_cluster.append(pc_truncate)
    
    point_cluster_ord = []
    obj_ord_ord = np.argsort(obj_ord)
    for pos in np.nditer(obj_ord_ord):
        point_cluster_ord.append(points_cluster[pos])

    """
    Generate the Initial Pseudo 3D boxes.
    """
    
    pseudo_center = []
    pseudo_rot = []
    pseudo_bboxes_3d = []
    points_mean_in_frustum = []
    points_num_in_frustum = []
    mask_valid = []
    for index, cur_clt in enumerate(point_cluster_ord):
        if cur_clt.shape[0] == 0:
            points_mean_in_frustum.append(np.array([[0., 0., 0.]]))
            points_num_in_frustum.append(0)
            pseudo_center.append(np.array([[0.0, 0.0]]))
            pseudo_rot.append(0.0)
            mask_valid.append(False)
            pseudo_bboxes_3d.append(np.zeros([1, 7]))
            continue

        # if name[index] == 'Cyclist' or name[index] == 'Pedestrian':
        #     #  Cyclist and pedestrian are often completion. 
        #     pesudo_center.append(cur_clt[:, :2].mean(axis=0, keepdims=True))
        #     pesudo_rot.append(0.0)
        #     points_mean_in_frustum.append(cur_clt[:, :3].mean(axis=0, keepdims=True))
        #     points_num_in_frustum.append(cur_clt.shape[0])
        #     sign_valid.append(True)
        # else:
        rot_list = np.arange(0, (np.pi/2.0-1e-6), np.pi/72.0).tolist()
        rot_dis = []
        rot_center = []
        rot_edge = []
        for rot_bin in rot_list:
            clt_r = rotation_3d_in_axis(cur_clt[..., :2], rot_bin, axis=2, clockwise=True)
            top_xmin = np.min(clt_r[..., 0])
            top_xmax = np.max(clt_r[..., 0])
            top_ymin = np.min(clt_r[..., 1])
            top_ymax = np.max(clt_r[..., 1])
            rot_dis.append((top_xmax - top_xmin) * (top_ymax - top_ymin))
            rot_center.append(np.array([(top_xmin + top_xmax)/2.0, (top_ymin + top_ymax)/2.0]))
            rot_edge.append(np.array([top_xmax - top_xmin, top_ymax - top_ymin]))

        rot_dis = np.array(rot_dis)
        rot_center = np.stack(rot_center)
        rot_edge = np.stack(rot_edge)
        sel_ind = np.argsort(rot_dis)[0]
        sel_rot = rot_list[sel_ind]
        sel_center = rot_center[sel_ind, None]
        sel_edge = rot_edge[sel_ind, None]
        sel_center_ori = rotation_3d_in_axis(sel_center, sel_rot, axis=2, clockwise=False)

        if sel_edge[:, 0] < sel_edge[:, 1]:
            sel_edge = sel_edge[:, ::-1]
            sel_rot = sel_rot + np.pi / 2.0

        clt_max_h = np.max(cur_clt[:, 2])
        pseudo_center_z_ground = np.array((clt_max_h + ground_plane_height) / 2.0)[np.newaxis]
        pseudo_dim_z_ground = np.array(clt_max_h - ground_plane_height)[np.newaxis]
        pseudo_bbox_3d = np.concatenate([sel_center_ori.squeeze(), pseudo_center_z_ground,
                                        sel_edge.squeeze(), pseudo_dim_z_ground,
                                        np.array(sel_rot)[np.newaxis]])[np.newaxis]

        # pesudo_center_z = np.array((clt_max_h + clt_min_h) / 2.0)[np.newaxis]

        pseudo_center.append(sel_center_ori)
        pseudo_rot.append(sel_rot)
        points_mean_in_frustum.append(cur_clt[:, :3].mean(axis=0, keepdims=True))
        points_num_in_frustum.append(cur_clt.shape[0])
        mask_valid.append(True)
        pseudo_bboxes_3d.append(pseudo_bbox_3d)

    # collect the attributes for GGA
    # mask
    annos['GGA_mask_valid'] = np.stack(mask_valid)
    # annos['GGA_boxes_img'] = gt_boxes_img
    # annos['GGA_mask_depth'] = np.array(depth_mask)
    # annos['GGA_mask2d'] = np.array(mask2d)
    # annos['GGA_mask_boundary'] = np.array(mask_boundary)
    # intial pseudo 3d box
    annos['GGA_in_box_points'] = point_cluster_ord
    annos['GGA_init_pseudo_label'] = np.concatenate(pseudo_bboxes_3d)
    annos['GGA_num_points_in_box2d'] = np.array(points_num_in_frustum)
    
    # TODO: add the empty for dontcare
    num_ignored = len(annos['dimensions']) - num_obj
    annos['GGA_boxes_img'] = np.concatenate((annos['GGA_boxes_img'], -np.zeros([num_ignored, 4])), axis=0)
    annos['GGA_mask2d'] = np.concatenate((annos['GGA_mask2d'], np.zeros([num_ignored]).astype(bool)))
    annos['GGA_mask_depth'] = np.concatenate((annos['GGA_mask_depth'], np.zeros([num_ignored]).astype(bool)))
    annos['GGA_mask_boundary'] = np.concatenate((annos['GGA_mask_boundary'], np.zeros([num_ignored]).astype(bool)))
    annos['GGA_mask_valid'] = np.concatenate((annos['GGA_mask_valid'], np.zeros([num_ignored]).astype(bool)))
    annos['GGA_num_points_in_box2d'] = np.concatenate((annos['GGA_num_points_in_box2d'], np.zeros([num_ignored])))
    annos['GGA_init_pseudo_label'] = np.concatenate((annos['GGA_init_pseudo_label'], np.zeros([num_ignored, 7])), axis=0)
    annos['GGA_bdry_masks'] = np.concatenate((annos['GGA_bdry_masks'], np.zeros([num_ignored, 4]).astype(bool)))
    ignore_list = [np.array([]) for _ in range(num_ignored)]
    annos['GGA_in_box_points'].extend(ignore_list)
    
    mmcv.dump(info, filename)
    print("Finish Processing Sample {}".format(info['image']['image_idx']))



def _create_reduced_point_cloud(data_path,
                                info_path,
                                save_path=None,
                                back=False,
                                num_features=4,
                                front_camera_id=2):
    """Create reduced point clouds for given info.

    Args:
        data_path (str): Path of original data.
        info_path (str): Path of data info.
        save_path (str, optional): Path to save reduced point cloud
            data. Default: None.
        back (bool, optional): Whether to flip the points to back.
            Default: False.
        num_features (int, optional): Number of point features. Default: 4.
        front_camera_id (int, optional): The referenced/front camera ID.
            Default: 2.
    """
    kitti_infos = mmcv.load(info_path)

    for info in mmcv.track_iter_progress(kitti_infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']

        v_path = pc_info['velodyne_path']
        v_path = Path(data_path) / v_path
        points_v = np.fromfile(
            str(v_path), dtype=np.float32,
            count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        if front_camera_id == 2:
            P2 = calib['P2']
        else:
            P2 = calib[f'P{str(front_camera_id)}']
        Trv2c = calib['Tr_velo_to_cam']
        # first remove z < 0 points
        # keep = points_v[:, -1] > 0
        # points_v = points_v[keep]
        # then remove outside.
        if back:
            points_v[:, 0] = -points_v[:, 0]
        points_v = box_np_ops.remove_outside_points(points_v, rect, Trv2c, P2,
                                                    image_info['image_shape'])
        if save_path is None:
            save_dir = v_path.parent.parent / (v_path.parent.stem + '_reduced')
            if not save_dir.exists():
                save_dir.mkdir()
            save_filename = save_dir / v_path.name
            # save_filename = str(v_path) + '_reduced'
            if back:
                save_filename += '_back'
        else:
            save_filename = str(Path(save_path) / v_path.name)
            if back:
                save_filename += '_back'
        with open(save_filename, 'w') as f:
            points_v.tofile(f)



def create_reduced_point_cloud(data_path,
                               pkl_prefix,
                               train_info_path=None,
                               val_info_path=None,
                               test_info_path=None,
                               save_path=None,
                               with_back=False):
    """Create reduced point clouds for training/validation/testing.

    Args:
        data_path (str): Path of original data.
        pkl_prefix (str): Prefix of info files.
        train_info_path (str, optional): Path of training set info.
            Default: None.
        val_info_path (str, optional): Path of validation set info.
            Default: None.
        test_info_path (str, optional): Path of test set info.
            Default: None.
        save_path (str, optional): Path to save reduced point cloud data.
            Default: None.
        with_back (bool, optional): Whether to flip the points to back.
            Default: False.
    """
    if train_info_path is None:
        train_info_path = Path(data_path) / f'{pkl_prefix}_infos_train_GGA.pkl'
    if val_info_path is None:
        val_info_path = Path(data_path) / f'{pkl_prefix}_infos_val_GGA.pkl'
    if test_info_path is None:
        test_info_path = Path(data_path) / f'{pkl_prefix}_infos_test.pkl'

    print('create reduced point cloud for training set')
    _create_reduced_point_cloud(data_path, train_info_path, save_path)
    print('create reduced point cloud for validation set')
    _create_reduced_point_cloud(data_path, val_info_path, save_path)
    print('create reduced point cloud for testing set')
    _create_reduced_point_cloud(data_path, test_info_path, save_path)
    if with_back:
        _create_reduced_point_cloud(
            data_path, train_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, val_info_path, save_path, back=True)
        _create_reduced_point_cloud(
            data_path, test_info_path, save_path, back=True)


def export_2d_annotation(root_path, info_path, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    kitti_infos = mmcv.load(info_path)
    cat2Ids = [
        dict(id=kitti_categories.index(cat_name), name=cat_name)
        for cat_name in kitti_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    from os import path as osp
    for info in mmcv.track_iter_progress(kitti_infos):
        coco_infos = get_2d_boxes(info, occluded=[0, 1, 2, 3], mono3d=mono3d)
        (height, width,
         _) = mmcv.imread(osp.join(root_path,
                                   info['image']['image_path'])).shape
        coco_2d_dict['images'].append(
            dict(
                file_name=info['image']['image_path'],
                id=info['image']['image_idx'],
                Tri2v=info['calib']['Tr_imu_to_velo'],
                Trv2c=info['calib']['Tr_velo_to_cam'],
                rect=info['calib']['R0_rect'],
                cam_intrinsic=info['calib']['P2'],
                width=width,
                height=height))
        for coco_info in coco_infos:
            if coco_info is None:
                continue
            # add an empty key for coco format
            coco_info['segmentation'] = []
            coco_info['id'] = coco_ann_id
            coco_2d_dict['annotations'].append(coco_info)
            coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(info, occluded, mono3d=True):
    """Get the 2D annotation records for a given info.

    Args:
        info: Information of the given sample data.
        occluded: Integer (0, 1, 2, 3) indicating occlusion state:
            0 = fully visible, 1 = partly occluded, 2 = largely occluded,
            3 = unknown, -1 = DontCare
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """
    # Get calibration information
    P2 = info['calib']['P2']

    repro_recs = []
    # if no annotations in info (test dataset), then return
    if 'annos' not in info:
        return repro_recs

    # Get all the annotation with the specified visibilties.
    ann_dicts = info['annos']
    mask = [(ocld in occluded) for ocld in ann_dicts['occluded']]
    for k in ann_dicts.keys():
        if isinstance(ann_dicts[k], list):
            ann_dicts[k] = [item for item, m in zip(ann_dicts[k], mask) if m]
        else:
            ann_dicts[k] = ann_dicts[k][mask]

    # convert dict of list to list of dict
    ann_recs = []
    for i in range(len(ann_dicts['occluded'])):
        ann_rec = {}
        for k in ann_dicts.keys():
            ann_rec[k] = ann_dicts[k][i]
        ann_recs.append(ann_rec)

    for ann_idx, ann_rec in enumerate(ann_recs):
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = \
            f"{info['image']['image_idx']}.{ann_idx}"
        ann_rec['sample_data_token'] = info['image']['image_idx']
        sample_data_token = info['image']['image_idx']

        loc = ann_rec['location'][np.newaxis, :]
        dim = ann_rec['dimensions'][np.newaxis, :]
        rot = ann_rec['rotation_y'][np.newaxis, np.newaxis]
        # transform the center from [0.5, 1.0, 0.5] to [0.5, 0.5, 0.5]
        dst = np.array([0.5, 0.5, 0.5])
        src = np.array([0.5, 1.0, 0.5])
        loc = loc + dim * (dst - src)
        offset = (info['calib']['P2'][0, 3] - info['calib']['P0'][0, 3]) \
            / info['calib']['P2'][0, 0]
        loc_3d = np.copy(loc)
        loc_3d[0, 0] += offset
        gt_bbox_3d = np.concatenate([loc, dim, rot], axis=1).astype(np.float32)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box_np_ops.center_to_corner_box3d(
            gt_bbox_3d[:, :3],
            gt_bbox_3d[:, 3:6],
            gt_bbox_3d[:, 6], [0.5, 0.5, 0.5],
            axis=1)
        corners_3d = corners_3d[0].T  # (1, 8, 3) -> (3, 8)
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        camera_intrinsic = P2
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token,
                                    info['image']['image_path'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            repro_rec['bbox_cam3d'] = np.concatenate(
                [loc_3d, dim, rot],
                axis=1).astype(np.float32).squeeze().tolist()
            repro_rec['velo_cam3d'] = -1  # no velocity in KITTI

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            repro_rec['attribute_name'] = -1  # no attribute in KITTI
            repro_rec['attribute_id'] = -1

        repro_recs.append(repro_rec)

    return repro_recs


def generate_record(ann_rec, x1, y1, x2, y2, sample_data_token, filename):
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, x_size, y_size of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    key_mapping = {
        'name': 'category_name',
        'num_points_in_gt': 'num_lidar_pts',
        'sample_annotation_token': 'sample_annotation_token',
        'sample_data_token': 'sample_data_token',
    }

    for key, value in ann_rec.items():
        if key in key_mapping.keys():
            repro_rec[key_mapping[key]] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in kitti_categories:
        return None
    cat_name = repro_rec['category_name']
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = kitti_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec


