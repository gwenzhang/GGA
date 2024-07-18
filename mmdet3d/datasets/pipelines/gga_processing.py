import numpy as np
from mmcv.utils import build_from_cfg

from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from ..builder import OBJECTSAMPLERS, PIPELINES
from scipy.spatial.distance import cdist
import torch
from functools import reduce

from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
import mmcv
import os
import warnings
from scipy.spatial.distance import pdist, squareform
import copy

@PIPELINES.register_module()
class ObjectSample_GGA(object):
    """Sample GT objects to the data.

    Args:
        db_sampler (dict): Config dict of the database sampler.
        sample_2d (bool): Whether to also paste 2D image patch to the images
            This should be true when applying multi-modality cut-and-paste.
            Defaults to False.
        use_ground_plane (bool): Whether to use gound plane to adjust the
            3D labels.
    """

    def __init__(self, min_distance, db_sampler, sample_2d=False, use_ground_plane=False):
        self.min_distance = min_distance
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler_GGA'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.use_ground_plane = use_ground_plane

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        """Remove the points in the sampled bounding boxes.

        Args:
            points (:obj:`BasePoints`): Input point cloud array.
            boxes (np.ndarray): Sampled ground truth boxes.

        Returns:
            np.ndarray: Points with those in the boxes removed.
        """
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points = points[np.logical_not(masks.any(-1))]
        return points

    @staticmethod
    def remove_points_in_boxes_v2(points, pts_mean, min_distance):
        '''
        We donnot have 3d gt bbox. We use points_mean_in_mask to remove points.
        '''

        dist_mat = cdist(points.tensor.numpy()[:, :2], pts_mean[:, :2])
        masks = dist_mat < min_distance
        points = points[np.logical_not(masks.any(-1))]

        return points


    def __call__(self, input_dict):
        """Call function to sample ground truth objects to the data.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after object sampling augmentation,
                'points', 'gt_bboxes_3d', 'gt_labels_3d' keys are updated
                in the result dict.
        """
        #  Ground Truth  #
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        # gt_bboxes_2d = input_dict['gt_bboxes']

        # annos = input_dict['ann_info'].copy()
        GGA_boxes_img = input_dict['GGA_boxes_img']
        GGA_lidar2img = input_dict['GGA_lidar2img']
        GGA_init_pseudo_labels = input_dict['GGA_init_pseudo_labels']
        GGA_in_box_points = input_dict['GGA_in_box_points']
        GGA_mask_valid = input_dict['GGA_mask_valid']
        GGA_bdry_masks = input_dict['GGA_bdry_masks']
        GGA_difficulty = input_dict['GGA_difficulty']
        GGA_num_points_in_box2d = input_dict['GGA_num_points_in_box2d']

        # sign2d = annos['sign2d']
        # we need to filter the sign_mask = False, because they donnt have point_mean_in_mask
        # points_mean_in_mask = points_mean_in_mask[sign_mask][:, :3]

        if self.use_ground_plane and 'plane' in input_dict['ann_info']:
            ground_plane = input_dict['ann_info']['plane']
            input_dict['plane'] = ground_plane
        else:
            ground_plane = None
        # change to float for blending operation
        points = input_dict['points']
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']
            # Assume for now 3D & 2D bboxes are the same
            sampled_dict = self.db_sampler.sample_all(
                # gt_bboxes_3d.tensor.numpy(),
                GGA_init_pseudo_labels,
                gt_labels_3d,
                gt_bboxes_2d=gt_bboxes_2d,
                img=img)
        else:
            sampled_dict = self.db_sampler.sample_all(
                GGA_init_pseudo_labels,
                gt_labels_3d,
                GGA_mask_valid,
                self.min_distance,
                ground_plane=ground_plane)

        if sampled_dict is not None:
            sampled_points = sampled_dict['points']
            sampled_gt_bboxes_3d = sampled_dict['gt_bbox_3ds']
            sampled_gt_labels = sampled_dict['gt_labels_3d']
            sampled_GGA_box_imgs = sampled_dict['GGA_box_imgs']
            sampled_GGA_lidar2imgs = sampled_dict['GGA_lidar2imgs']
            sampled_GGA_init_pseudo_labels = sampled_dict['GGA_init_pseudo_labels']
            sampled_GGA_mask_valids = sampled_dict['GGA_mask_valids']
            sampled_GGA_bdry_masks = sampled_dict['GGA_bdry_masks']
            sampled_GGA_difficulties = sampled_dict['GGA_difficulties']
            sampled_GGA_num_points_in_box2ds = sampled_dict['GGA_num_points_in_box2ds']
            sampled_GGA_in_box_points = sampled_dict['GGA_in_box_points']

            # sampled_gt_bboxes_img = sampled_dict['gt_bboxes_img']
            # sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            # sampled_points = sampled_dict['points']
            # sampled_gt_labels = sampled_dict['gt_labels_3d']
            # sampled_lidar2img = sampled_dict['gt_lidar2img']
            # sampled_pts_mean = sampled_dict['pts_mean']
            # sampled_pts_num_in_frustum = sampled_dict['gt_pts_num_in_frustum']
            # sampled_sign_valid = sampled_dict['gt_sign_valid']
            # sampled_sign_depth = sampled_dict['gt_sign_depth']
            # sampled_sign_boundary = sampled_dict['gt_sign_boundary']
            # sampled_difficutly = sampled_dict['difficulty']
            # sampled_pesudo_center = sampled_dict['pesudo_center']
            # sampled_pesudo_rot = sampled_dict['pesudo_rot']
            # sampled_num_points_in_gt = sampled_dict['num_points_in_gt']
            # ### For List ###
            # sampled_points_cluster = sampled_dict['points_cluster']

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))
            GGA_boxes_img = np.concatenate([GGA_boxes_img, sampled_GGA_box_imgs])
            GGA_lidar2img = np.concatenate([GGA_lidar2img, sampled_GGA_lidar2imgs])
            GGA_init_pseudo_labels = np.concatenate([GGA_init_pseudo_labels, sampled_GGA_init_pseudo_labels])
            GGA_mask_valid = np.concatenate([GGA_mask_valid, sampled_GGA_mask_valids])
            GGA_bdry_masks = np.concatenate([GGA_bdry_masks, sampled_GGA_bdry_masks])
            GGA_difficulty = np.concatenate([GGA_difficulty, sampled_GGA_difficulties])
            GGA_num_points_in_box2d = np.concatenate([GGA_num_points_in_box2d, sampled_GGA_num_points_in_box2ds])

            GGA_in_box_points += sampled_GGA_in_box_points


            # points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            points = self.remove_points_in_boxes_v2(points, sampled_GGA_init_pseudo_labels[:, :2], self.min_distance)

            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']
        
        input_dict['points'] = points
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['GGA_boxes_img'] = GGA_boxes_img
        input_dict['GGA_lidar2img'] = GGA_lidar2img
        input_dict['GGA_init_pseudo_labels'] = GGA_init_pseudo_labels
        input_dict['GGA_mask_valid'] = GGA_mask_valid
        input_dict['GGA_bdry_masks'] = GGA_bdry_masks
        input_dict['GGA_difficulty'] = GGA_difficulty
        input_dict['GGA_num_points_in_box2d'] = GGA_num_points_in_box2d
        input_dict['GGA_in_box_points'] = GGA_in_box_points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str
    

@PIPELINES.register_module()
class ObjectRangeFilter_GGA(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range, num_points_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)
        self.num_points_range = num_points_range

    def _bev_range(self, object_bev, bev_range):

        in_range_flags = ((object_bev[:, 0] > bev_range[0])
                          & (object_bev[:, 1] > bev_range[1])
                          & (object_bev[:, 0] < bev_range[2])
                          & (object_bev[:, 1] < bev_range[3]))
        return in_range_flags

    def _boundary_range(self, object_range, img_shape):

        in_range_flags = ((object_range[:, 0] > img_shape[0]))

        return in_range_flags

    def _num_range(self, object_num):

        in_num_flags = object_num > self.num_points_range

        return in_num_flags

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d'
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        # bev_range = self.pcd_range[[0, 1, 3, 4]]
        # GT information
        gt_labels_3d = input_dict['gt_labels_3d']
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        # GGA
        GGA_mask_valid = input_dict['GGA_mask_valid']
        GGA_boxes_img = input_dict['GGA_boxes_img']
        GGA_lidar2img = input_dict['GGA_lidar2img']
        GGA_init_pseudo_labels = input_dict['GGA_init_pseudo_labels']
        GGA_bdry_masks = input_dict['GGA_bdry_masks']
        GGA_difficulty = input_dict['GGA_difficulty']
        GGA_num_points_in_box2d = input_dict['GGA_num_points_in_box2d']
        GGA_in_box_points = input_dict['GGA_in_box_points']

        in_range_flags = self._bev_range(GGA_init_pseudo_labels[:, :2], bev_range)
        in_difficulty_flag = GGA_difficulty > -1

        # num_points_in_gt = input_dict['num_points_in_gt']
        # numpts_flag = num_points_in_gt > 15

        num_mask = self._num_range(GGA_num_points_in_box2d)
        mask = GGA_mask_valid & num_mask & in_difficulty_flag & in_range_flags

        # GT information
        gt_labels_3d = gt_labels_3d[mask]
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        # GGA information
        GGA_boxes_img = GGA_boxes_img[mask]
        GGA_bdry_masks = GGA_bdry_masks[mask]
        GGA_lidar2img = GGA_lidar2img[mask]
        GGA_init_pseudo_labels = GGA_init_pseudo_labels[mask]
        GGA_in_box_points = [cluster for cluster, include in zip(GGA_in_box_points, mask.tolist()) if include]

        input_dict['gt_labels_3d'] = gt_labels_3d
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['GGA_boxes_img'] = GGA_boxes_img
        input_dict['GGA_bdry_masks'] = GGA_bdry_masks
        input_dict['GGA_lidar2img'] = GGA_lidar2img
        input_dict['GGA_init_pseudo_labels'] = GGA_init_pseudo_labels
        input_dict['GGA_in_box_points'] = GGA_in_box_points

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str
    

@PIPELINES.register_module()
class DefaultFormatBundle_GGA(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img",
    "proposals", "gt_bboxes", "gt_labels", "gt_masks" and "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    - gt_masks: (1)to tensor, (2)to DataContainer (cpu_only=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __init__(self, ):
        return

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = [img.transpose(2, 0, 1) for img in results['img']]
                imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
                results['img'] = DC(to_tensor(imgs), stack=False)
            else:
                img = np.ascontiguousarray(results['img'].transpose(2, 0, 1))
                results['img'] = DC(to_tensor(img), stack=False)
        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))
        if 'gt_bboxes_3d' in results:
            if isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = DC(
                    results['gt_bboxes_3d'], cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DC(
                    to_tensor(results['gt_bboxes_3d']))

        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None, ...]), stack=True)

        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class DefaultFormatBundle3D_GGA(DefaultFormatBundle_GGA):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(DefaultFormatBundle3D_GGA, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        if self.with_gt:
            # Clean GT bboxes in the final
            if 'gt_bboxes_3d_mask' in results:
                gt_bboxes_3d_mask = results['gt_bboxes_3d_mask']
                results['gt_bboxes_3d'] = results['gt_bboxes_3d'][
                    gt_bboxes_3d_mask]
                if 'gt_names_3d' in results:
                    results['gt_names_3d'] = results['gt_names_3d'][
                        gt_bboxes_3d_mask]
                if 'centers2d' in results:
                    results['centers2d'] = results['centers2d'][
                        gt_bboxes_3d_mask]
                if 'depths' in results:
                    results['depths'] = results['depths'][gt_bboxes_3d_mask]
            if 'gt_bboxes_mask' in results:
                gt_bboxes_mask = results['gt_bboxes_mask']
                if 'gt_bboxes' in results:
                    results['gt_bboxes'] = results['gt_bboxes'][gt_bboxes_mask]
                results['gt_names'] = results['gt_names'][gt_bboxes_mask]
            if self.with_label:
                if 'gt_names' in results and len(results['gt_names']) == 0:
                    results['gt_labels'] = np.array([], dtype=np.int64)
                    results['attr_labels'] = np.array([], dtype=np.int64)
                elif 'gt_names' in results and isinstance(
                        results['gt_names'][0], list):
                    # gt_labels might be a list of list in multi-view setting
                    results['gt_labels'] = [
                        np.array([self.class_names.index(n) for n in res],
                                 dtype=np.int64) for res in results['gt_names']
                    ]
                elif 'gt_names' in results:
                    results['gt_labels'] = np.array([
                        self.class_names.index(n) for n in results['gt_names']
                    ],
                                                    dtype=np.int64)
                # we still assume one pipeline for one frame LiDAR
                # thus, the 3D name is list[string]
                if 'gt_names_3d' in results:
                    results['gt_labels_3d'] = np.array([
                        self.class_names.index(n)
                        for n in results['gt_names_3d']
                    ],
                                                       dtype=np.int64)
                # Our GGA
                if 'GGA_boxes_img' in results:
                    results['GGA_boxes_img'] = DC(torch.from_numpy(results['GGA_boxes_img']))
                if 'GGA_lidar2img' in results:
                    results['GGA_lidar2img'] = DC(torch.from_numpy(results['GGA_lidar2img']))
                if 'GGA_bdry_masks' in results:
                    results['GGA_bdry_masks'] = DC(torch.from_numpy(results['GGA_bdry_masks']))
                if 'GGA_init_pseudo_labels' in results:
                    results['GGA_init_pseudo_labels'] = DC(torch.from_numpy(results['GGA_init_pseudo_labels']))
                if 'GGA_in_box_points' in results:
                    if isinstance(results['GGA_in_box_points'], list):
                        results['GGA_in_box_points'] = DC([to_tensor(res) for res in results['GGA_in_box_points']])

        results = super(DefaultFormatBundle3D_GGA, self).__call__(results)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
    


@PIPELINES.register_module()
class Collect3D_GGA(object):
    """Collect data from the loader relevant to the specific task.

    This is usually the last stage of the data loader pipeline. Typically keys
    is set to some subset of "img", "proposals", "gt_bboxes",
    "gt_bboxes_ignore", "gt_labels", and/or "gt_masks".

    The "img_meta" item is always populated.  The contents of the "img_meta"
    dictionary depends on "meta_keys". By default this includes:

        - 'img_shape': shape of the image input to the network as a tuple
            (h, w, c).  Note that images may be zero padded on the
            bottom/right if the batch tensor is larger than this shape.
        - 'scale_factor': a float indicating the preprocessing scale
        - 'flip': a boolean indicating if image flip transform was used
        - 'filename': path to the image file
        - 'ori_shape': original shape of the image as a tuple (h, w, c)
        - 'pad_shape': image shape after padding
        - 'lidar2img': transform from lidar to image
        - 'depth2img': transform from depth to image
        - 'cam2img': transform from camera to image
        - 'pcd_horizontal_flip': a boolean indicating if point cloud is
            flipped horizontally
        - 'pcd_vertical_flip': a boolean indicating if point cloud is
            flipped vertically
        - 'box_mode_3d': 3D box mode
        - 'box_type_3d': 3D box type
        - 'img_norm_cfg': a dict of normalization information:
            - mean: per channel mean subtraction
            - std: per channel std divisor
            - to_rgb: bool indicating if bgr was converted to rgb
        - 'pcd_trans': point cloud transformations
        - 'sample_idx': sample index
        - 'pcd_scale_factor': point cloud scale factor
        - 'pcd_rotation': rotation applied to point cloud
        - 'pts_filename': path to point cloud file.

    Args:
        keys (Sequence[str]): Keys of results to be collected in ``data``.
        meta_keys (Sequence[str], optional): Meta keys to be converted to
            ``mmcv.DataContainer`` and collected in ``data[img_metas]``.
            Default: ('filename', 'ori_shape', 'img_shape', 'lidar2img',
            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
            'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
            'box_type_3d', 'img_norm_cfg', 'pcd_trans',
            'sample_idx', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename')
    """
    def __init__(
        self,
        keys,
        meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                   'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 'flip',
                   'pcd_horizontal_flip', 'pcd_vertical_flip', 'box_mode_3d',
                   'box_type_3d', 'img_norm_cfg', 'pcd_trans', 'sample_idx',
                   'pcd_scale_factor', 'pcd_rotation', 'pcd_rotation_angle',
                   'pts_filename', 'transformation_3d_flow', 'trans_mat',
                   'affine_aug',
                   'calib'
                   )):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]

        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'
    



class BatchSampler:
    """Class for sampling specific category of ground truths.

    Args:
        sample_list (list[dict]): List of samples.
        name (str, optional): The category of samples. Default: None.
        epoch (int, optional): Sampling epoch. Default: None.
        shuffle (bool, optional): Whether to shuffle indices. Default: False.
        drop_reminder (bool, optional): Drop reminder. Default: False.
    """

    def __init__(self,
                 sampled_list,
                 name=None,
                 epoch=None,
                 shuffle=True,
                 drop_reminder=False):
        self._sampled_list = sampled_list
        self._indices = np.arange(len(sampled_list))
        if shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0
        self._example_num = len(sampled_list)
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num):
        """Sample specific number of ground truths and return indices.

        Args:
            num (int): Sampled number.

        Returns:
            list[int]: Indices of sampled ground truths.
        """
        if self._idx + num >= self._example_num:
            ret = self._indices[self._idx:].copy()
            self._reset()
        else:
            ret = self._indices[self._idx:self._idx + num]
            self._idx += num
        return ret

    def _reset(self):
        """Reset the index of batchsampler to zero."""
        assert self._name is not None
        # print("reset", self._name)
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num):
        """Sample specific number of ground truths.

        Args:
            num (int): Sampled number.

        Returns:
            list[dict]: Sampled ground truths.
        """
        indices = self._sample(num)
        return [self._sampled_list[i] for i in indices]



@OBJECTSAMPLERS.register_module()
class DataBaseSampler_GGA(object):
    """Class for sampling data from the ground truth database.

    Args:
        info_path (str): Path of groundtruth database info.
        data_root (str): Path of groundtruth database.
        rate (float): Rate of actual sampled over maximum sampled number.
        prepare (dict): Name of preparation functions and the input value.
        sample_groups (dict): Sampled classes and numbers.
        classes (list[str], optional): List of classes. Default: None.
        bbox_code_size (int, optional): The number of bbox dimensions.
            Default: None.
        points_loader(dict, optional): Config of points loader. Default:
            dict(type='LoadPointsFromFile', load_dim=4, use_dim=[0,1,2,3])
    """

    def __init__(self,
                 info_path,
                 data_root,
                 rate,
                 prepare,
                 sample_groups,
                 classes=None,
                 bbox_code_size=None,
                 points_loader=dict(
                     type='LoadPointsFromFile',
                     coord_type='LIDAR',
                     load_dim=4,
                     use_dim=[0, 1, 2, 3]),
                 file_client_args=dict(backend='disk')):
        super().__init__()
        self.data_root = data_root
        self.info_path = info_path
        self.rate = rate
        self.prepare = prepare
        self.classes = classes
        self.cat2label = {name: i for i, name in enumerate(classes)}
        self.label2cat = {i: name for i, name in enumerate(classes)}
        self.points_loader = mmcv.build_from_cfg(points_loader, PIPELINES)
        self.file_client = mmcv.FileClient(**file_client_args)

        # load data base infos
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(info_path) as local_path:
                # loading data from a file-like object needs file format
                db_infos = mmcv.load(open(local_path, 'rb'), file_format='pkl')
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {info_path} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            db_infos = mmcv.load(info_path)

        # filter database infos
        from mmdet3d.utils import get_root_logger
        logger = get_root_logger()
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')
        for prep_func, val in prepare.items():
            db_infos = getattr(self, prep_func)(db_infos, val)
        logger.info('After filter database:')
        for k, v in db_infos.items():
            logger.info(f'load {len(v)} {k} database infos')

        self.db_infos = db_infos

        self.bbox_code_size = bbox_code_size
        if bbox_code_size is not None:
            for k, info_cls in self.db_infos.items():
                for info in info_cls:
                    info['box3d_lidar'] = info['box3d_lidar'][:self.
                                                              bbox_code_size]

        # load sample groups
        # TODO: more elegant way to load sample groups
        self.sample_groups = []
        for name, num in sample_groups.items():
            self.sample_groups.append({name: int(num)})

        self.group_db_infos = self.db_infos  # just use db_infos

        self.sample_classes = []
        self.sample_max_nums = []
        for group_info in self.sample_groups:
            self.sample_classes += list(group_info.keys())
            self.sample_max_nums += list(group_info.values())

        self.sampler_dict = {}
        for k, v in self.group_db_infos.items():
            self.sampler_dict[k] = BatchSampler(v, k, shuffle=True)
        # TODO: No group_sampling currently

    @staticmethod
    def filter_by_difficulty(db_infos, removed_difficulty):
        """Filter ground truths by difficulties.

        Args:
            db_infos (dict): Info of groundtruth database.
            removed_difficulty (list): Difficulties that are not qualified.

        Returns:
            dict: Info of database after filtering.
        """
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
        return new_db_infos

    @staticmethod
    def filter_by_min_points(db_infos, min_gt_points_dict):
        """Filter ground truths by number of points in the bbox.

        Args:
            db_infos (dict): Info of groundtruth database.
            min_gt_points_dict (dict): Different number of minimum points
                needed for different categories of ground truths.

        Returns:
            dict: Info of database after filtering.
        """
        for name, min_num in min_gt_points_dict.items():
            min_num = int(min_num)
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos

    def sample_all(self,
                   GGA_init_pseudo_labels,
                   gt_labels,
                   mask_valid,
                   min_distance=5.0,
                   ground_plane=None):
        """Sampling all categories of bboxes.

        Args:
            gt_bboxes (np.ndarray): Ground truth bounding boxes.
            gt_labels (np.ndarray): Ground truth labels of boxes.

        Returns:
            dict: Dict of sampled 'pseudo ground truths'.

                - gt_labels_3d (np.ndarray): ground truths labels
                    of sampled objects.
                - gt_bboxes_3d (:obj:`BaseInstance3DBoxes`):
                    sampled ground truth 3D bounding boxes
                - points (np.ndarray): sampled points
                - group_ids (np.ndarray): ids of sampled ground truths
        """

        # filter the point_mean_in_mask = None
        # avoid_coll_points = est_points_mean[sign_valid]

        avoid_coll_points = GGA_init_pseudo_labels[mask_valid]

        # obtain the index of db sampler
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self.sample_classes,
                                              self.sample_max_nums):
            class_label = self.cat2label[class_name]
            # sampled_num = int(max_sample_num -
            #                   np.sum([n == class_name for n in gt_names]))
            sampled_num = int(max_sample_num -
                              np.sum([n == class_label for n in gt_labels]))
            sampled_num = np.round(self.rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        # Sample GT
        sampled_gt_bbox_3ds = []
        # Sample GGA
        sampled_GGA_lidar2imgs = []
        sampled_GGA_init_pseudo_labels = []
        sampled_GGA_box_imgs = []
        sampled_GGA_init_pseudo_labels = []
        sampled_GGA_in_box_points = []
        sampled_GGA_mask_valids = []
        sampled_GGA_bdry_masks = []
        sampled_GGA_difficulties = []
        sampled_GGA_num_points_in_box2ds = []

        # avoid_coll_points = est_points_mean
        # avoid_coll_boxes = gt_bboxes_2d
        # avoid_coll_lidar2img = lidar2img
        ## avoid collision based on point point_mean_in_mask
        for class_name, sampled_num in zip(self.sample_classes,
                                           sample_num_per_class):
            if sampled_num > 0:
                sampled_cls = self.sample_class_GGA(class_name, sampled_num, avoid_coll_points, min_distance)

                sampled += sampled_cls
                if len(sampled_cls) > 0:
                    if len(sampled_cls) == 1:
                        # sample GT
                        sampled_gt_box3d = sampled_cls[0]['box3d_lidar'][np.newaxis, ...]
                        # sample GGA
                        sampled_GGA_lidar2img = sampled_cls[0]['GGA_lidar2img'][np.newaxis, ...]
                        sampled_GGA_init_pseudo_label = sampled_cls[0]['GGA_init_pseudo_label'][np.newaxis, ...]
                        sampled_GGA_box_img = sampled_cls[0]['GGA_box_img'][np.newaxis, ...]
                        sampled_GGA_mask_valid = (sampled_cls[0]['GGA_mask2d'] & sampled_cls[0]['GGA_mask_valid'] & sampled_cls[0]['GGA_mask_depth'])[np.newaxis, ...]
                        sampled_GGA_bdry_mask = sampled_cls[0]['GGA_bdry_mask'][np.newaxis, ...]
                        sampled_GGA_difficulty = sampled_cls[0]['difficulty'][np.newaxis, ...]
                        sampled_GGA_num_points_in_box2d = sampled_cls[0]['GGA_num_points_in_box2d'][np.newaxis, ...]
                        sampled_GGA_in_box_point = [sampled_cls[0]['GGA_in_box_points']]

                    else:
                        sampled_gt_box3d = np.stack(
                            [s['box3d_lidar'] for s in sampled_cls], axis=0)
                        # GGA
                        sampled_GGA_box_img = np.stack(
                            [s['GGA_box_img'] for s in sampled_cls], axis=0)
                        sampled_GGA_lidar2img = np.stack(
                            [(s['GGA_lidar2img']) for s in sampled_cls], axis=0)
                        sampled_GGA_init_pseudo_label = np.stack(
                            [(s['GGA_init_pseudo_label']) for s in sampled_cls], axis=0)
                        sampled_GGA_mask_valid = np.stack(
                            [(s['GGA_mask2d']&s['GGA_mask_depth']&s['GGA_mask_valid']) for s in sampled_cls], axis=0)
                        sampled_GGA_bdry_mask = np.stack(
                            [(s['GGA_bdry_mask']) for s in sampled_cls], axis=0)
                        sampled_GGA_difficulty = np.stack(
                            [(s['difficulty']) for s in sampled_cls], axis=0)
                        sampled_GGA_num_points_in_box2d = np.stack(
                            [(s['GGA_num_points_in_box2d']) for s in sampled_cls], axis=0)
                        sampled_GGA_in_box_point = [s['GGA_in_box_points'] for s in sampled_cls]

                    sampled_gt_bbox_3ds += [sampled_gt_box3d]
                    sampled_GGA_box_imgs += [sampled_GGA_box_img]
                    sampled_GGA_lidar2imgs += [sampled_GGA_lidar2img]
                    sampled_GGA_init_pseudo_labels += [sampled_GGA_init_pseudo_label]
                    sampled_GGA_mask_valids += [sampled_GGA_mask_valid]
                    sampled_GGA_bdry_masks += [sampled_GGA_bdry_mask]
                    sampled_GGA_difficulties += [sampled_GGA_difficulty]
                    sampled_GGA_num_points_in_box2ds += [sampled_GGA_num_points_in_box2d]
                    sampled_GGA_in_box_points += sampled_GGA_in_box_point

                    avoid_coll_points = np.concatenate(
                        [avoid_coll_points, sampled_GGA_init_pseudo_label], axis=0)

        ret = None
        if len(sampled) > 0:
            sampled_gt_bbox_3ds = np.concatenate(sampled_gt_bbox_3ds, axis=0)
            sampled_GGA_box_imgs = np.concatenate(sampled_GGA_box_imgs, axis=0)
            sampled_GGA_lidar2imgs = np.concatenate(sampled_GGA_lidar2imgs, axis=0)
            sampled_GGA_init_pseudo_labels = np.concatenate(sampled_GGA_init_pseudo_labels, axis=0)
            sampled_GGA_mask_valids = np.concatenate(sampled_GGA_mask_valids, axis=0)
            sampled_GGA_bdry_masks = np.concatenate(sampled_GGA_bdry_masks, axis=0)
            sampled_GGA_difficulties = np.concatenate(sampled_GGA_difficulties, axis=0)
            sampled_GGA_num_points_in_box2ds = np.concatenate(sampled_GGA_num_points_in_box2ds, axis=0)

            # center = sampled_gt_bboxes[:, 0:3]

            # num_sampled = len(sampled)
            s_points_list = []
            count = 0
            for info in sampled:
                file_path = os.path.join(
                    self.data_root,
                    info['path']) if self.data_root else info['path']
                results = dict(pts_filename=file_path)
                s_points = self.points_loader(results)['points']
                # s_points.translate(info['box3d_lidar'][:3]) # GGA use the absolute coordinates

                count += 1

                s_points_list.append(s_points)

            gt_labels = np.array([self.cat2label[s['name']] for s in sampled],
                                 dtype=np.long)

            # if ground_plane is not None:
            #     xyz = sampled_gt_bboxes[:, :3]
            #     dz = (ground_plane[:3][None, :] *
            #           xyz).sum(-1) + ground_plane[3]
            #     sampled_gt_bboxes[:, 2] -= dz
            #     for i, s_points in enumerate(s_points_list):
            #         s_points.tensor[:, 2].sub_(dz[i])

            ret = {
                'gt_labels_3d': gt_labels,
                'gt_bbox_3ds': sampled_gt_bbox_3ds,
                'GGA_box_imgs': sampled_GGA_box_imgs,
                'GGA_lidar2imgs': sampled_GGA_lidar2imgs,
                'GGA_init_pseudo_labels': sampled_GGA_init_pseudo_labels,
                'GGA_mask_valids': sampled_GGA_mask_valids,
                'GGA_bdry_masks': sampled_GGA_bdry_masks,
                'GGA_difficulties': sampled_GGA_difficulties,
                'GGA_in_box_points': sampled_GGA_in_box_points,
                'GGA_num_points_in_box2ds': sampled_GGA_num_points_in_box2ds,
                'points': s_points_list[0].cat(s_points_list),
                'group_ids': np.arange(mask_valid.shape[0],
                          mask_valid.shape[0] + len(sampled))
            }

        return ret

    def sample_class_GGA(self, name, num, est_points_mean, min_distance):
        """Sampling specific categories of bounding boxes.

        Args:
            name (str): Class of objects to be sampled.
            num (int): Number of sampled bboxes.
            gt_bboxes (np.ndarray): Ground truth boxes.

        Returns:
            list[dict]: Valid samples after collision test.
        """
        # we need first filter sign_mask = False
        sampled = self.sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)
        sp_mask_valid = np.stack([i['GGA_mask_valid'] for i in sampled], axis=0)
        sliced_list = np.arange(len(sampled))[sp_mask_valid]
        sampled = [sampled[i] for i in sliced_list]

        if len(sampled) == 0:
            valid_samples = []
            return valid_samples

        num_gt = est_points_mean.shape[0]
        num_sampled = len(sampled)

        # we cannot get 3d bboxes, we use the distance between point center to filter.
        pts_mean_bv = est_points_mean[:, 0:2]
        sp_pts = np.stack([i['GGA_init_pseudo_label'][:2] for i in sampled], axis=0)

        pts = np.concatenate([est_points_mean[:, :2], sp_pts], axis=0).copy()

        sp_pts_new = pts[est_points_mean.shape[0]:]
        sp_pts_bv = sp_pts_new[:, 0:2]

        total_bv = np.concatenate([pts_mean_bv, sp_pts_bv], axis=0)
        distance = pdist(total_bv)
        distance_mat = squareform(distance)
        coll_mat = distance_mat < min_distance
        coll_mat[:num_gt, :num_gt] = False

        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):
            if coll_mat[i].any():
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                valid_samples.append(sampled[i - num_gt])
        return valid_samples
    
    # will do in the future
    # def sample_class_GGA_v2(self, name, num, gt_bboxes):
    #     """Sampling specific categories of bounding boxes.

    #     Args:
    #         name (str): Class of objects to be sampled.
    #         num (int): Number of sampled bboxes.
    #         gt_bboxes (np.ndarray): Ground truth boxes.

    #     Returns:
    #         list[dict]: Valid samples after collision test.
    #     """
    #     sampled = self.sampler_dict[name].sample(num)
    #     sampled = copy.deepcopy(sampled)
    #     num_gt = gt_bboxes.shape[0]
    #     num_sampled = len(sampled)
    #     gt_bboxes_bv = box_np_ops.center_to_corner_box2d(
    #         gt_bboxes[:, 0:2], gt_bboxes[:, 3:5], gt_bboxes[:, 6])

    #     sp_boxes = np.stack([i['box3d_lidar'] for i in sampled], axis=0)
    #     boxes = np.concatenate([gt_bboxes, sp_boxes], axis=0).copy()

    #     sp_boxes_new = boxes[gt_bboxes.shape[0]:]
    #     sp_boxes_bv = box_np_ops.center_to_corner_box2d(
    #         sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5], sp_boxes_new[:, 6])

    #     total_bv = np.concatenate([gt_bboxes_bv, sp_boxes_bv], axis=0)
    #     coll_mat = data_augment_utils.box_collision_test(total_bv, total_bv)
    #     diag = np.arange(total_bv.shape[0])
    #     coll_mat[diag, diag] = False

    #     valid_samples = []
    #     for i in range(num_gt, num_gt + num_sampled):
    #         if coll_mat[i].any():
    #             coll_mat[i] = False
    #             coll_mat[:, i] = False
    #         else:
    #             valid_samples.append(sampled[i - num_gt])
    #     return valid_samples