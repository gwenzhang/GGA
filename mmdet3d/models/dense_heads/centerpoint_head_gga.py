# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmcv.cnn import ConvModule, build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch import nn

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr)
from mmdet3d.core.post_processing import nms_bev
from mmdet3d.models import builder
from mmdet3d.models.utils import clip_sigmoid
from mmdet.core import build_bbox_coder, multi_apply
from ..builder import HEADS, build_loss
from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis


@HEADS.register_module()
class CenterHead_GGA(BaseModule):
    """CenterHead for CenterPoint.

    Args:
        in_channels (list[int] | int, optional): Channels of the input
            feature map. Default: [128].
        tasks (list[dict], optional): Task information including class number
            and class names. Default: None.
        train_cfg (dict, optional): Train-time configs. Default: None.
        test_cfg (dict, optional): Test-time configs. Default: None.
        bbox_coder (dict, optional): Bbox coder configs. Default: None.
        common_heads (dict, optional): Conv information for common heads.
            Default: dict().
        loss_cls (dict, optional): Config of classification loss function.
            Default: dict(type='GaussianFocalLoss', reduction='mean').
        loss_bbox (dict, optional): Config of regression loss function.
            Default: dict(type='L1Loss', reduction='none').
        separate_head (dict, optional): Config of separate head. Default: dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3)
        share_conv_channel (int, optional): Output channels for share_conv
            layer. Default: 64.
        num_heatmap_convs (int, optional): Number of conv layers for heatmap
            conv layer. Default: 2.
        conv_cfg (dict, optional): Config of conv layer.
            Default: dict(type='Conv2d')
        norm_cfg (dict, optional): Config of norm layer.
            Default: dict(type='BN2d').
        bias (str, optional): Type of bias. Default: 'auto'.
    """

    def __init__(self,
                 in_channels=[128],
                 tasks=None,
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 common_heads=dict(),
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(
                     type='L1Loss', reduction='none', loss_weight=0.25),
                 loss_center=dict(type='MarginL1Loss', reduction='mean'),
                 separate_head=dict(
                     type='SeparateHead', init_bias=-2.19, final_kernel=3),
                 share_conv_channel=64,
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv2d'),
                 norm_cfg=dict(type='BN2d'),
                 bias='auto',
                 norm_bbox=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
            'behavior, init_cfg is not allowed to be set'
        super(CenterHead_GGA, self).__init__(init_cfg=init_cfg)

        num_classes = [len(t['class_names']) for t in tasks]
        self.class_names = [t['class_names'] for t in tasks]
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.norm_bbox = norm_bbox

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        # self.loss_center = build_loss(loss_center)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.num_anchor_per_locs = [n for n in num_classes]
        self.fp16_enabled = False

        # a shared convolution
        self.shared_conv = ConvModule(
            in_channels,
            share_conv_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=bias)

        self.task_heads = nn.ModuleList()

        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(num_cls, num_heatmap_convs)))
            separate_head.update(
                in_channels=share_conv_channel, heads=heads, num_cls=num_cls)
            self.task_heads.append(builder.build_head(separate_head))

        self.with_velocity = 'vel' in common_heads.keys()

    def forward_single(self, x):
        """Forward function for CenterPoint.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            list[dict]: Output results for tasks.
        """
        ret_dicts = []

        x = self.shared_conv(x)

        for task in self.task_heads:
            ret_dicts.append(task(x))

        return ret_dicts

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        return multi_apply(self.forward_single, feats)

    def _gather_feat(self, feat, ind, mask=None):
        """Gather feature map.

        Given feature map and index, return indexed feature map.

        Args:
            feat (torch.tensor): Feature map with the shape of [B, H*W, 10].
            ind (torch.Tensor): Index of the ground truth boxes with the
                shape of [B, max_obj].
            mask (torch.Tensor, optional): Mask of the feature map with the
                shape of [B, max_obj]. Default: None.

        Returns:
            torch.Tensor: Feature map after gathering with the shape
                of [B, max_obj, 10].
        """
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat
    
    ## GGA Operations
    def GGA_calculate_rotation(self, pred):
        
        rot_sine = pred[..., 0]
        rot_cosine = pred[..., 1]
        rot = torch.atan2(rot_sine, rot_cosine).squeeze()

        ones = torch.ones_like(rot_cosine)
        zeros = torch.zeros_like(rot_cosine)
        
        rmat_T = torch.stack([
            torch.stack([rot_cosine, rot_sine, zeros], dim=-1),
            torch.stack([-rot_sine, rot_cosine, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ], dim=-1)

        return rot, rmat_T
    
    def get_distance_single(self, ibp_points, box_bev):
        
        xdiss_all = []
        ydiss_all = []
        mindiss_all = []
        xdiss = box_bev.new_zeros((box_bev.shape[0], 1), dtype=torch.float32)
        ydiss = box_bev.new_zeros((box_bev.shape[0], 1), dtype=torch.float32)
        mindiss = box_bev.new_zeros((box_bev.shape[0], 1), dtype=torch.float32)

        for index, clt in enumerate(ibp_points):
            # calculate the rotation            
            crot = box_bev[index, -1, None]
            cbox_center = box_bev[index, None, :2]
            half_l = box_bev[index, 2, None] / 2.0
            half_h = box_bev[index, 3, None] / 2.0 
            # rotate the points
            # clt_r = rotation_3d_in_axis_with_rmat(clt[..., :3], crot, axis=2, clockwise=True)
            clt_r = rotation_3d_in_axis(clt[..., :2].float(), crot, axis=2, clockwise=True)
            cbox_center_r = rotation_3d_in_axis(cbox_center, crot, axis=2, clockwise=True)[0]
            # extend as boxes
            boxes_xmin = cbox_center_r[0] - half_l
            boxes_xmax = cbox_center_r[0] + half_l
            boxes_ymin = cbox_center_r[1] - half_h
            boxes_ymax = cbox_center_r[1] + half_h
            
            # calculate the distance/(inbox)
            dx1 = clt_r[..., 0] - boxes_xmin
            dx2 = clt_r[..., 0] - boxes_xmax
            dy1 = clt_r[..., 1] - boxes_ymin
            dy2 = clt_r[..., 1] - boxes_ymax

            # v1
            p2c_x = torch.abs(clt_r[..., 0] - cbox_center_r[0])
            p2c_y = torch.abs(clt_r[..., 1] - cbox_center_r[1])
            dx = torch.relu(p2c_x - 2 * half_l)
            dy = torch.relu(p2c_y - 2 * half_h)

            dis = torch.stack([dx1, dx2, dy1, dy2]).transpose(1, 0)
            dis = torch.abs(dis)
            # dis = torch.min(dis[..., :2], dim=-1)[0] + torch.min(dis[..., 2:4], dim=-1)[0]
            all_dis = torch.min(dis, dim=-1)[0]
            mindiss[index] = all_dis.sum()
            xdiss[index] = dx.sum()
            ydiss[index] = dy.sum()
            
            # filter (whether)
            # k = int(0.8 * dis.numel())
            # threshold, _ = torch.kthvalue(dis, k)
            # dis_mask = dis < threshold
            # dis = dis * dis_mask.float()
        
        # dis_sat_all.append(dis_sat)
        xdiss_all.append(xdiss)
        ydiss_all.append(ydiss)
        mindiss_all.append(mindiss)
        return mindiss_all, xdiss_all, ydiss_all

    def get_distance_bev(self, ibp_points, pred_box_bev):

        pts_min_dis, pts_x_dis, pts_y_dis = multi_apply(self.get_distance_single, ibp_points, pred_box_bev)
        pts_min_dis = torch.stack([dis[0] for dis in pts_min_dis])
        pts_x_dis = torch.stack([dis[0] for dis in pts_x_dis])
        pts_y_dis = torch.stack([dis[0] for dis in pts_y_dis])

        return pts_min_dis, pts_x_dis, pts_y_dis
    
    def get_prediction_single(self, pred_all, ind, ann_lidar2img, rot):

        def corners(dims, center3d, rot):
            import numpy as np
            from mmdet3d.core.bbox.structures.utils import rotation_3d_in_axis

            if self.norm_bbox:
                dims = torch.exp(dims)

            corners_norm = torch.from_numpy(
                np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)).to(
                device=dims.device, dtype=dims.dtype)

            corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
            # use relative origin [0.5, 0.5, 0]
            corners_norm = corners_norm - dims.new_tensor([0.5, 0.5, 0])
            corners = dims.view([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

            # bbox = torch.cat(center3d, dims)
            # rotate around z axis
            corners = rotation_3d_in_axis(corners, rot, axis=2)
            # rmat = rot.permute(1, 2, 0)
            # corners = rotation_3d_in_axis_with_rmat(corners, rmat, axis=2)
            corners += center3d.view(-1, 1, 3)

            return corners
        
        def get_box_bev(boxes_temp, rot):
            
            rot = rot
            center = boxes_temp[..., :2]
            w = (torch.exp(boxes_temp[..., 2, None]))
            h = (torch.exp(boxes_temp[..., 3, None]))
            
            boxes = torch.cat([center, w, h, rot.unsqueeze(-1)], dim=-1)

            return boxes

        grid_size = torch.tensor(self.train_cfg['grid_size'])
        feature_map_size = torch.div(grid_size[:2], self.train_cfg['out_size_factor'], rounding_mode='trunc')
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])

        voxel_y = (torch.div(ind, feature_map_size[0], rounding_mode='trunc') + pred_all[..., 1]) * voxel_size[1] * self.train_cfg['out_size_factor'] + pc_range[1]
        voxel_x = ((ind % feature_map_size[0]) + pred_all[..., 0]) * voxel_size[0] * self.train_cfg['out_size_factor'] + pc_range[0]
        voxel = torch.cat([voxel_x[..., None], voxel_y[..., None]], dim=-1)
        bev_center = voxel
        pred_box = torch.cat([bev_center, pred_all[..., 2:]], dim=-1)

        pred = pred_box
        center3d = pred[..., :3].view(-1, 3)

        b, max_objs, _ = pred.shape
        device = pred.device

        dim = pred[..., 3:6].view(-1, 3)
        pred_bev_temp = pred[..., [0, 1, 3, 4]]
        pred_box_bev = get_box_bev(pred_bev_temp, rot)
        rot = rot.view(-1)

        bottom_center3d = center3d
        bias = torch.zeros_like(bottom_center3d)
        if self.norm_bbox:
            bias[..., -1] = -torch.exp(pred[..., 5]).view(-1) * 0.5
        else:
            bias[..., -1] = -pred[..., 5].view(-1) * 0.5
        bottom_center3d = bottom_center3d + bias
        corner3d = corners(dim, bottom_center3d, rot)
        corner3d = torch.cat((corner3d, torch.ones(corner3d.shape[0], corner3d.shape[1], 1).to(device)), dim=-1)

        lidar2imgs = ann_lidar2img.view(-1, 4, 4)

        """
        Supervised the dimension, rotation
        """
        # pts_img = lidar2imgs @ corner3d.T
        pts_img = torch.einsum('bij,bjk->bik', lidar2imgs, corner3d.permute(0, 2, 1))
        depth = pts_img[:, 2, None, :]
        corner_valid = (depth > 0).squeeze()
        depth = torch.maximum(depth, torch.tensor([0.1], device=depth.device))
        pixel = (pts_img[:, :2, :] / (depth)).permute(0, 2, 1)

        xmin = torch.min(pixel[..., 0], dim=-1)[0]
        xmax = torch.max(pixel[..., 0], dim=-1)[0]
        ymin = torch.min(pixel[..., 1], dim=-1)[0]
        ymax = torch.max(pixel[..., 1], dim=-1)[0]
        
        pred_ratio = torch.cat([torch.exp(pred[..., 3, None]), torch.exp(pred[..., 4, None])], dim=-1)
        pred_iou = torch.cat((xmin[:, None], ymin[:, None], xmax[:, None], ymax[:, None]), dim=-1)
        pred_iou = pred_iou.reshape(b, max_objs, -1)

        return pred_ratio, pred_iou, pred_box_bev

    def get_targets(self, gt_bboxes_3d, gt_labels_3d,
                    GGA_boxes_img, GGA_lidar2img, GGA_init_pseudo_labels, GGA_bdry_masks, GGA_in_box_points, img_metas):
        """Generate targets.

        How each output is transformed:

            Each nested list is transposed so that all same-index elements in
            each sub-list (1, ..., N) become the new sub-lists.
                [ [a0, a1, a2, ... ], [b0, b1, b2, ... ], ... ]
                ==> [ [a0, b0, ... ], [a1, b1, ... ], [a2, b2, ... ] ]

            The new transposed nested list is converted into a list of N
            tensors generated by concatenating tensors in the new sub-lists.
                [ tensor0, tensor1, tensor2, ... ]

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which
                        boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks, anno_lidar2imgs, ibp_points, anno_bound_masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d, GGA_boxes_img, GGA_lidar2img, GGA_init_pseudo_labels, GGA_bdry_masks, GGA_in_box_points, img_metas)
        # Transpose heatmaps
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # Transpose anno_boxes
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # Transpose inds
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        # Transpose inds
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]

        # GGA
        anno_lidar2imgs = list(map(list, zip(*anno_lidar2imgs)))
        anno_lidar2imgs = [torch.stack(anno_lidar2imgs_) for anno_lidar2imgs_ in anno_lidar2imgs]
        # In-Box-Points
        ibp_points = list(map(list, zip(*ibp_points)))
        # Bound Masks
        anno_bound_masks = list(map(list, zip(*anno_bound_masks)))
        anno_bound_masks = [torch.stack(anno_bound_masks_) for anno_bound_masks_ in anno_bound_masks]

        return heatmaps, anno_boxes, inds, masks, anno_lidar2imgs, ibp_points, anno_bound_masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d,
                           GGA_boxes_img, GGA_lidar2img, GGA_init_pseudo_labels, GGA_bdry_masks, GGA_in_box_points, img_meta):
        """Generate training targets for GGA.
           !!!!! The GT(gt_bboxes_3d) is only used to debug, not for providing annotations  !!!!
        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            GGA_boxes_img: 2D Labels for GGA.
            GGA_lidar2img: The projection calibration from LiDAR to Image.
            GGA_init_pseudo_labels: The intial pseudo 3D boxes, which is generated from valid foreground points.
            GGA_bdry_mask: To indicate the 2D objects on the image boundaries.
            GGA_in_box_points: In-Box-Points.
        """
        device = gt_labels_3d.device
        # Debug_gt_bboxes_3d_corners = gt_bboxes_3d.corners.to(device)
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size']).to(device)
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        # GGA: GGA_boxes_img, GGA_lidar2img, GGA_init_pseudo_labels, GGA_bdry_masks, GGA_in_box_points,
        task_GGA_box_imgs = []
        task_GGA_lidar2imgs = []
        task_GGA_init_pseudo_labels = []
        task_GGA_bdry_masks = []
        task_GGA_in_box_points = []
        # Debug
        # task_Debug_box_corners = [] 

        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            # GGA
            task_GGA_box_img = []
            task_GGA_lidar2img = []
            task_GGA_init_pseudo_label = []
            task_GGA_bdry_mask = []
            task_GGA_in_box_point = []
            # Debug
            # task_Debug_box_corner = [] 

            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
                # GGA
                task_GGA_box_img.append(GGA_boxes_img[m])
                task_GGA_bdry_mask.append(GGA_bdry_masks[m])
                task_GGA_init_pseudo_label.append(GGA_init_pseudo_labels[m])
                task_GGA_lidar2img.append(GGA_lidar2img[m])
                for p in m:
                    task_GGA_in_box_point.append([GGA_in_box_points[pp].to(device) for pp in p])
                # Debug
                # task_Debug_box_corner.append(Debug_gt_bboxes_3d_corners[m])

            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            # GGA
            task_GGA_box_imgs.append(torch.cat(task_GGA_box_img, axis=0).to(device))
            task_GGA_lidar2imgs.append(torch.cat(task_GGA_lidar2img).to(device))
            task_GGA_bdry_masks.append(torch.cat(task_GGA_bdry_mask).to(device))
            task_GGA_init_pseudo_labels.append(torch.cat(task_GGA_init_pseudo_label).to(device))
            task_GGA_in_box_points.append([element for sub_cluster in task_GGA_in_box_point for element in sub_cluster])
            # Debug
            # task_Debug_box_corners.append(torch.cat(task_Debug_box_corner).to(device))

            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []
        anno_lidar2imgs = []
        anno_boundry_masks = []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            if self.with_velocity:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                                  dtype=torch.float32)
            else:
                anno_box = gt_bboxes_3d.new_zeros((max_objs, 5),
                                                  dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)
            GGA_boundary_mask = gt_bboxes_3d.new_zeros((max_objs, 4), dtype=torch.uint8)

            # to avoid generate lidr2img witho zeros
            GGA_base_lidar2img = torch.from_numpy(img_meta['lidar2img']).to(device)
            GGA_base_lidar2img = GGA_base_lidar2img[None].repeat(max_objs, 1, 1)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            # Semantic Ratio Loss
            if idx == 0: # Pedestrian
                random_tensor = torch.normal(torch.tensor(1.35), torch.tensor(0.48))
                random_tensor = torch.clamp(random_tensor, min=1e-3)
                srl = random_tensor.to(device)
            elif idx == 1: # Cyclist
                random_tensor = torch.normal(torch.tensor(3.60), torch.tensor(0.68))
                random_tensor = torch.clamp(random_tensor, min=1e-3)
                srl = random_tensor.to(device)
            else: # Car
                random_tensor = torch.normal(torch.tensor(2.40), torch.tensor(0.28))
                random_tensor = torch.clamp(random_tensor, min=1e-3)
                srl = random_tensor.to(device)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                # GT Gassuian Center (Can be used to verify)
                # width = task_boxes[idx][k][3]
                # length = task_boxes[idx][k][4]
                # width = width / voxel_size[0] / self.train_cfg[
                #     'out_size_factor']
                # length = length / voxel_size[1] / self.train_cfg[
                #     'out_size_factor']
                # GGA Gassuian Center

                lidar2img = task_GGA_lidar2imgs[idx][k]
                bpl_x1, bpl_y1, bpl_x2, bpl_y2 = task_GGA_box_imgs[idx][k]
                GGA_width = task_GGA_init_pseudo_labels[idx][k][3]
                GGA_length = task_GGA_init_pseudo_labels[idx][k][4]
                GGA_width = GGA_width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                GGA_length = GGA_length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if GGA_width > 0 and GGA_length > 0:
                    radius = gaussian_radius(
                        (GGA_length, GGA_width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    # GGA
                    GGA_x, GGA_y = task_GGA_init_pseudo_labels[idx][k][0], task_GGA_init_pseudo_labels[idx][k][1]
                    GGA_coor_x = (
                        GGA_x - pc_range[0]
                    ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    GGA_coor_y = (
                        GGA_y - pc_range[1]
                    ) / voxel_size[1] / self.train_cfg['out_size_factor']
                    GGA_center = torch.tensor([GGA_coor_x, GGA_coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    GGA_center_int = GGA_center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap

                    # GGA
                    if not (0 <= GGA_center_int[0] < feature_map_size[0] and 0 <= GGA_center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], GGA_center_int, radius)

                    new_idx = k
                    x, y = GGA_center_int[0], GGA_center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # rot = task_boxes[idx][k][6]
                    # box_dim = task_boxes[idx][k][3:6]
                    GGA_base_lidar2img[new_idx] = lidar2img
                    GGA_boundary_mask[new_idx] = ~task_GGA_bdry_masks[idx][k]

                    # Debug: 
                    # Debug_3D_anno_corners = task_Debug_box_corners[idx][k]
                    # Debug_3D_anno_corners = torch.cat((Debug_3D_anno_corners, torch.ones(Debug_3D_anno_corners.shape[0], 1).to(device)),dim=-1)
                    # Debug_lidar2img = lidar2img
                    # Debug_pts_img = Debug_lidar2img @ Debug_3D_anno_corners.T
                    # pixel = (Debug_pts_img[:2, :] / Debug_pts_img[2, None, :]).permute(1, 0)
                    # debug_valid = (Debug_pts_img[2, None, :] > 0).squeeze()
                    # debug_xmin = torch.min(pixel[debug_valid][..., 0], dim=-1)[0] # Is equal to bpl_x1 ?
                    # debug_xmax = torch.max(pixel[debug_valid][..., 0], dim=-1)[0] # bpl_y1
                    # debug_ymin = torch.min(pixel[debug_valid][..., 1], dim=-1)[0] # bpl_x2
                    # debug_ymax = torch.max(pixel[debug_valid][..., 1], dim=-1)[0] # bpl_y2

                    if self.with_velocity:
                        # GGA don not support velocity !!!!!
                        # Please don not set with_velocity = True !!!!!!!
                        vx, vy = task_boxes[idx][k][7:]
                        anno_box[new_idx] = torch.cat([
                            bpl_x1.unsqueeze(0), bpl_y1.unsqueeze(0),
                            bpl_x2.unsqueeze(0), bpl_y2.unsqueeze(0),
                            srl.unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0),
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            bpl_x1.unsqueeze(0), bpl_y1.unsqueeze(0),
                            bpl_x2.unsqueeze(0), bpl_y2.unsqueeze(0),
                            srl.unsqueeze(0),
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
            anno_lidar2imgs.append(GGA_base_lidar2img)
            anno_boundry_masks.append(GGA_boundary_mask)
        return heatmaps, anno_boxes, inds, masks, anno_lidar2imgs, task_GGA_in_box_points, anno_boundry_masks

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, 
             GGA_boxes_img, GGA_lidar2img, GGA_init_pseudo_labels, GGA_bdry_masks, GGA_in_box_points, img_metas,
             **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks, anno_lidar2imgs, ibp_points, anno_bound_masks = self.get_targets(
            gt_bboxes_3d, gt_labels_3d,
            GGA_boxes_img, GGA_lidar2img, GGA_init_pseudo_labels, GGA_bdry_masks, GGA_in_box_points, img_metas)
        loss_dict = dict()
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_heatmap = self.loss_cls(
                preds_dict[0]['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1))
            target_box = anno_boxes[task_id]
            boundary_mask = anno_bound_masks[task_id] 
            # reconstruct the anno_box from multiple reg heads
            if self.with_velocity:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['vel']),
                    dim=1)
            else:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot']),
                    dim=1)

            # Regression loss for dimension, offset, height, rotation
            ind = inds[task_id]
            num = masks[task_id].float().sum()
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)

            # GGA
            rot, rmat = self.GGA_calculate_rotation(pred[..., 6:])
            pred_ratio, pred_iou, pred_box_bev = self.get_prediction_single(pred, ind, anno_lidar2imgs[task_id], rot)

            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan

            code_weights = self.train_cfg.get('code_weights', None)
            bbox_weights = mask * mask.new_tensor(code_weights)

            # GGA
            # Point-to-Box Alignment Loss
            loss_pal_weights = bbox_weights[..., 0, None]
            p2c_min, p2c_x, p2c_y = self.get_distance_bev(ibp_points[task_id], pred_box_bev)
            p2c_target = torch.zeros_like(p2c_min)
            loss_pal = self.loss_bbox(p2c_min, p2c_target, loss_pal_weights, avg_factor=(num + 1e-4))
            loss_palx = self.loss_bbox(p2c_x, p2c_target, loss_pal_weights, avg_factor=(num + 1e-4))
            loss_paly = self.loss_bbox(p2c_y, p2c_target, loss_pal_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.distancex'] = loss_palx * 0.1
            loss_dict[f'task{task_id}.distancey'] = loss_paly * 0.1
            loss_dict[f'task{task_id}.distancemin'] = loss_pal * 0.1
            # if task_id == 2:
            #     loss_dict[f'task{task_id}.distancemin'] = loss_pal * 0.1

            # Semantic Ratio Loss
            loss_ratio_weight = bbox_weights[..., -1, None]
            srl_w_index = torch.min(pred_ratio, dim=-1)[1]
            srl_l_index = torch.max(pred_ratio, dim=-1)[1]
            ratio_coef = target_box[..., -1, None]
            ratio_w = torch.gather(pred_ratio, dim=-1, index=srl_w_index[..., None])
            ratio_l = torch.gather(pred_ratio, dim=-1, index=srl_l_index[..., None])
            srl = ratio_l - ratio_w * ratio_coef
            srl_target = torch.zeros_like(srl)
            loss_srl = self.loss_bbox(srl, srl_target, loss_ratio_weight, avg_factor=(num + 1e-4))

            # Boundary Projection Loss
            loss_bpl_weights = bbox_weights[..., :4]
            masked_loss_bpl_weights = loss_bpl_weights * boundary_mask
            loss_bpl = self.loss_bbox(pred_iou, target_box[..., :4], masked_loss_bpl_weights, avg_factor=(num + 1e-4))

            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap * 5.0
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bpl * 0.3
            loss_dict[f'task{task_id}.loss_ratio'] = loss_srl * 0.1

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['min_radius'][task_id],
                            post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](
                        bboxes, self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_task_detections(self, num_class_with_bg, batch_cls_preds,
                            batch_reg_preds, batch_cls_labels, img_metas):
        """Rotate nms for each task.

        Args:
            num_class_with_bg (int): Number of classes for the current task.
            batch_cls_preds (list[torch.Tensor]): Prediction score with the
                shape of [N].
            batch_reg_preds (list[torch.Tensor]): Prediction bbox with the
                shape of [N, 9].
            batch_cls_labels (list[torch.Tensor]): Prediction label with the
                shape of [N].
            img_metas (list[dict]): Meta information of each sample.

        Returns:
            list[dict[str: torch.Tensor]]: contains the following keys:

                -bboxes (torch.Tensor): Prediction bboxes after nms with the
                    shape of [N, 9].
                -scores (torch.Tensor): Prediction scores after nms with the
                    shape of [N].
                -labels (torch.Tensor): Prediction labels after nms with the
                    shape of [N].
        """
        predictions_dicts = []
        post_center_range = self.test_cfg['post_center_limit_range']
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device)

        for i, (box_preds, cls_preds, cls_labels) in enumerate(
                zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)):

            # Apply NMS in bird eye view

            # get the highest score per prediction, then apply nms
            # to remove overlapped box.
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long)

            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.test_cfg['score_threshold'] > 0.0:
                thresh = torch.tensor(
                    [self.test_cfg['score_threshold']],
                    device=cls_preds.device).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.test_cfg['score_threshold'] > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](
                    box_preds[:, :], self.bbox_coder.code_size).bev)
                # the nms in 3d detection just remove overlap boxes.

                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.test_cfg['nms_thr'],
                    pre_max_size=self.test_cfg['pre_max_size'],
                    post_max_size=self.test_cfg['post_max_size'])
            else:
                selected = []

            # if selected is not None:
            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            # finally generate predictions.
            if selected_boxes.shape[0] != 0:
                box_preds = selected_boxes
                scores = selected_scores
                label_preds = selected_labels
                final_box_preds = box_preds
                final_scores = scores
                final_labels = label_preds
                if post_center_range is not None:
                    mask = (final_box_preds[:, :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (final_box_preds[:, :3] <=
                             post_center_range[3:]).all(1)
                    predictions_dict = dict(
                        bboxes=final_box_preds[mask],
                        scores=final_scores[mask],
                        labels=final_labels[mask])
                else:
                    predictions_dict = dict(
                        bboxes=final_box_preds,
                        scores=final_scores,
                        labels=final_labels)
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = dict(
                    bboxes=torch.zeros([0, self.bbox_coder.code_size],
                                       dtype=dtype,
                                       device=device),
                    scores=torch.zeros([0], dtype=dtype, device=device),
                    labels=torch.zeros([0],
                                       dtype=top_labels.dtype,
                                       device=device))

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
