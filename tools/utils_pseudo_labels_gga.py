import gc
import io as sysio

import numba
import numpy as np
import copy
import mmcv
from mmdet3d.core.evaluation.kitti_utils.eval import get_split_parts, calculate_iou_partly


def drop_arrays_by_name(gt_names, used_classes=['Pedestrian', 'Car', 'Cyclist']):

    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def pseudo_label_matching_kitti(gt_infos, dt_annos, metric=0, num_parts=200):
    
    gt_annos = [info['annos'] for info in gt_infos]
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)
    gt_infos_reserve = copy.deepcopy(gt_infos)
    
    # remove points cluster in gt_annos for the type(List)
    for i in range(len(gt_annos)):
        gt_annos[i].pop('GGA_in_box_points')

    # Remove Donnot Care samples in gt_annos: (Need to be synchronized with the training process)
    for i in range(len(gt_annos)):
        curt_gt_annos = gt_annos[i]

        num_obj = len([n for n in curt_gt_annos['name'] if n != 'DontCare'])
        for key, value in curt_gt_annos.items():
            curt_gt_annos[key] = value[:num_obj]

        select = drop_arrays_by_name(curt_gt_annos['name'])
        for key, value in curt_gt_annos.items():
            curt_gt_annos[key] = value[select]

    # calcuate the 2D iou on the image between peojected predicted 3D boxes and original 2D bounding boxes
    if num_examples < num_parts:
        num_parts = num_examples
    split_parts = get_split_parts(num_examples, num_parts)

    rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)
    overlaps, parted_overlaps, total_dt_num, total_gt_num = rets

    new_gt_annos = []
    for i, c_overlap in enumerate(overlaps):
        gt_anno_c = gt_annos[i]
        dt_anno_c = dt_annos[i]
        new_dict = dict()

        if len(dt_anno_c['name']) == 0:
            for key, value in gt_anno_c.items():
                new_dict[key] = gt_anno_c[key][:0]
            new_gt_annos.append(new_dict)
            continue

        dt_match_gt = np.argmax(c_overlap, axis=-1)
        for key, value in gt_anno_c.items():
            if key in dt_anno_c:
                # use the predicted results in dt_anno_c
                new_dict[key] = dt_anno_c[key]
            else:
                # keep the GGA information in gt_anno_c
                new_dict[key] = gt_anno_c[key][dt_match_gt]
        new_gt_annos.append(new_dict)

    filename = './data/kitti_pesudo/kitti_infos_trainval_GGA_pseudo.pkl'
    for index, sample in enumerate(gt_infos_reserve):
        sample.pop('annos')
        # adjust the predicted rotation in dt_anno_c
        for j in range(new_gt_annos[index]['rotation_y'].shape[0]):
            dim = new_gt_annos[index]['dimensions'][j:j+1]
            if dim[:, 2] > dim[:, 0]:
                new_gt_annos[index]['dimensions'][j:j+1] = new_gt_annos[index]['dimensions'][j:j+1][:, [2, 1, 0]]
                new_gt_annos[index]['rotation_y'][j:j+1] = new_gt_annos[index]['rotation_y'][j:j+1] + np.pi/2.0

        # replace the anno in gt_info by dt_anno results
        sample['annos'] = new_gt_annos[index]

    mmcv.dump(gt_infos_reserve, filename)
    return gt_annos # compared to original gt_annos, only remove In-Box-Points and Donnot care


def clean_prediction_kitti(gt_annos, dt_annos, metric=0):
    
    assert len(gt_annos) == len(dt_annos)
    num_examples = len(gt_annos)

    # clean gt_annos: (Need to be synchronized with the training process)
    for i in range(len(gt_annos)):
        curt_gt_annos = gt_annos[i]
        num_obj = len([n for n in curt_gt_annos['name'] if n != 'DontCare'])
        for key, value in curt_gt_annos.items():
            if key in dt_annos[0]:
                curt_gt_annos[key] = value[:num_obj]

        select = drop_arrays_by_name(curt_gt_annos['name'])
        for key, value in curt_gt_annos.items():
            if key in dt_annos[0]:
                curt_gt_annos[key] = value[select]

        # clearn sign
        num_points_in_gt = gt_annos[i]['num_points_in_gt']
        difficulty = gt_annos[i]['difficulty']
        sign_valid = gt_annos[i]['sign_valid']
        sign_depth = gt_annos[i]['sign_depth']
        curt_sign = np.logical_and(sign_valid, sign_depth)
        curt_sign = np.logical_and(curt_sign, difficulty > -1)
        curt_sign = np.logical_and(curt_sign, num_points_in_gt > 30)
        for key, value in curt_gt_annos.items():
            if key in dt_annos[0]:
                curt_gt_annos[key] = value[curt_sign]

    # filter dt_annos:

    # if num_examples < num_parts:
    #     num_parts = num_examples
    # split_parts = get_split_parts(num_examples, num_parts)

    # rets = calculate_iou_partly(dt_annos, gt_annos, metric, num_parts)

    return gt_annos