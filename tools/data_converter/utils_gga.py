import numpy as np
from mmdet3d.core.bbox.box_np_ops import projection_matrix_to_CRT_kitti, get_frustum, camera_to_lidar, \
    corner_to_surfaces_3d_jit, points_in_convex_polygon_3d_jit


def region_grow(pc, mask_search, mask_origin, thresh, ratio=0.8):
    pc_search = pc[mask_search==1]
    mask = mask_origin.copy()
    best_len = 0
    mask_best = np.zeros((pc.shape[0]))
    while mask.sum() > 0:
        seed = pc[mask==1][0]
        seed_mask = np.zeros((pc_search.shape[0]))
        seed_mask_all = np.zeros((pc.shape[0]))
        seed_list = [seed]
        flag = 1
        while len(seed_list) > 0:
            temp = seed_list.pop(0)
            dis = np.linalg.norm(pc_search - temp, axis=-1)
            index = np.argmin(dis)
            seed_mask[index] = 1
            valid_mask = (dis < thresh) * (1 - seed_mask)
            seed_list += list(pc_search[valid_mask==1])
            seed_mask[valid_mask == 1] = 1
            seed_mask_all[mask_search==1] = seed_mask
            if ratio is not None and (seed_mask_all*mask_origin).sum()/seed_mask.sum().astype(np.float32)<ratio:
                flag = 0
                break
        if flag == 1:
            if seed_mask.sum() > best_len:
                best_len = seed_mask.sum()
                mask_best = seed_mask_all
        mask *= (1 - seed_mask_all)

    if ratio is not None:
        return mask_best*mask_origin
    else:
        return mask_best


def check_parallel(points):
    a = np.linalg.norm(points[0] - points[1])
    b = np.linalg.norm(points[1] - points[2])
    c = np.linalg.norm(points[2] - points[0])
    p = (a + b + c) / 2
    
    area = np.sqrt(p * (p - a) * (p - b) * (p - c))
    if area < 1e-2:
        return True
    else:
        return False
    
def fitPlane(points):
    if points.shape[0] == points.shape[1]:
        return np.linalg.solve(points, np.ones(points.shape[0]))
    else:
        return np.linalg.lstsq(points, np.ones(points.shape[0]))[0]


def project_pts_on_img(points,
                       raw_img,
                       lidar2img):

    img = raw_img.copy()
    num_points = points.shape[0]
    pts_4d = np.concatenate([points[:, :3], np.ones((num_points, 1))], axis=-1)
    pts_2d = pts_4d @ lidar2img.T

    # cam_points is Tensor of Nx4 whose last column is 1
    # transform camera coordinate to image coordinate
    pts_2d[:, 2] = np.clip(pts_2d[:, 2], a_min=1e-5, a_max=99999)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]

    pts_2d = np.round(pts_2d[:, :2]).astype(np.int64)

    fov_inds = ((pts_2d[:, 0] < img.shape[1])
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 1] < img.shape[0])
                & (pts_2d[:, 1] >= 0))

    imgfov_pts_2d = pts_2d[fov_inds, :3]
    points = points[fov_inds]

    return imgfov_pts_2d, points, fov_inds


def points_in_frustm_indices(points, rect, Trv2c, P2, bbox_shape):

    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = bbox_shape.tolist()
    frustum = get_frustum(image_bbox, C)
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    # points = points[indices.reshape([-1])]
    # num_points = indices.sum()

    return indices


def calculate_ground(point_cloud, thresh_ransac=0.15, back_cut=False, back_cut_z=-5.0):

    if back_cut:
        point_cloud = point_cloud[point_cloud[:,2] > back_cut_z]   # camera frame 3 x N
    planeDiffThreshold = thresh_ransac
    temp = np.sort(point_cloud[:,1])[int(point_cloud.shape[0]*0.75)]
    cloud = point_cloud[point_cloud[:,1]>temp]
    points_np = point_cloud
    mask_all = np.ones(points_np.shape[0])
    final_sample_points = None
    for i in range(5):
         best_len = 0
         for iteration in range(min(cloud.shape[0], 100)):
             sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
                
             while check_parallel(sampledPoints) == True:
                sampledPoints = cloud[np.random.choice(np.arange(cloud.shape[0]), size=(3), replace=False)]
                continue

             plane = fitPlane(sampledPoints)
             diff = np.abs(np.matmul(points_np, plane) - np.ones(points_np.shape[0])) / np.linalg.norm(plane)
             inlierMask = diff < planeDiffThreshold
             numInliers = inlierMask.sum()
             if numInliers > best_len and np.abs(np.dot(plane/np.linalg.norm(plane),np.array([0,1,0])))>0.9:
                 mask_ground = inlierMask
                 best_len = numInliers
                 best_plane = plane
                 final_sample_points = sampledPoints
         mask_all *= 1 - mask_ground
    return mask_all, final_sample_points