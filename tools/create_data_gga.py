import argparse
from os import path as osp

from tools.data_converter import indoor_converter as indoor
# from tools.data_converter import kitti_converter as kitti
from tools.data_converter import kitti_converter_gga as kitti
# from tools.data_converter.create_gt_database import (
#     GTDatabaseCreater, create_groundtruth_database)
from tools.data_converter.create_gt_database_gga import (
    GTDatabaseCreater, create_groundtruth_database)


## our addition
# from tools.data_converter.create_gt_database import(
#     visual_and_calculate_boxiou)


def kitti_data_prep(root_path,
                    info_prefix,
                    version,
                    out_dir,
                    with_plane=False):
    """Prepare data related to Kitti dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        out_dir (str): Output directory of the groundtruth database info.
        with_plane (bool, optional): Whether to use plane information.
            Default: False.
    """
    kitti.create_kitti_info_file(root_path, info_prefix, with_plane)
    kitti.create_reduced_point_cloud(root_path, info_prefix)

    info_train_path = osp.join(root_path, f'{info_prefix}_infos_train_GGA.pkl')
    info_val_path = osp.join(root_path, f'{info_prefix}_infos_val_GGA.pkl')
    info_trainval_path = osp.join(root_path, f'{info_prefix}_infos_trainval_GGA.pkl')
    info_test_path = osp.join(root_path, f'{info_prefix}_infos_test.pkl')
    kitti.export_2d_annotation(root_path, info_train_path)
    kitti.export_2d_annotation(root_path, info_val_path)
    kitti.export_2d_annotation(root_path, info_trainval_path)
    kitti.export_2d_annotation(root_path, info_test_path)

    create_groundtruth_database(
        'KittiDataset_GGA',
        root_path,
        info_prefix,
        f'{out_dir}/{info_prefix}_infos_train_GGA.pkl',
        relative_path=False,
        mask_anno_path='instances_train.json',
        with_mask=(version == 'mask'))
    

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--num-points',
    type=int,
    default=-1,
    help='Number of points to sample for indoor datasets.')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'kitti':
        kitti_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            with_plane=args.with_plane)
    # elif args.dataset == 'sunrgbd':
    #     sunrgbd_data_prep(
    #         root_path=args.root_path,
    #         info_prefix=args.extra_tag,
    #         num_points=args.num_points,
    #         out_dir=args.out_dir,
    #         workers=args.workers)
