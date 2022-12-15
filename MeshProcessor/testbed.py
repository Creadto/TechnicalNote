import os
import copy
import numpy as np
from util.files import load_pcds, load_meshes, get_filename, get_position, load_ply
from proc.preprocessing import convert_img
from proc.clustering import get_largest_cluster, get_parts
import open3d as o3d
import cv2


def transform_test(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]

    run_flag = True

    pcds = []
    new_pcds = []
    for idx, filename in enumerate(file_list):
        cam_loc = get_position(os.path.join(root, filename))
        origin_pcd: o3d.geometry.PointCloud = load_ply(root=root, filename=filename, cam_loc=cam_loc)
        o3d.visualization.draw_geometries([origin_pcd])
        origin_pcd = get_largest_cluster(origin_pcd)
        origin_pcd = origin_pcd.translate((0, 2.0 * idx, 0))
        offset_unit = 1.0
        for idx in range(1, 9):
            target_pcd: o3d.geometry.PointCloud = copy.deepcopy(origin_pcd)
            r = target_pcd.get_rotation_matrix_from_xyz((0, 2 * np.pi * (idx / 8), 0))
            target_pcd = target_pcd.translate((offset_unit * idx, 0, offset_unit * idx))
            target_pcd = target_pcd.rotate(r)
            new_pcds.append(target_pcd)
        origin_pcd.paint_uniform_color((0.5, 0.1, 0.1))
        pcds.append(origin_pcd)

    o3d.visualization.draw_geometries(pcds + new_pcds)


def change_filename(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if 'Loaded' in file]
    for file in file_list:
        filename = get_filename(os.path.join(root, file)) + '.ply'
        os.rename(os.path.join(root, file), os.path.join(root, filename))


if __name__ == '__main__':
    dev_root = r'./data'
    change_filename(dev_root)
    #transform_test(dev_root)
    pcds = load_pcds(dev_root)
    #pcds = load_meshes(dev_root)
    for name, pcd in pcds.items():
        pcd = get_largest_cluster(pcd)
        img_rgb = convert_img(pcd)
        img_bgr = img_rgb[..., ::-1]
        cv2.imwrite(os.path.join('./images', name + '.jpg'), img_bgr * 255)
        get_parts(os.path.join('./images', name + '.jpg'), name)