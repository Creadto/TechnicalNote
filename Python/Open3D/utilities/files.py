import os
import open3d as o3d
import numpy as np
from utilities.yaml_config import YamlConfig


def get_yaml(path):
    return YamlConfig.get_dict(path)


def load_ply(root, filename, cam_loc):
    pcd = o3d.io.read_point_cloud(os.path.join(root, filename))
    n_of_points = len(pcd.points)
    pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    if cam_loc is not None:
        for key in cam_loc.keys():
            if key in filename:
                transform = cam_loc[key]
                location = transform[-1][0:3]
                pcd.orient_normals_towards_camera_location(np.array(location))
                pcd.transform(np.identity(4))
                break
    return pcd, n_of_points


def load_pcds(path, cam_loc=None):
    pcds = []
    file_list = os.listdir(path)
    file_list = [file for file in file_list if '.ply' in file]
    for ply in file_list:
        pcd, _ = load_ply(root=path, filename=ply, cam_loc=cam_loc)
        pcds.append(pcd)
    return pcds


def clone_ply(ply, root, filename):
    o3d.io.write_point_cloud(os.path.join(root, filename), ply, write_ascii=True)
    return os.path.join(root, filename)


def trim_ply(ply, begin, end, root, filename):
    ascii_path = clone_ply(ply, root, filename)
    f = open(ascii_path, 'r')
    new_f = open(ascii_path.replace('.ply', '_trim.ply'), 'w')
    row = 0
    is_header = True
    while row < end:
        line = f.readline()
        if not line:
            break
        row += 1
        if "element vertex" in line:
            line = "element vertex " + str(end - begin) + "\n"
        if row > begin or is_header:
            new_f.write(line)
        if "end_header" in line:
            end += row
            begin += row
            is_header = False
    f.close()
    new_f.close()
