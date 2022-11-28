import utilities.files as files
import numpy as np


def get_height(**kwargs):
    if "path" in kwargs:
        path = kwargs['path']
        cam_loc = files.get_position(path)
        pcd, _ = files.load_ply(root='', filename=path, cam_loc=cam_loc)
    else:
        pcd = kwargs['front_pcd']

    points = np.asarray(pcd.points)
    y_values = points[:, 1]
    height = abs(y_values.max() - y_values.min())
    return height


def get_width(**kwargs):
    if "path" in kwargs:
        path = kwargs['path']
        cam_loc = files.get_position(path)
        pcd, _ = files.load_ply(root='', filename=path, cam_loc=cam_loc)
    else:
        pcd = kwargs['front_pcd']

    points = np.asarray(pcd.points)
    y_values = points[:, 2]
    height = abs(y_values.max() - y_values.min())
    return height


if __name__ == '__main__':
    get_height(path='../data/Test.ply')