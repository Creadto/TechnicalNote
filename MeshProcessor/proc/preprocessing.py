import copy
from numba import jit
import numpy as np
import open3d as o3d


#@jit(nopython=True)
# 현재 numpy version을 낮춰야 jit를 쓸듯
def draw_image(backboard, points, colors):
    for idx in  range(len(points)):
        y = points[idx, 1]
        xz = np.array(points[idx, (0, 2)])
        x = int(np.linalg.norm(xz))
        y_min = int(y-2)
        y_max = int(y+3)
        x_min = int(x-2)
        x_max = int(x+3)
        backboard[y_min:y_max, x_min:x_max, 0:3] = colors[idx, :]
        backboard[y_min:y_max, x_min:x_max, 3:5] = xz
    return backboard


def convert_img(pcd: o3d.geometry.PointCloud, resolution: int = 500, padding: float = 0.3) -> np.array:
    """
    :param pcd: a single PointCloud object
    :param resolution: means a distance from pixel to neighbor pixel meter per resolution
    :param padding: means an white space of generated image by meter
    :return:
    """
    target_object = copy.deepcopy(pcd)

    # get whole section
    width = get_width(pcd)
    height = get_height(pcd)
    # make empty image
    img_width = (width + padding) * resolution
    img_height = (height + padding) * resolution
    image_array = np.zeros((int(img_height), int(img_width), 5))
    image_array[:, :, 1] = 0.6

    # translate min_bound to (0, 0, 0)
    points = np.asarray(target_object.points)
    norm_points = points - target_object.get_min_bound()

    # Draw image
    norm_points = (norm_points + padding / 2.0) * resolution
    norm_points = norm_points.round()
    image_rgb = draw_image(image_array, norm_points.astype(np.uint), np.asarray(target_object.colors))

    return np.rot90(image_rgb[:, :, 0:3], 2), np.rot90(image_rgb[:, :, 3:5], 2)


# region measurement
def get_height(pcd: o3d.geometry.PointCloud):
    min_y = pcd.get_min_bound()[1]
    max_y = pcd.get_max_bound()[1]
    return max_y - min_y


def get_width(pcd: o3d.geometry.PointCloud):
    min_bound = pcd.get_min_bound()
    min_bound[1] = 0.0
    max_bound = pcd.get_max_bound()
    max_bound[1] = 0.0
    return np.linalg.norm(max_bound - min_bound)
# endregion
