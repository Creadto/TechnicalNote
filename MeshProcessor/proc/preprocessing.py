import copy
from numba import jit
from proc.calculating import get_theta_from
from proc.clustering import get_largest_cluster
import numpy as np
import open3d as o3d


#@jit(nopython=True)
# 현재 numpy version을 낮춰야 jit를 쓸듯
def draw_image(points, colors):
    y_vector = points[:, 1]
    x_vector = points[:, 2]
    img_width = x_vector.max() - x_vector.min() + 1
    img_height = y_vector.max() - y_vector.min() + 1
    backboard = np.zeros((int(img_height), int(img_width), 4))
    backboard[:, :, 1] = 1.0

    y_max = y_vector.max()
    y_min = y_vector.min()
    x_min = x_vector.min()
    for idx in range(len(points)):
        y = y_max - y_vector[idx] + y_min
        x = x_vector[idx] + x_min
        # y_min = int(y-2)
        # y_max = int(y+3)
        # x_min = int(z-2)
        # x_max = int(z+3)
        backboard[y, x, 0:3] = colors[idx, :]
        backboard[y, x, 3:5] = points[idx, 0]
    return backboard


def convert_img(pcd: o3d.geometry.PointCloud, resolution: int = 250, padding: float = 0.3) -> np.array:
    """
    :param pcd: a single PointCloud object
    :param resolution: means a distance from pixel to neighbor pixel meter per resolution
    :param padding: means a white space of generated image by meter
    :return:
    """
    target_object = copy.deepcopy(pcd)
    target_object = get_largest_cluster(target_object)
    min_bound = target_object.get_min_bound()
    target_object = target_object.translate((-1 * min_bound[0], -1 * min_bound[1], -1 * min_bound[2]))

    # 동일한 x z 중에서 제일 작은 y를 구하는 것 제일 멀리있는 x가 기준
    points = np.asarray(target_object.points)
    points = np.round(points, 2)
    sorted_index = points[:, 0].argsort()
    sorted_point = points[sorted_index, :]

    center_bound = target_object.get_max_bound() / 2.0
    half_index = np.where(sorted_point[:, 0] <= center_bound[0])[0]
    half_points = sorted_point[half_index, :]

    max_index = np.where(half_points[:, 0] == half_points[:, 0].max())[0]
    max_points = half_points[max_index, :]
    min_pivot = int(len(half_points) * 0.4)
    min_index = np.where(half_points[:, 0] < half_points[min_pivot, 0])[0]
    min_points = half_points[min_index, :]

    # (0, min(y_0)) 와 (hx, min(y_hx)) 간의 각도 구하기
    theta = get_theta_from([0, min_points[:, 0].max(), min_points[:, 1].min()],
                           [0, max_points[:, 0].max(), max_points[:, 1].min()])

    # rotation
    r = target_object.get_rotation_matrix_from_xyz((0, 0, -1 * theta))
    target_object = target_object.rotate(r)

    # 다시 0, 0, 0 정렬
    min_bound = target_object.get_min_bound()
    target_object = target_object.translate((-1 * min_bound[0], -1 * min_bound[1], -1 * min_bound[2]))

    coordi = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([coordi, target_object])
    # Draw image
    points = np.asarray(target_object.points)
    norm_points = (points + padding / 2.0) * resolution
    norm_points = norm_points.round()

    image_rgb = draw_image(norm_points.astype(np.uint), np.asarray(target_object.colors))

    return image_rgb[:, :, 0:3], image_rgb[:, :, 3]


# region measurement
def get_height(pcd: o3d.geometry.PointCloud):
    min_y = pcd.get_min_bound()[1]
    max_y = pcd.get_max_bound()[1]
    return max_y - min_y


def get_width(pcd: o3d.geometry.PointCloud):
    min_z = pcd.get_min_bound()[2]
    max_z = pcd.get_max_bound()[2]
    return max_z - min_z
# endregion
