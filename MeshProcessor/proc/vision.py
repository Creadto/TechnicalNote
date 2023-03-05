import os
import cv2
import copy
import numpy as np
import open3d as o3d
from PIL import Image
import tensorflow as tf
from pathlib import Path
from proc.calculating import get_theta_from
from proc.clustering import get_largest_cluster
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths


# @jit(nopython=True)
# 현재 numpy version을 낮춰야 jit를 쓸듯
def draw_image(points, colors):
    y_vector = points[:, 1]
    x_vector = points[:, 2]
    img_width = x_vector.max() - x_vector.min() + 1
    img_height = y_vector.max() - y_vector.min() + 1
    backboard = np.zeros((int(img_height), int(img_width), 4))
    backboard[:, :, 1] = 0.6

    y_max = y_vector.max()
    y_min = y_vector.min()
    x_min = x_vector.min()
    for idx in range(len(points)):
        y = y_max - y_vector[idx] + y_min
        x = x_vector[idx] + x_min
        for x_o in range(-1, 2):
            for y_o in range(-1, 2):
                if y_min < (y + y_o) < y_max and x_min < (x + x_o) < img_width:
                    if backboard[y + y_o, x + x_o, 1] == 0.6:
                        backboard[y + y_o, x + x_o, 0:3] = colors[idx, :]
                        backboard[y + y_o, x + x_o, 3:5] = points[idx, 0]
    return backboard


def convert_img(pcd: o3d.geometry.PointCloud, resolution: int = 500, padding: float = 0.3) -> np.array:
    """
    :param pcd: a single PointCloud object
    :param resolution: means a distance from pixel to neighbor pixel meter per resolution
    :param padding: means a white space of generated image by meter
    :return:
    """
    target_object = copy.deepcopy(pcd)
    # target_object = get_largest_cluster(target_object)
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

    # Draw image
    points = np.asarray(target_object.points)
    norm_points = (points + padding / 2.0) * resolution
    norm_points = norm_points.round()

    image_rgb = draw_image(norm_points.astype(np.uint), np.asarray(target_object.colors))

    return image_rgb[:, :, 0:3], image_rgb[:, :, 3]


def get_segmentation(pcds, config, without=None):
    if without is None:
        without = []
    result = {'images': dict(), 'masks': dict(), 'pcds': dict(), 'depth': dict()}
    for name, pcd in pcds.items():
        img_rgb, depth = convert_img(pcd, resolution=config['image']['resolution'], padding=0.0)
        if name not in without:
            img_bgr = img_rgb[..., ::-1]
            cv2.imwrite(os.path.join(config['path']['data_path'], 'images', name + '.jpg'), img_bgr * 255)

        img_rgb = img_rgb * 255
        img_rgb = img_rgb.astype(np.uint8)
        image = cv2pil(img_rgb)
        # setup input and output paths
        seg_path = Path(config['path']['seg_path'])
        seg_path.mkdir(parents=True, exist_ok=True)
        seg_path = os.path.join(seg_path, name)

        mask = run_segmentation(seg_path, image, config['image']['custom_colors'], config['image']['part_labels'])

        result['images'][name] = img_rgb
        result['masks'][name] = mask
        result['pcds'][name] = pcd
        result['depth'][name] = depth
    result['res'] = config['image']['resolution']
    return result


def run_segmentation(path, image, colors, labels):
    # load model (once)
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_RESNET50_FLOAT_STRIDE_16
    ))

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    result = bodypix_model.predict_single(image_array)

    # simple mask
    mask = result.get_mask(threshold=0.25)

    # colored mask (separate colour for each body part)
    colored_mask = result.get_colored_part_mask(mask, colors, labels)
    tf.keras.preprocessing.image.save_img(
        path + 'colored-mask.jpg',
        colored_mask
    )
    blend = cv2.addWeighted(image_array.copy(), 0.5, colored_mask, 0.5, 0, dtype=cv2.CV_32F)
    tf.keras.preprocessing.image.save_img(
        path + 'blended-mask.jpg',
        blend
    )

    # # poses
    # from tf_bodypix.draw import draw_poses  # utility function using OpenCV
    #
    # poses = result.get_poses()
    # image_with_poses = draw_poses(
    #     image_array.copy(),  # create a copy to ensure we are not modifing the source image
    #     poses,
    #     keypoints_color=(255, 100, 100),
    #     skeleton_color=(100, 100, 255)
    # )
    # tf.keras.preprocessing.image.save_img(
    #     f'{output_path}/' + name + 'total-mask.jpg',
    #     image_with_poses
    # )

    return colored_mask


def cv2pil(cv2image):
    return Image.fromarray(cv2image)


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
