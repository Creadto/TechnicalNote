import os
import cv2
import copy
import numba
import numpy as np
import open3d as o3d
from PIL import Image
import tensorflow as tf
from pathlib import Path
from proc.calculating import get_theta_from
from proc.clustering import get_largest_cluster
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths


class VisionProcessor:
    def __init__(self, config):
        # utility
        self.data_path = config['path']['data_path']
        self.save_path = config['path']['seg_path']
        # for segmentation
        self.colors = config['image']['custom_colors']
        self.labels = config['image']['part_labels']
        self.seg_model = load_model(download_model(BodyPixModelPaths.MOBILENET_RESNET50_FLOAT_STRIDE_16))
        # for image processing
        self.resolution = config['image']['resolution']
        self.offset = config['image']['offset']
        self.padding = config['image']['padding']

    def get_info_from_pcds(self, pcds, without=None):
        if without is None:
            without = []
        result = {'images': dict(), 'masks': dict(), 'pcds': dict(), 'depth': dict()}
        for name, pcd in pcds.items():
            result['pcds'][name] = pcd
            if "face" in name:
                continue
            img_rgb, depth = self.__convert_img(pcd)
            if name not in without:
                img_bgr = img_rgb[..., ::-1]
                cv2.imwrite(os.path.join(self.data_path, 'images', name + '.jpg'), img_bgr * 255)

            img_rgb = img_rgb * 255
            img_rgb = img_rgb.astype(np.float32)
            # setup input and output paths
            seg_path = Path(self.save_path)
            seg_path.mkdir(parents=True, exist_ok=True)
            seg_path = os.path.join(seg_path, name)

            mask = self.run_segmentation(seg_path, img_rgb)

            result['images'][name] = img_rgb
            result['masks'][name] = mask
            result['depth'][name] = depth
        result['res'] = self.resolution
        result['custom_colors'] = self.colors
        result['part_labels'] = self.labels
        return result

    def run_segmentation(self, path, image):
        # load model (once)
        image_array = image
        result = self.seg_model.predict_single(image_array)

        # simple mask
        mask = result.get_mask(threshold=0.25)

        # colored mask (separate colour for each body part)
        colored_mask = result.get_colored_part_mask(mask, self.colors, self.labels)
        tf.keras.preprocessing.image.save_img(
            path + 'colored-mask.jpg',
            colored_mask
        )
        blend = cv2.addWeighted(image_array.copy(), 0.5, colored_mask, 0.5, 0, dtype=cv2.CV_32F)
        tf.keras.preprocessing.image.save_img(
            path + 'blended-mask.jpg',
            blend
        )
        return colored_mask

    @staticmethod
    @numba.jit
    def __draw_image2(points, colors):
        y_vector = points[:, 1]
        x_vector = points[:, 2]
        img_height = y_vector.max() + 1
        img_width = int(img_height * 0.75)
        lsb = int((x_vector.max() - img_width) / 2.0)
        backboard = np.zeros((int(img_height), int(img_width), 4))
        backboard[:, :, 1] = 0.6

        y_max = y_vector.max()
        x_vector -= lsb
        for idx in range(len(points)):
            y = y_max - y_vector[idx]
            x = x_vector[idx]
            for x_o in range(-2, 3):
                for y_o in range(-2, 3):
                    if 0 < (y + y_o) < img_height and 0 < (x + x_o) < img_width:
                        if backboard[y + y_o, x + x_o, 1] == 0.6:
                            backboard[y + y_o, x + x_o, 0:3] = colors[idx, :]
                            backboard[y + y_o, x + x_o, 3:5] = points[idx, 0]
        return backboard

    @staticmethod
    @numba.jit
    def __draw_image(points, colors):
        y_max = points[:, 1].max()
        img_height = y_max + 1
        img_width = int(img_height * 0.75)
        backboard = np.zeros((int(img_height), int(img_width), 4))
        backboard[:, :, 1] = 0.6

        x_vector = points[:, 2] - int((points[:, 2].max() - img_width) / 2.0)
        for idx in range(len(points)):
            y = y_max - points[:, 1][idx]
            x = x_vector[idx]
            for x_o in range(-2, 3):
                for y_o in range(-2, 3):
                    if 0 < (y + y_o) < img_height and 0 < (x + x_o) < img_width:
                        if backboard[y + y_o, x + x_o, 1] == 0.6:
                            backboard[y + y_o, x + x_o, 0:3] = colors[idx, :]
                            backboard[y + y_o, x + x_o, 3:5] = points[idx, 0]
        return backboard

    def __convert_img(self, pcd: o3d.geometry.PointCloud) -> np.array:
        """
        :param pcd: a single PointCloud object
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

        # 다시 0, 0, 0 정렬 and Cut
        min_bound = target_object.get_min_bound()
        target_object = target_object.translate((-1 * min_bound[0], -1 * min_bound[1], -1 * min_bound[2]))
        max_bound = target_object.get_max_bound()
        cut_bbox = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=(0, 0, 0),
            max_bound=(0.5, max_bound[1], max_bound[2]))
        target_object = target_object.crop(cut_bbox)
        min_bound = target_object.get_min_bound()
        target_object = target_object.translate((-1 * min_bound[0], -1 * min_bound[1], -1 * min_bound[2]))
        # Check
        coordi = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # o3d.visualization.draw_geometries([coordi, target_object])
        # Draw image
        points = np.asarray(target_object.points)
        norm_points = (points + self.padding / 2.0) * self.resolution
        norm_points = norm_points.round()

        image_rgb = self.__draw_image(norm_points.astype(np.uint), np.asarray(target_object.colors))

        return image_rgb[:, :, 0:3], image_rgb[:, :, 3]


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
