import copy
import os
import cv2
from pathlib import Path
import tensorflow as tf
from tf_bodypix.api import download_model, load_model, BodyPixModelPaths
import numpy as np
import open3d as o3d
from scipy.stats import mode
from sklearn.cluster import KMeans


def get_segmentation(path, image, color, label):
    # load model (once)
    bodypix_model = load_model(download_model(
        BodyPixModelPaths.MOBILENET_RESNET50_FLOAT_STRIDE_16
    ))

    image_array = tf.keras.preprocessing.image.img_to_array(image)
    result = bodypix_model.predict_single(image_array)

    # simple mask
    mask = result.get_mask(threshold=0.25)

    # colored mask (separate colour for each body part)
    colored_mask = result.get_colored_part_mask(mask, color, label)
    tf.keras.preprocessing.image.save_img(
        os.path.join(path, 'colored-mask.jpg'),
        colored_mask
    )
    blend = cv2.addWeighted(image_array.copy(), 0.5, colored_mask, 0.5, 0, dtype=cv2.CV_32F)
    tf.keras.preprocessing.image.save_img(
        os.path.join(path, 'blended-mask.jpg'),
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


def get_parts2(path, name, color, label):
    return NotImplementedError


def get_parts(path, name):
    rainbow_colors = [
        (110, 64, 170), (143, 61, 178), (178, 60, 178), (210, 62, 167),
        (238, 67, 149), (255, 78, 125), (255, 94, 99), (255, 115, 75),
        (255, 140, 56), (239, 167, 47), (217, 194, 49), (194, 219, 64),
        (175, 240, 91), (135, 245, 87), (96, 247, 96), (64, 243, 115),
        (40, 234, 141), (28, 219, 169), (26, 199, 194), (33, 176, 213),
        (47, 150, 224), (65, 125, 224), (84, 101, 214), (99, 81, 195)
    ]
    custom_colors = [
        (110, 64, 170), (110, 64, 170), (178, 60, 178), (178, 60, 178),
        (238, 67, 149), (238, 67, 149), (255, 94, 99), (255, 94, 99),
        (255, 140, 56), (255, 140, 56), (255, 94, 99), (255, 140, 56),
        (175, 240, 91), (175, 240, 91), (96, 247, 96), (96, 247, 96),
        (40, 234, 141), (40, 234, 141), (26, 199, 194), (26, 199, 194),
        (47, 150, 224), (47, 150, 224), (26, 199, 194), (47, 150, 224)
    ]
    part_labels = [
        'leftFace', 'rightFace', 'leftUpperArmBack', 'leftUpperArmFront',
        'rightUpperArmBack', 'rightUpperArmFront', 'leftLowerArmBack', 'leftLowerArmFront',
        'rightLowerArmBack', 'rightLowerArmFront', 'leftHand', 'rightHand',
        'torsoFront', 'torsoBack', 'leftUpperLegFront', 'leftUpperLegBack',
        'rightUpperLegFront', 'rightUpperLegBack', 'leftLowerLegFront', 'leftLowerLegBack',
        'rightLowerLegFront', 'rightLowerLegBack', 'leftFeet', 'rightFeet'
    ]

    # local_input_path = tf.keras.utils.get_file(origin=input_url)
    local_input_path = path
    print(local_input_path)

    # setup input and output paths
    output_path = Path('./images/parts')
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_path, name)

    # get prediction result
    image = tf.keras.preprocessing.image.load_img(local_input_path)
    return get_segmentation(output_path, image, custom_colors, part_labels)


def get_largest_cluster(ply: o3d.geometry.PointCloud):
    labels = np.empty
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            ply.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    # the largest cluster label
    largest_label = mode(labels, keepdims=False).mode.item()
    largest_idx = np.where(labels == largest_label)[0]
    largest_cluster = ply.select_by_index(largest_idx.tolist())
    return largest_cluster


def separate_skin_cloth(ply: o3d.geometry.PointCloud):
    colors = np.asarray(ply.colors)
    colors = np.array((colors * 255.0), dtype=np.uint)
    model = KMeans(init='k-means++', n_clusters=2, n_init=10)
    model.fit(colors)
    # 군집화 모델 학습 및 클러스터 예측 결과 반환
    labels = model.labels_
    parts = []
    for label in range(labels.max() + 1):
        part_idx = np.where(labels == label)[0]
        part = ply.select_by_index(part_idx.tolist())
        print(len(part.points))
        parts.append(part)

    body_parts = {'body': None, 'arm': None, 'head': None}
    if len(parts[0].points) > len(parts[1].points):
        body_parts['body'] = parts[0]
        target_part: o3d.geometry.PointCloud = parts[1]
    else:
        body_parts['body'] = parts[1]
        target_part: o3d.geometry.PointCloud = parts[0]

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            target_part.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
    arm_label = mode(labels, keepdims=False).mode.item()
    arm_idx = np.where(labels == arm_label)[0]
    body_parts['arm'] = target_part.select_by_index(arm_idx.tolist())
    head_idx = np.where(labels != arm_label)[0]
    body_parts['head'] = target_part.select_by_index(head_idx.tolist())

    return body_parts
