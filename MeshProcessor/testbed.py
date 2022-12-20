import os
import copy
import numpy as np
from util.files import load_pcds, load_meshes, get_filename, get_position, load_ply, make_ply_header, write_tri_mesh
from proc.preprocessing import convert_img
from proc.meshing import gen_tri_mesh
from proc.clustering import get_largest_cluster, get_parts
import open3d as o3d
import cv2


def mesh_test(root):
    pcds = load_pcds(path=root)
    for name, pcd in pcds.items():
        mesh = gen_tri_mesh(pcd)
        write_tri_mesh(mesh, name + 'mesh.ply', root)


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


def matching_parts(**result):
    new_points = [0, 0, 0, 0, 0, 0, 0]
    total_model = []
    custom_colors = [
        (110, 64, 170), (110, 64, 170), (178, 60, 178), (178, 60, 178),
        (238, 67, 149), (238, 67, 149), (255, 94, 99), (255, 94, 99),
        (255, 140, 56), (255, 140, 56), (217, 194, 49), (194, 219, 64),
        (175, 240, 91), (175, 240, 91), (96, 247, 96), (96, 247, 96),
        (40, 234, 141), (40, 234, 141), (26, 199, 194), (26, 199, 194),
        (47, 150, 224), (47, 150, 224), (84, 101, 214), (99, 81, 195)
    ]
    part_labels = [
        'leftFace', 'rightFace', 'leftUpperArmBack', 'leftUpperArmFront',
        'rightUpperArmBack', 'rightUpperArmFront', 'leftLowerArmBack', 'leftLowerArmFront',
        'rightLowerArmBack', 'rightLowerArmFront', 'leftHand', 'rightHand',
        'torsoFront', 'torsoBack', 'leftUpperLegFront', 'leftUpperLegBack',
        'rightUpperLegFront', 'rightUpperLegBack', 'leftLowerLegFront', 'leftLowerLegBack',
        'rightLowerLegFront', 'rightLowerLegBack', 'leftFeet', 'rightFeet'
    ]
    labels = ['Face', 'Arm', 'Hand', 'torso', 'Leg', 'Feet']

    images = copy.deepcopy(result['images'])
    pcds = copy.deepcopy(result['pcds'])
    masks = copy.deepcopy(result['masks'])
    depth_xy = copy.deepcopy(result['depth'])
    resolution = result['res']

    name_flow = ['front', 'left', 'back', 'right']
    origin_location = np.array([0.0, 0.0, 0.0])
    # y 값으로 깊이를 매칭하고 green으로 추출한다.
    for label in labels:
        vertex_xyzrgb = []
        for name in name_flow:
            mask = masks[name][:, :, 1]
            xz_set = depth_xy[name]
            image = images[name]

            surface_x = np.array([], dtype=np.int64)
            surface_y = np.array([], dtype=np.int64)
            for color, part_label in zip(custom_colors, part_labels):
                if label not in part_label:
                    continue
                # 부위 구하기
                part_x, part_y = np.where(mask == color[1])
                surface_x = np.concatenate([surface_x, part_x])
                surface_y = np.concatenate([surface_y, part_y])
            # 포인트가 없는 인덱스는 제외
            real_x, real_y, real_z, rgb = [], [], [], []
            for x, y in zip(surface_x, surface_y):
                if image[x, y, 0] != 0.0:
                    real_x.append(x)
                    real_y.append(y)
                    xz = xz_set[x, y, :]
                    # depth 구하기
                    gap = min(xz[0], xz[1]) + (abs(xz[0] - xz[1]) / 2.0)
                    depth = np.linalg.norm(xz - gap) / resolution
                    real_z.append(depth)
                    rgb.append(np.array(image[x, y, :] * 255, dtype=np.uint8))
            real_x, real_y, real_z = np.array(real_x), np.array(real_y), np.array(real_z)

            real_x = mask.shape[0] - real_x
            # 얘가 x, y값이 됨(포인트 클라우드 좌표)
            real_x = real_x / resolution
            real_x = real_x - real_x.min()
            origin_index = np.where(real_x <= (15.0 * (1.0 / resolution)))

            real_y = real_y / resolution
            origin_y = real_y[origin_index].min()
            real_y = real_y - origin_y
            next_y = real_y[origin_index].max()

            real_z -= real_z.min()
            real_z *= 0.0

            # tuple 만들기 x, y, z, red, green, blue, alpha
            if name == 'left':
                real_y, real_z = real_z, -1 * real_y
            elif name == 'back':
                real_y, real_z = -1 * real_y, -1 * real_z
            elif name == 'right':
                real_y, real_z = -1 * real_z, real_y
            else:
                origin_location[0] = real_x.min()
                origin_location[1] = next_y
                origin_location[2] = real_z[real_y[origin_index].argmin()]

            real_x += origin_location[0]
            real_y += origin_location[1]
            real_z += origin_location[2]

            vertex_xyzrgb.append((real_y, real_x, real_z, rgb))
        # 새로운 ply 양식 만들기
        make_ply_header(path='./images/' + label + '.ply',
                        vertex=vertex_xyzrgb,
                        comment=label)
        from util.visualization import draw_geometries
        pcd = load_ply(root='./images/', filename=label + '.ply', cam_loc=None)
        draw_geometries(pcd)


def main():
    dev_root = r'./data'
    proc_result = {'images': dict(), 'masks': dict(), 'pcds': dict(), 'depth': dict()}
    change_filename(dev_root)
    # transform_test(dev_root)
    pcds = load_pcds(dev_root)
    # pcds = load_meshes(dev_root)
    for name, pcd in pcds.items():
        pcd = get_largest_cluster(pcd)
        img_rgb, depth_xz = convert_img(pcd)
        img_bgr = img_rgb[..., ::-1]
        cv2.imwrite(os.path.join('./images', name + '.jpg'), img_bgr * 255)
        mask = get_parts(os.path.join('./images', name + '.jpg'), name)

        proc_result['images'][name] = img_rgb
        proc_result['masks'][name] = mask
        proc_result['pcds'][name] = pcd
        proc_result['depth'][name] = depth_xz
    proc_result['res'] = 500
    matching_parts(**proc_result)


if __name__ == '__main__':
    main()
