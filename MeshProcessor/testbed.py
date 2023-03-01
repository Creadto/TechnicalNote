import os
import cv2
import copy
import time
import torch
import numpy as np
import open3d as o3d
from util.yaml_config import YamlConfig
from proc.meshing import gen_tri_mesh, combine_pcds
from util.files import load_pcds, get_position, load_ply, make_ply_header, write_tri_mesh


# region deprecated
def transform_test(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]

    run_flag = True

    pcds = []
    new_pcds = []
    for idx, filename in enumerate(file_list):
        cam_loc = get_position(os.path.join(root, filename))
        origin_pcd: o3d.geometry.PointCloud = load_ply(root=root, filename=filename, cam_loc=cam_loc)
        # origin_pcd = get_largest_cluster(origin_pcd)
        origin_pcd = origin_pcd.translate((0, 2.0 * idx, 0))
        offset_unit = 1.0
        for idx in range(1, 9):
            target_pcd: o3d.geometry.PointCloud = copy.deepcopy(origin_pcd)
            r = target_pcd.get_rotation_matrix_from_xyz((0, 2 * np.pi * (idx / 8), 0))
            target_pcd = target_pcd.translate((0, 0, offset_unit * idx))
            target_pcd = target_pcd.rotate(r)
            new_pcds.append(target_pcd)
        origin_pcd.paint_uniform_color((0.5, 0.1, 0.1))
        pcds.append(origin_pcd)

    o3d.visualization.draw_geometries(pcds + new_pcds)


def to_xyz(root):
    pcds = load_pcds(root)

    offset = 1.5
    new_pcds = []
    cnt = 0
    for name, pcd in pcds.items():
        lower = name.lower()
        new_pcd = copy.deepcopy(pcd)
        if "back" in lower:
            R = pcd.get_rotation_matrix_from_xyz((0, np.pi, 0))
            new_pcd.translate((offset, 0, offset))
        elif "left" in lower:
            R = pcd.get_rotation_matrix_from_xyz((0, np.pi / 2, 0))
            new_pcd.translate((-offset, 0, offset))
        elif "right" in lower:
            R = pcd.get_rotation_matrix_from_xyz((0, np.pi * 1.6, 0))
            new_pcd.translate((0.35, 0, 0.05))
        else:
            R = pcd.get_rotation_matrix_from_xyz((0, 0, 0))
            new_pcd.translate((-offset, 0, -offset))

        new_pcd = new_pcd.rotate(R)
        # if "right" in lower:
        #     # new_pcd.translate((-0.0159, 0, 0.0547))
        #     new_pcd.translate((-0.05, 0, 0.05))
        # if "front" in lower:
        #     new_pcd.translate((0.05, 0, -0.05))
        # new_pcd.transform(cam_locs[cnt].T)
        cnt += 1
        # new_pcd = get_largest_cluster(new_pcd)
        new_pcds.append(new_pcd)

    combined = combine_pcds(new_pcds, True)
    meshed = gen_tri_mesh(combined)
    write_tri_mesh(path='test_mesh.ply', mesh=meshed)
    tgt = load_ply(root, 'combined.xyz', None)
    new_pack = o3d.geometry.PointCloud()
    o3d.visualization.draw_geometries([combined])
    o3d.io.write_point_cloud(filename=os.path.join(root, 'combined.xyz'), pointcloud=combined, write_ascii=True)


# endregion

# region applied
def matching_parts(**result):
    new_points = [0, 0, 0, 0, 0, 0, 0]
    total_model = []
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
    labels = ['Face', 'Arm', 'Hand', 'torso', 'Leg', 'Feet']

    images = copy.deepcopy(result['images'])
    pcds = copy.deepcopy(result['pcds'])
    masks = copy.deepcopy(result['masks'])
    depth_xy = copy.deepcopy(result['depth'])
    resolution = result['res']

    name_flow = ['front', 'right', 'back', 'left']
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

            real_z -= real_z.min()
            # real_z *= 0.0

            # tuple 만들기 x, y, z, red, green, blue, alpha
            if name == 'left':
                real_y, real_z = -1 * real_z, -1 * real_y
            elif name == 'right':
                real_y, real_z = -1 * real_z, real_y

            real_x += origin_location[0]
            real_y += origin_location[1]
            real_z += origin_location[2]

            if name == 'front':
                next_y = real_y[origin_index].max()
                # origin_location[0] = real_x[real_y[origin_index].argmax()]
                origin_location[1] = next_y + 0.03
                origin_location[2] = real_z[real_y[origin_index].argmax()]
            elif name == 'back':
                origin_location[1] = real_y[origin_index].max() - 0.08
                origin_location[2] = real_z[real_y[origin_index].argmax()]
            elif name == 'right':
                origin_location[1] = real_y[real_z[origin_index].argmax()] - 0.13
                origin_location[2] = real_z[origin_index].max()

            vertex_xyzrgb.append((real_y, real_x, real_z, rgb))
            # 새로운 ply 양식 만들기
        make_ply_header(path='./images/' + label + '.ply',
                        vertex=vertex_xyzrgb,
                        comment=label)
        from util.visualization import draw_geometries
        pcd = load_ply(root='./images/', filename=label + '.ply', cam_loc=None)
        draw_geometries(pcd)


def classify_gender():
    import torch
    import torch.nn as nn
    from torchvision import models
    img_bgr = cv2.imread(r"D:\Creadto\TechnicalNote\MeshProcessor\data\images\front.jpg")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # device object
    save_path = os.path.join(r"D:\Creadto\TechnicalNote\MeshProcessor\models\gender_sub\gender_classifier.pth")
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # binary classification (num_of_class == 2)
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    model.to(device)

    tensor_rgb = torch.tensor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)) / 255.0
    shape = tensor_rgb.shape
    tensor_rgb = tensor_rgb.view([3, shape[0], shape[1]])
    output = model(tensor_rgb.unsqueeze(dim=0))
    gender = torch.argmax(output).item()

    in_file = YamlConfig.get_dict(os.path.join(r"D:\Creadto\TechnicalNote\MeshProcessor\config\smpl", "fit_smplx.yaml"))
    in_file['gender'] = "female" if gender == 1 else "male"
    YamlConfig.write_yaml(r"D:\Creadto\TechnicalNote\MeshProcessor\config\smpl\fit_smplx_c.yaml", in_file)
    print(output)


# endregion

# region non-module
def create_bounding_box(obb):
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]
    eight_points = np.asarray(obb.get_box_points())
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(eight_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def mesh_test(root):
    pcds = load_pcds(path=root)
    for name, pcd in pcds.items():
        mesh = gen_tri_mesh(pcd)
        write_tri_mesh(mesh, name + 'mesh.ply', root)


def get_openpose():
    inputWidth = 320
    inputHeight = 240
    inputScale = 1.0 / 255

    path = './data/images/ans.jpg'
    protoFile = "./openpose/models/pose/body_25/" + "pose_deploy.prototxt"
    weightsFile = "./openpose/models/pose/body_25/" + "pose_iter_584000.caffemodel"

    # 위의 path에 있는 network 모델 불러오기
    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    img_bgr = cv2.imread(path)
    frameWidth = img_bgr.shape[1]
    frameHeight = img_bgr.shape[0]

    inpBlob = cv2.dnn.blobFromImage(img_bgr, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    imgb = cv2.dnn.imagesFromBlob(inpBlob)

    net.setInput(inpBlob)
    output = net.forward()
    test = 1


def pre_mesh_seq():
    import subprocess
    import util.files as files
    from proc.preprocessing import convert_img
    from proc.clustering import get_parts
    from proc.calculating import measure_bodies2
    import shutil
    import time

    data_path = GlobalConfig['path']['data_path']

    folders = os.listdir(data_path)
    for folder in folders:
        if "pointclouds" not in folder:
            shutil.rmtree(os.path.join(data_path, folder))
            os.mkdir(os.path.join(data_path, folder))

    pcds = files.load_pcds(os.path.join(data_path, 'pointclouds'), imme_remove=False)
    proc_result = {'images': dict(), 'masks': dict(), 'pcds': dict(), 'depth': dict()}
    for name, pcd in pcds.items():
        img_rgb, depth = convert_img(pcd, padding=0.0)
        img_bgr = img_rgb[..., ::-1]
        cv2.imwrite(os.path.join(data_path, 'images', name + '.jpg'), img_bgr * 255)
        mask = get_parts(os.path.join(data_path, 'images', name + '.jpg'), name)

        proc_result['images'][name] = img_rgb
        proc_result['masks'][name] = mask
        proc_result['pcds'][name] = pcd
        proc_result['depth'][name] = depth

        if 'front' not in name:
            os.remove(os.path.join(data_path, 'images', name + '.jpg'))

    # measurement
    proc_result['res'] = 500
    proc_result['template'] = None
    proc_result['total_height'] = 1.77
    output = measure_bodies2(**proc_result)

    # pose estimation
    abs_path = os.path.dirname(os.path.abspath(__file__))
    subprocess.call([os.path.join(abs_path, 'script', "pose_estimation.bat")], shell=True)
    while len(os.listdir(os.path.join(data_path, 'keypoints'))) == 0:
        time.sleep(0.1)
    file_list = os.listdir(os.path.join(data_path, 'keypoints'))
    for file in file_list:
        new_name = file.replace('000000000000_', '')
        os.rename(os.path.join(data_path, 'keypoints', file),
                  os.path.join(data_path, 'keypoints', new_name))

    # make mesh file
    subprocess.call([os.path.join(abs_path, 'script', "mesh_maker.bat")], shell=True)
    # os.system(os.path.join(self.script_path, "mesh_maker.bat"))
    mesh_path = os.path.join(data_path, 'meshes', 'meshes', 'front')
    while os.path.isdir(mesh_path) is False or os.path.isfile(os.path.join(mesh_path, '000.obj')) is False:
        time.sleep(1.0)

    # read mesh file
    mesh_file = files.load_mesh(mesh_path, '000.obj')
    # mesh_file.paint_uniform_color(face_color * 2)
    mesh_file.paint_uniform_color([222.0 / 255.0, 171.0 / 255.0, 127.0 / 255.0])
    # write_tri_mesh(mesh_file, filename='Meshed.ply', path=os.path.join(data_path))
    return mesh_file, proc_result


# endregion

# region non-validated
def crop_n_attach(mesh_file, proc_result):
    head = proc_result['pcds']['face']

    # 기존 길이 알아내기
    mask = proc_result['masks']['front'][:, :, 1]
    height_x, _ = np.where(mask != 0)
    origin_height = ((height_x.max() - height_x.min()) / 500.0) + 0.032

    # 기존 길이에 맞춰 스케일링 및 정리정돈
    min_bound = mesh_file.get_min_bound()
    max_bound = mesh_file.get_max_bound()
    gap = max_bound - min_bound
    scale_ratio = origin_height / gap[1]
    mesh_file.scale(scale_ratio, center=mesh_file.get_center())
    mesh_file = mesh_file.transform(np.identity(4))
    mesh_file.compute_vertex_normals()
    min_bound = mesh_file.get_min_bound()
    max_bound = mesh_file.get_max_bound()

    mask = proc_result['masks']['back'][:, :, 1]

    r = head.get_rotation_matrix_from_xyz((0, -1 * np.pi / 2.0, 0))
    head = head.rotate(r)

    # 위치 0, 0, 0으로 맞춤
    head_min_bound = head.get_min_bound()
    mesh_file = mesh_file.translate((-1 * min_bound[0], -1 * min_bound[1], -1 * min_bound[2]))
    head = head.translate((-1 * head_min_bound[0], -1 * head_min_bound[1], -1 * head_min_bound[2]))
    head_min_bound = head.get_min_bound()
    head_max_bound = head.get_max_bound()

    # 머리를 아주 이쁘게 자를 수 있도록 바운드 박스 구성
    # 얼굴 컬러: 64(Green)
    target_color = 64
    face_row, face_col = np.where(mask == target_color)
    face_len = ((max(face_row) - min(face_row)) / 500.0) + 0.048 + 0.16
    head_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(head_min_bound[0], head_max_bound[1] - face_len, head_min_bound[2]),
        max_bound=head_max_bound)
    cut_head = head.crop(head_bbox)
    cut_min_bound = cut_head.get_min_bound()
    cut_head = cut_head.translate((-1 * cut_min_bound[0], -1 * cut_min_bound[1], -1 * cut_min_bound[2]))
    hbb = cut_head.get_axis_aligned_bounding_box()
    bbb = mesh_file.get_oriented_bounding_box()

    # atan 구하기 세우기
    points = np.asarray(bbb.get_box_points())

    from proc.calculating import get_theta_from
    x_rot = get_theta_from(points[2], points[7])
    r = cut_head.get_rotation_matrix_from_xyz((-1 * x_rot, 0, 0))
    mesh_file = mesh_file.rotate(r)
    min_bound = mesh_file.get_min_bound()
    mesh_file = mesh_file.translate((-1 * min_bound[0], -1 * min_bound[1], -1 * min_bound[2]))
    max_bound = mesh_file.get_max_bound()

    # 머리 자르기
    mask = proc_result['masks']['front'][:, :, 1]
    torso_color = 240
    _, torso_width = np.where(mask == torso_color)
    height, _ = mask.shape
    cut_smpl_min = ((face_col.min() / 500.0) - (round(torso_width.mean()) / 500.0) + (
                (torso_width.mean() - torso_width.min()) / 500 / 2.0),
                    (height - face_row.max()) / 500.0, 0)
    cut_smpl_max = (
    (face_col.max() / 500.0) - (round(torso_width.mean()) / 500.0) + ((torso_width.mean() - torso_width.min()) / 500),
    (height - face_row.min()) / 500.0, max_bound[2])
    head_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=cut_smpl_min,
                                                    max_bound=cut_smpl_max)
    head_bbox.color = [1, 0, 0]
    dummy_head = copy.deepcopy(mesh_file).crop(head_bbox)

    dummy_max_bound = dummy_head.get_max_bound()
    height_gap = (height - face_row.min()) / 500.0 - dummy_max_bound[1]
    cut_smpl_min = ((face_col.min() / 500.0) - (round(torso_width.mean()) / 500.0) + (
                (torso_width.mean() - torso_width.min()) / 500 / 2.0),
                    ((height - face_row.max()) / 500.0) - height_gap, 0)
    cut_smpl_max = (
    (face_col.max() / 500.0) - (round(torso_width.mean()) / 500.0) + ((torso_width.mean() - torso_width.min()) / 500),
    ((height - face_row.min()) / 500.0) - height_gap, max_bound[2])
    head_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=cut_smpl_min,
                                                    max_bound=cut_smpl_max)
    dummy_head = copy.deepcopy(mesh_file).crop(head_bbox)

    head_gap = dummy_head.get_max_bound() - dummy_head.get_min_bound()
    if head_gap[1] > 0.05:
        cut_head_max_bound = cut_head.get_max_bound()
        head_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, cut_head_max_bound[1] - head_gap[1], 0),
                                                        max_bound=(cut_head_max_bound[0], cut_head_max_bound[1],
                                                                   cut_head_max_bound[2]))
        cut_head = cut_head.crop(head_bbox)
    cut_min_bound = cut_head.get_min_bound()
    cut_head = cut_head.translate((-1 * cut_min_bound[0], -1 * cut_min_bound[1], -1 * cut_min_bound[2]))
    # body = copy.deepcopy(mesh_file).crop(body_bbox)
    dhbb = dummy_head.get_axis_aligned_bounding_box()
    dummy_gap = dhbb.get_max_bound() - dhbb.get_min_bound()
    hbb = cut_head.get_axis_aligned_bounding_box()
    head_gap = hbb.get_max_bound() - hbb.get_min_bound()
    gap = dummy_gap - head_gap
    head_pos = dhbb.get_min_bound()
    cut_head = cut_head.translate((head_pos[0], head_pos[1], head_pos[2] + gap[2]))

    # mesh 만들기
    # body_pcd = body.sample_points_poisson_disk(len(body.triangles))
    # target_pcd = combine_pcds([body_pcd, cut_head])
    # target_mesh = gen_tri_mesh(target_pcd)
    # mesh_taubin = target_mesh.filter_smooth_taubin(number_of_iterations=50)
    # mesh_taubin.compute_vertex_normals()
    head_mesh = gen_tri_mesh(cut_head)
    mesh_taubin = head_mesh.filter_smooth_taubin(number_of_iterations=50)
    mesh_taubin.compute_vertex_normals()
    mesh_taubin += mesh_file
    center_bound = (mesh_taubin.get_max_bound() - mesh_taubin.get_min_bound()) / 2.0
    mesh_taubin.translate((-1 * center_bound[0], (-1 * center_bound[1]) + 0.2, -1 * center_bound[2]))

    #
    # write_tri_mesh(mesh_taubin, 'temp.ply')
    # import pymeshlab
    # ms = pymeshlab.MeshSet()
    # ms.load_new_mesh('./temp.ply')
    # ms.save_current_mesh('./testbed.obj')
    # o3d.visualization.draw_geometries([mesh_taubin])
    return mesh_taubin


# endregion


def main():
    points = 3773033 + 70700 + 2590271 + 3133489 + 3487231

    print(points)
    begin = time.time()
    mesh_file, proc_result = pre_mesh_seq()
    crop_n_attach(mesh_file, proc_result)
    end = time.time()
    print(f"{end - begin:.5f} sec")


if __name__ == '__main__':
    GlobalConfig = YamlConfig.get_dict(r'config/default.yaml')
    GlobalConfig = GlobalConfig['default']
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
