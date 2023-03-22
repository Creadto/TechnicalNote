import os
import time
import copy
from util.yaml_config import YamlConfig


# region deprecated

# endregion

# region applied
def matching_parts(**result):

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
    masks = copy.deepcopy(result['masks'])
    depth_xy = copy.deepcopy(result['depth'])
    resolution = result['res']

    name_flow = ['front', 'right', 'back', 'left']
    import numpy as np
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
        from util.files import make_ply_header
        make_ply_header(path='./images/' + label + '.ply',
                        vertex=vertex_xyzrgb,
                        comment=label)
        from util.visualization import draw_geometries
        from util.files import load_ply
        pcd = load_ply(root='./images/', filename=label + '.ply')
        draw_geometries(pcd)


def classify_gender():
    import torch
    import torch.nn as nn
    from torchvision import models
    import cv2

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
def pre_mesh_seq():
    global_config = YamlConfig.get_dict(r'config/default.yaml')
    global_config = global_config['default']
    data_path = global_config['path']['data_path']
    from util.files import change_filename
    change_filename(os.path.join(data_path, "pointclouds"))

    from util.files import clean_folder, load_pcds
    clean_folder(data_path, ["pointclouds"])
    pcds = load_pcds(os.path.join(data_path, 'pointclouds'), imme_remove=False)

    from proc.vision import VisionProcessor
    processor = VisionProcessor(global_config)
    proc_result = processor.get_info_from_pcds(pcds=pcds, without=['back', 'face', 'left', 'right'])
    # only for images from camera
    # from proc.vision import run_segmentation
    # from PIL import Image
    # for key in pcds.keys():
    #     if "face" in key:
    #         continue
    #     im_path = r"D:\Creadto\TechnicalNote\MeshProcessor\storage\230307"
    #     im_path = os.path.join(im_path, key + '.jpg')
    #     image = Image.open(im_path)
    #     _ = run_segmentation(os.path.join(GlobalConfig['path']['seg_path'], key + 'm'), image, GlobalConfig['image']['custom_colors'], GlobalConfig['image']['part_labels'])

    # measurement
    from proc.calculating import measure_bodies2
    proc_result['measure'] = measure_bodies2(**proc_result)
    YamlConfig.write_yaml('./Measured.yaml', proc_result['measure'])

    # pose estimation
    import subprocess
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
    from util.files import load_mesh
    mesh_file = load_mesh(mesh_path, '000.obj')
    # mesh_file.paint_uniform_color(face_color * 2)
    mesh_file.paint_uniform_color([222.0 / 255.0, 171.0 / 255.0, 127.0 / 255.0])
    # write_tri_mesh(mesh_file, filename='Meshed.ply', path=os.path.join(data_path))
    return mesh_file, proc_result


# endregion

# region non-validated
def crop_n_attach(mesh_file, proc_result):
    import numpy as np
    import open3d as o3d
    from proc.meshing import gen_tri_mesh

    global_config = YamlConfig.get_dict(r'config/default.yaml')
    global_config = global_config['default']

    res = global_config['image']['resolution']
    head = proc_result['pcds']['face']
    # 기존 길이 알아내기
    mask = proc_result['masks']['front'][:, :, 1]
    height_x, _ = np.where(mask != 0)
    origin_height = ((height_x.max() - height_x.min()) / res) + 0.032
    # 기존 길이에 맞춰 스케일링 및 정리정돈
    min_bound = mesh_file.get_min_bound()
    max_bound = mesh_file.get_max_bound()
    gap = max_bound - min_bound
    scale_ratio = origin_height / gap[1]
    mesh_file.scale(scale_ratio, center=mesh_file.get_center())
    mesh_file = mesh_file.transform(np.identity(4))
    mesh_file.compute_vertex_normals()
    min_bound = mesh_file.get_min_bound()

    mask = proc_result['masks']['front'][:, :, 1]

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
    face_len = ((max(face_row) - min(face_row)) / res)
    head_bbox = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=(head_min_bound[0], head_max_bound[1] - face_len, head_min_bound[2]),
        max_bound=head_max_bound)
    cut_head = head.crop(head_bbox)
    cut_min_bound = cut_head.get_min_bound()
    cut_head = cut_head.translate((-1 * cut_min_bound[0], -1 * cut_min_bound[1], -1 * cut_min_bound[2]))
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
    _, torso_col = np.where(mask == torso_color)
    height, _ = mask.shape # ***** 이거 왜 이렇게 값이 높지?
    # 머리의 간격에서 torso의 4/1씩 오프셋으로 양쪽을 자름
    torso_q_width = (torso_col.max() - torso_col.min()) / res / 4.0
    cut_smpl_min = ((face_col.min() / res) - torso_q_width,
                    (height - face_row.max()) / res, 0)
    cut_smpl_max = ((face_col.max() / res) + torso_q_width,
                    (height - face_row.min()) / res, max_bound[2])
    # 그리고 smplx의 center로 축을 이동함
    cut_smpl_center = (cut_smpl_max[0] - cut_smpl_min[0]) / 2.0
    mf_abb = mesh_file.get_axis_aligned_bounding_box()
    mf_center = mf_abb.get_center()
    cut_smpl_min = (mf_center[0] - cut_smpl_center, cut_smpl_min[1], cut_smpl_min[2])
    cut_smpl_max = (mf_center[0] + cut_smpl_center, cut_smpl_max[1], cut_smpl_max[2])

    # 자름
    head_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=cut_smpl_min,
                                                    max_bound=cut_smpl_max)
    head_bbox.color = [1, 0, 0]
    dummy_head = copy.deepcopy(mesh_file).crop(head_bbox)

    # Fit한 헤드를 추출
    dummy_max_bound = dummy_head.get_max_bound()
    height_gap = (height - face_row.min()) / res - dummy_max_bound[1]
    cut_smpl_min = (cut_smpl_min[0], cut_smpl_min[1] - height_gap, cut_smpl_min[2])
    cut_smpl_max = (cut_smpl_max[0], cut_smpl_max[1] - height_gap, cut_smpl_max[2])
    head_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=cut_smpl_min,
                                                    max_bound=cut_smpl_max)
    dummy_head = copy.deepcopy(mesh_file).crop(head_bbox)

    head_gap = dummy_head.get_max_bound() - dummy_head.get_min_bound()
    # 이거 왜 있더라
    if head_gap[1] > 0.05:
        cut_head_max_bound = cut_head.get_max_bound()
        head_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(0, cut_head_max_bound[1] - head_gap[1], 0),
                                                        max_bound=(cut_head_max_bound[0], cut_head_max_bound[1],
                                                                   cut_head_max_bound[2]))
        cut_head = cut_head.crop(head_bbox)

    # 안면만 떼어내는 작업(cut_head to cut_face)
    chbb = cut_head.get_axis_aligned_bounding_box()
    face_min_bound = (chbb.get_min_bound()[0], chbb.get_min_bound()[1], chbb.get_max_bound()[2] * 0.75 )#GlobalConfig['mesh']['face_depth'])
    face_gap = (chbb.get_max_bound()[1] - chbb.get_min_bound()[1]) * GlobalConfig['mesh']['face_height']
    face_max_bound = (chbb.get_max_bound()[0], chbb.get_min_bound()[1] + face_gap, chbb.get_max_bound()[2])
    face_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=face_min_bound, max_bound=face_max_bound)
    face_bbox.color = [1, 0, 0]
    o3d.visualization.draw_geometries([cut_head, face_bbox])
    face = cut_head.crop(face_bbox)

    # max bound의 z값과 min bound z값의 차이정도를 smplx dummy head box에 적용하여 face를 추출해야함
    face_depth = face_bbox.get_max_bound()[2] - face_bbox.get_min_bound()[2]
    dhbb = dummy_head.get_axis_aligned_bounding_box()
    sf_min_bound = (dhbb.get_min_bound()[0], dhbb.get_min_bound()[1], dhbb.get_max_bound()[2] - face_depth)
    sf_max_bound = (dhbb.get_max_bound()[0], dhbb.get_min_bound()[1] + face_gap, dhbb.get_max_bound()[2])
    sf_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=sf_min_bound, max_bound=sf_max_bound)
    sf_bbox.color = [0, 1, 0]
    o3d.visualization.draw_geometries([dummy_head, sf_bbox])
    smplx_face = dummy_head.crop(sf_bbox)

    # x, y scale을 조정 (z 축은 건드리면 안돼서 manual 조정)
    sfbb = smplx_face.get_axis_aligned_bounding_box()
    sfbb.color = [0, 1, 1]
    fbb = face.get_axis_aligned_bounding_box()
    ratio = (sfbb.get_max_bound() - sfbb.get_min_bound()) / (fbb.get_max_bound() - fbb.get_min_bound())

    # 머리로 이동
    face.vertices = o3d.utility.Vector3dVector(np.asarray(face.vertices) * np.array([ratio[0], ratio[1], 1.]))
    xbb = face.get_min_bound()
    xxbb = sfbb.get_min_bound()
    face = face.translate((-1 * xbb[0], -1 * xbb[1], -1 * xbb[2]))
    face = face.translate((xxbb[0], xxbb[1], xxbb[2]))
    # 탐색용 트리 생성
    pcd_tree = o3d.geometry.KDTreeFlann(face)
    # 각 smplx face의 포인트를 돌면서 face와 교체

    backup_mesh = copy.deepcopy(mesh_file)
    backup_face = copy.deepcopy(face)
    mesh_file = copy.deepcopy(backup_mesh)
    face = copy.deepcopy(backup_face)
    mesh_file.paint_uniform_color([211.0 / 255.0, 150.0 / 255.0, 135.0 / 255.0])
    mesh_file = mesh_file.subdivide_loop(number_of_iterations=2)
    face_surface = np.asarray(mesh_file.vertices)
    fbb_min_bound = sfbb.get_min_bound()
    fbb_max_bound = sfbb.get_max_bound()
    for i, face_point in enumerate(face_surface):
        # temp
        point_x = face_point[0]
        point_y = face_point[1]
        point_z = face_point[2]
        condi1 = fbb_min_bound[0] < point_x < fbb_max_bound[0]
        condi2 = fbb_min_bound[1] < point_y < fbb_max_bound[1]
        condi3 = fbb_min_bound[2] < point_z < fbb_max_bound[2]
        if condi1 and condi2 and condi3:
            point = mesh_file.vertices[i]
            point[2] = point[2] * 0.94
            mesh_file.vertices[i] = point
        # [_, idx_vec, dist_vec] = pcd_tree.search_knn_vector_3d(face_point, 1)
        # dist = dist_vec.pop()
        # if dist < 0.0001:
        #     idx = idx_vec.pop()
        #     # point = face.points[idx]
        #     point = mesh_file.vertices[i]
        #     if face.vertices[idx][2] < point[2]:
        #         point = copy.deepcopy(face.vertices[idx])
        #         point[2] = point[2] * 0.98
        #     mesh_file.vertices[i] = point
            # mesh_file.vertex_colors[i] = face.vertex_colors[idx]

    face = face.subdivide_loop(number_of_iterations=2)
    # o3d.visualization.draw_geometries([mesh_file])
    whole = o3d.geometry.TriangleMesh()
    whole += mesh_file
    whole += face
    o3d.io.write_triangle_mesh("./mesh_file.ply", whole, write_ascii=True)
    # dummy1 = np.asarray(mesh_file.vertices)
    # dummy2 = np.asarray(face.points)
    # merged1 = np.concatenate((dummy1, dummy2), axis=0)
    #
    # dummy3 = np.asarray(mesh_file.vertex_colors)
    # dummy4 = np.asarray(face.colors)
    # merged2 = np.concatenate((dummy3, dummy4), axis=0)
    #
    # mesh_file.vertices = o3d.utility.Vector3dVector(merged1)
    # mesh_file.vertex_colors = o3d.utility.Vector3dVector(merged2)

    cut_min_bound = cut_head.get_min_bound()
    cut_head = cut_head.translate((-1 * cut_min_bound[0], -1 * cut_min_bound[1], -1 * cut_min_bound[2]))
    # body = copy.deepcopy(mesh_file).crop(body_bbox)

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


def test():
    begin = time.time()
    mesh_file, proc_result = pre_mesh_seq()
    crop_n_attach(mesh_file, proc_result)
    end = time.time()
    print(f"{end - begin:.5f} sec")


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test()
