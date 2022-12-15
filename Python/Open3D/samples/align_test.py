import os
import copy
import numpy as np
import open3d as o3d
import utilities.files as files
import utilities.visualization as vis
from preproc.registration import align_point_cloud_hsa
from preproc.clustering import get_largest_cluster
import processor.meshing as meshing


def transform_test():
    root = '../data/me'
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]

    run_flag = True

    pcds = []
    new_pcds = []
    for idx, filename in enumerate(file_list):
        cam_loc = files.get_position(os.path.join(root, filename))
        origin_pcd: o3d.geometry.PointCloud = files.load_ply(root=root, filename=filename, cam_loc=cam_loc)
        origin_pcd, _ = get_largest_cluster(origin_pcd)
        origin_pcd = origin_pcd.translate((0, 2.0 * idx, 0))
        offset_unit = 1.0
        for idx in range(1, 9):
            target_pcd: o3d.geometry.PointCloud = copy.deepcopy(origin_pcd)
            r = target_pcd.get_rotation_matrix_from_xyz((0, 2 * np.pi * (idx / 8), 0))
            target_pcd = target_pcd.translate((offset_unit * idx, 0, offset_unit * idx))
            target_pcd = target_pcd.rotate(r)
            new_pcds.append(target_pcd)
        origin_pcd.paint_uniform_color((0.5, 0.1, 0.1))0
        pcds.append(origin_pcd)

    o3d.visualization.draw_geometries(pcds + new_pcds)


def change_filename(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    for file in file_list:
        filename = files.get_filename(ply_path=os.path.join(root, file))
        filename += '.ply'
        os.rename(os.path.join(root, file), os.path.join(root, filename))


def crop_and_check(root, begin, end):
    # 안됨
    target = "Front.ply"
    file_list = os.listdir(root)
    file_list = [file for file in file_list if target in file]
    cam_loc = files.get_position(os.path.join(root, file_list[0]))

    target_pcd, n_of_points = files.load_ply(root=root, filename=file_list[0], cam_loc=cam_loc)
    target_path = root.replace('me', 'target')
    begin = n_of_points * begin
    end = n_of_points * end
    trimmed_pcd = files.trim_ply(ply=target_pcd, begin=begin, end=end, root=target_path, filename=file_list[0])
    vis.draw_geometries(target_pcd)
    vis.draw_geometries(trimmed_pcd)


def convert_to_image(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if 'scene.ply' in file]
    filenames = []
    pcds = []

    for ply in file_list:
        filenames.append(ply)
        # cam_loc = files.get_position(ply_path=os.path.join(root, ply))
        pcd, _ = files.load_ply(root=root, filename=ply, cam_loc=None)
        pcds.append(pcd)

    import utilities.visualization as vis_handler
    vis_handler.draw_geometries(pcds, False)
    for name, pcd in zip(filenames, pcds):
        test = files.trim_ply(pcd, 0.0, 0.4, './', 'test.ply')
        dists = pcd.compute_point_cloud_distance(pcd)
        print(dists)
        files.convert_ply_to_img(pcd=pcd, path=os.path.join(root, name.replace('ply', 'jpg')))


def mesh_filtering(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if 'mesh.ply' in file]
    raw_mesh = files.load_mesh(root=root, filename=file_list[0])
    tau_mesh = meshing.taubin_filter(raw_mesh, 30)
    removed = meshing.remove_noise(tau_mesh, 1000)
    files.write_tri_mesh(mesh=tau_mesh, path=root, filename='removed_mesh.glb')
    mesh = removed.subdivide_loop(number_of_iterations=2)
    o3d.visualization.draw_geometries([mesh])
    files.write_tri_mesh(mesh=mesh, path=root, filename='upscaling_mesh.glb')


def automated_mesh(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    filenames = []
    pcds = []
    bodies = []
    arms = []
    heads = []
    sub_root = '../data'
    for ply in file_list:
        filenames.append(ply)
        cam_loc = files.get_position(ply_path=os.path.join(root, ply))
        pcd = files.load_ply(root=root, filename=ply, cam_loc=cam_loc)
        pcd, parts = get_largest_cluster(pcd)
        pcds.append(pcd)

        bodies.append(parts['body'])
        arms.append(parts['arm'])
        heads.append(parts['head'])

        files.write_pcd(parts['body'], os.path.join(sub_root, 'body', ply))
        files.write_pcd(parts['arm'], os.path.join(sub_root, 'arm', ply))
        files.write_pcd(parts['head'], os.path.join(sub_root, 'head', ply))

    align_point_cloud_hsa()


def to_mesh(root):
    import open3d as o3d
    import numpy as np
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    filenames = []
    pcds = []
    cam_locs = []
    for ply in file_list:
        filenames.append(ply)
        cam_loc = files.get_position(ply_path=os.path.join(root, ply))
        pcd = files.load_ply(root=root, filename=ply, cam_loc=cam_loc)
        pcds.append(pcd)
        cam_locs.append(cam_loc)

    offset = 0.1
    new_pcds = []
    cnt = 0
    for name, pcd in zip(filenames, pcds):
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
        if "right" in lower:
            # new_pcd.translate((-0.0159, 0, 0.0547))
            new_pcd.translate((-0.05, 0, 0.05))
        if "front" in lower:
            new_pcd.translate((0.05, 0, -0.05))
        # new_pcd.transform(cam_locs[cnt].T)
        cnt += 1
        new_pcd, parts = get_largest_cluster(new_pcd)
        new_pcds.append(new_pcd)

    o3d.visualization.draw_geometries(new_pcds)
    combined = meshing.combine_pcds(new_pcds, True)
    mesh = meshing.gen_tri_mesh(combined)
    files.write_tri_mesh(mesh, '../data/me/mesh.ply', './')


def mesh_script(root):
    # 대상 PointCloud 객체 호출
    pcds = files.load_pcds(root)
    # 보고 제어하기 편하게 90도 회전 (노이즈 제거하고)
    bodies = dict()
    for key, pcd in pcds.items():
        pcd, parts = get_largest_cluster(pcd)
        pcds[key] = pcd
        bodies[key] = parts
    # 확인하고
    vis.draw_geometries(list(pcds.values()))

    # 옆 통의 길이를 잰다.
    left_body = bodies['left']
    left_pcd: o3d.geometry.PointCloud = left_body['arm']
    max_bnd = left_pcd.get_max_bound()
    max_bnd[1] = 0.0
    min_bnd = left_pcd.get_min_bound()
    min_bnd[1] = 0.0
    depth = np.linalg.norm(max_bnd - min_bnd) * 0.9

    # Front x, y와 Back x, y가 depth 만큼 벌어져 있어야함
    back_pcd: o3d.geometry.PointCloud = pcds['back']
    front_pcd: o3d.geometry.PointCloud = pcds['front']

    offset = depth * 0.8 / np.sqrt(2)
    m_distance = front_pcd.get_center() - back_pcd.get_center()
    m_distance[0] += offset
    m_distance[2] -= offset
    back_pcd = back_pcd.translate(tuple(m_distance))

    # Left와 Right는 Front와 Back의 Depth의 절반의 거리로 이동을하고
    left_pcd: o3d.geometry.PointCloud = pcds['left']
    right_pcd: o3d.geometry.PointCloud = pcds['right']

    # 이동과 회전 + Front의 최대 x, z로 Left가 이동
    offset = (depth / 2.0) / np.sqrt(2)
    m_distance = front_pcd.get_max_bound() - left_pcd.get_center()
    y_meter = (front_pcd.get_center() - left_pcd.get_center())[1]
    m_distance[0] += offset * 2.8
    m_distance[1] = y_meter
    m_distance[2] -= offset * 2.8
    # Test
    m_distance[0] *= 0.47
    m_distance[2] *= 0.47
    left_pcd = left_pcd.translate(tuple(m_distance))

    # 이동과 회전 + Front의 최소 x, z로 Right가 이동
    m_distance = front_pcd.get_min_bound() - right_pcd.get_center()
    y_meter = (front_pcd.get_center() - left_pcd.get_center())[1]
    m_distance[0] += offset
    m_distance[1] = y_meter
    m_distance[2] -= offset
    # Test
    m_distance[0] *= 0.4
    m_distance[2] *= 0.4
    right_pcd = right_pcd.translate(tuple(m_distance))

    r = back_pcd.get_rotation_matrix_from_xyz((0, np.pi, 0))
    back_pcd = back_pcd.rotate(r)
    pcds['back'] = back_pcd

    r = left_pcd.get_rotation_matrix_from_xyz((0, np.pi * 0.5, 0))
    left_pcd = left_pcd.rotate(r)
    pcds['left'] = left_pcd

    r = back_pcd.get_rotation_matrix_from_xyz((0, np.pi * 1.5, 0))
    right_pcd = right_pcd.rotate(r)
    pcds['right'] = right_pcd

    vis.draw_geometries(list(pcds.values()))
    combined = meshing.combine_pcds(list(pcds.values()), True)
    mesh = meshing.gen_tri_mesh(combined)
    files.write_tri_mesh(mesh, '../data/me/mesh.ply', './')


if __name__ == '__main__':
    root_dir = r"../data/me"
    mesh_script(root_dir)
    mesh_filtering(root_dir)
