import os
import utilities.files as files
import utilities.visualization as vis
from preproc.registration import align_point_cloud_hsa
import processor.meshing as meshing


def change_filename(root):
    import os
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

    target_pcd, n_of_points = files.load_ply(root =root, filename=file_list[0], cam_loc=cam_loc)
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
    files.write_tri_mesh(mesh=tau_mesh, path=root, filename='tau_mesh.ply')

    removed = meshing.remove_noise(tau_mesh, 1000)
    mesh = removed.subdivide_loop(number_of_iterations=1)
    import open3d as o3d
    o3d.visualization.draw_geometries([mesh])


def to_mesh(root):
    import open3d as o3d
    import copy
    import numpy as np
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    filenames = []
    pcds = []
    cam_locs = []
    for ply in file_list:
        filenames.append(ply)
        cam_loc = files.get_position(ply_path=os.path.join(root, ply))
        pcd, _ = files.load_ply(root=root, filename=ply, cam_loc=cam_loc)
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
            R = pcd.get_rotation_matrix_from_xyz((0, -1 * np.pi / 2, 0))
            new_pcd.translate((0.35, 0, 0.05))
        else:
            R = pcd.get_rotation_matrix_from_xyz((0, 0, 0))
            new_pcd.translate((-offset, 0, -offset))
        if "right" in lower:
            new_pcd.translate((-0.05, 0, 0.05))
        if "front" in lower:
            new_pcd.translate((0.05, 0, -0.05))

        new_pcd = new_pcd.rotate(R)
        new_pcd.transform(cam_locs[cnt].T)
        cnt += 1
        new_pcds.append(new_pcd)
    o3d.visualization.draw_geometries(new_pcds)

    combined = meshing.combine_pcds(new_pcds, True)
    mesh = meshing.gen_tri_mesh(combined)
    files.write_tri_mesh(mesh, '../data/me/mesh.ply', './')


if __name__ == '__main__':
    root_dir = r"../data/me"
    #crop_and_check(root_dir, begin=0.3, end=1.0)
    align_point_cloud_hsa()
