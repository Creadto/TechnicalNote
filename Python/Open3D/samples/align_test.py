import os
import utilities.files as files
import processor.meshing as meshing


def change_filename(root):
    import os
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    for file in file_list:
        filename = files.get_filename(ply_path=os.path.join(root, file))
        filename += '.ply'
        os.rename(os.path.join(root, file), os.path.join(root, filename))


def convert_to_image(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    filenames = []
    pcds = []

    for ply in file_list:
        filenames.append(ply)
        cam_loc = files.get_position(ply_path=os.path.join(root, ply))
        pcd, _ = files.load_ply(root=root, filename=ply, cam_loc=cam_loc)
        pcds.append(pcd)

    for name, pcd in zip(filenames, pcds):
        test = files.trim_ply(pcd, 0.0, 0.4, './', 'test.ply')
        dists = pcd.compute_point_cloud_distance(pcd)
        print(dists)
        files.convert_ply_to_img(pcd=pcd, path=os.path.join(root, name.replace('ply', 'jpg')))

def to_mesh(root):
    import open3d as o3d
    import copy
    import numpy as np
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    filenames = []
    pcds = []

    for ply in file_list:
        filenames.append(ply)
        cam_loc = files.get_position(ply_path=os.path.join(root, ply))
        pcd, _ = files.load_ply(root=root, filename=ply, cam_loc=cam_loc)
        pcds.append(pcd)

    offset = 0.1
    new_pcds = []
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
        new_pcd = new_pcd.rotate(R)
        if "right" in lower:
            new_pcd.translate((-0.05, 0, 0.05))
        if "front" in lower:
            new_pcd.translate((0.05, 0, -0.05))
        new_pcds.append(new_pcd)
    o3d.visualization.draw_geometries(new_pcds)

    combined = meshing.combine_pcds(new_pcds, True)
    mesh = meshing.gen_tri_mesh(combined)
    files.write_tri_mesh(mesh, 'mesh.ply', './')


if __name__ == '__main__':
    root_dir = r"../data/me/"
    #change_filename(root=root_dir)
    convert_to_image(root=root_dir)