import copy
import shutil
import os
import open3d as o3d
import numpy as np
from util.yaml_config import YamlConfig


def get_yaml(path):
    return YamlConfig.get_dict(path)


def init_position():
    return np.identity(4)


def get_filename(ply_path):
    ply_file = open(ply_path, 'r')
    while True:
        line = ply_file.readline()
        if "direction" in line:
            value_string = line[:-1]
            value_string = value_string.replace('comment direction ', '')
            return value_string
        elif "StandardCyborg" in line:
            return "Face"


def change_filename(root):
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    for file in file_list:
        filename = get_filename(os.path.join(root, file)) + '.ply'
        os.rename(os.path.join(root, file), os.path.join(root, filename))


def convert_ply_to_img(pcd, path):
    render_pcd = copy.deepcopy(pcd)
    r = render_pcd.get_rotation_matrix_from_xyz((0, -np.pi / 4, 0))
    render_pcd.rotate(r)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_color_option = o3d.visualization.PointColorOption.Color
    vis.get_render_option().point_size = 3.0
    vis.add_geometry(render_pcd)
    vis.capture_screen_image(path, do_render=True)
    vis.destroy_window()


def convert_http_ply(path, new_filename):
    ply_file = open(path, 'r')
    new_file = open(new_filename, 'w')
    is_header = False
    while True:
        line = ply_file.readline()
        if not line:
            break
        if "ply\n" == line:
            is_header = True
        if "boundary" not in line and '\n' not in line:
            continue
        if is_header:
            new_file.write(line)
    new_file.close()
    ply_file.close()


def load_ply(root, filename):
    pcd = o3d.io.read_point_cloud(os.path.join(root, filename))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    return pcd


def load_mesh(root, filename):
    mesh = o3d.io.read_triangle_mesh(os.path.join(root, filename))
    return mesh


def load_meshes(path):
    meshes = dict()
    file_list = os.listdir(path)
    file_list = [file for file in file_list if '.glb' in file]
    for ms in file_list:
        filename = ms.replace('.glb', '').lower()
        mesh = load_mesh(root=path, filename=ms)
        meshes[filename] = mesh
        o3d.visualization.draw_geometries([mesh])
    return meshes


def load_pcds(path, cam_loc=None, imme_remove=False):
    pcds = dict()
    file_list = os.listdir(path)
    file_list = [file for file in file_list if '.ply' in file]
    for ply in file_list:
        filename = ply.replace('.ply', '').lower()
        if cam_loc is None:
            cam_loc = init_position()
        if 'face' not in filename:
            pcd = load_ply(root=path, filename=ply)
            transform = cam_loc
            location = transform[-1][0:3]
            pcd.orient_normals_towards_camera_location(location)
        else:
            pcd = load_mesh(root=path, filename=ply)
            r = pcd.get_rotation_matrix_from_xyz((0, -1 * np.pi / 2.0, 0))
            pcd = pcd.rotate(r)

        pcds[filename] = pcd
        if imme_remove:
            os.remove(os.path.join(path, ply))
    return pcds


def write_tri_mesh(mesh, filename, path=""):
    o3d.io.write_triangle_mesh(os.path.join(path, filename), mesh, write_ascii=True)


def write_pcd(pcd, filename, path=""):
    o3d.io.write_point_cloud(filename=os.path.join(path, filename), pointcloud=pcd, write_ascii=True)


def clone_ply(ply, root, filename):
    o3d.io.write_point_cloud(os.path.join(root, filename), ply, write_ascii=True)
    return os.path.join(root, filename)


def make_ply_header(path, comment, vertex):
    headers = ['ply', 'format ascii 1.0',
               'comment ', 'element vertex ',
               'property float x', 'property float y', 'property float z',
               'property uchar red', 'property uchar green', 'property uchar blue', 'property uchar alpha',
               'element face 0', 'property list uchar int vertex_indices', 'end_header']

    new_f = open(path, 'w')
    for header in headers:
        if "comment" in header:
            line = header + comment
        elif "element vertex" in header:
            vertex_len = 0
            for dir_vertex in vertex:
                vertex_len += len(dir_vertex[0])
            line = header + str(vertex_len)
        else:
            line = header
        line = line + '\n'
        new_f.write(line)

    for dir_vertex in vertex:
        x_set, y_set, z_set, rgb_set = dir_vertex
        for x, y, z, rgb in zip(x_set, y_set, z_set, rgb_set):
            r, g, b = rgb
            line = f"{x:.8f} {y:.8f} {z:.8f} "
            line += "%d %d %d 255\n" % (r, g, b)
            new_f.write(line)
    new_f.close()
    return new_f


def trim_ply(ply, begin, end, root, filename):
    ascii_path = clone_ply(ply, root, filename)
    f = open(ascii_path, 'r')
    new_f = open(ascii_path.replace('.ply', '_trim.ply'), 'w')
    row = 0
    is_header = True
    while row < end:
        line = f.readline()
        if not line:
            break
        row += 1
        if "element vertex" in line:
            line = "element vertex " + str(end - begin) + "\n"
        if row > begin or is_header:
            new_f.write(line)
        if "end_header" in line:
            end += row
            begin += row
            is_header = False
    f.close()
    new_f.close()

    new_pcd, _ = load_ply(root='', filename=ascii_path.replace('.ply', '_trim.ply'))
    return new_pcd


def clean_folder(path, without=None):
    if without is None:
        without = []

    folders = os.listdir(path)
    for folder in folders:
        ban = False
        for key in without:
            if key in folder:
                ban = True
                break
        if ban is False:
            shutil.rmtree(os.path.join(path, folder))
            os.mkdir(os.path.join(path, folder))


if __name__ == '__main__':
    clean_folder('./')
