import copy
import os
import open3d as o3d
import numpy as np
from utilities.yaml_config import YamlConfig


def get_yaml(path):
    return YamlConfig.get_dict(path)


def get_position(ply_path):
    ply_file = open(ply_path, 'r')
    while True:
        line = ply_file.readline()
        if "lastCameraTransform" in line:
            begin = line.find('[')
            value_string = line[begin:-2]
            value_string = value_string.replace('[', '')
            value_string = value_string.replace(']', '')
            vector_string = value_string.split(',')
            vector = [float(value) for value in vector_string]
            matrix = np.array(vector).reshape((4, 4))
            return matrix


def get_filename(ply_path):
    ply_file = open(ply_path, 'r')
    while True:
        line = ply_file.readline()
        if "direction" in line:
            value_string = line[:-1]
            value_string = value_string.replace('comment direction ', '')
            return value_string


def convert_ply_to_img(pcd, path):
    render_pcd = copy.deepcopy(pcd)
    R = render_pcd.get_rotation_matrix_from_xyz((0, -np.pi / 4, 0))
    render_pcd.rotate(R)

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


def load_ply(root, filename, cam_loc):
    pcd = o3d.io.read_point_cloud(os.path.join(root, filename))
    n_of_points = len(pcd.points)
    pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=1.5)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    if cam_loc is not None:
        if isinstance(cam_loc, dict):
            for key in cam_loc.keys():
                if key in filename:
                    transform = cam_loc[key]
                    location = transform[-1][0:3]
                    pcd.orient_normals_towards_camera_location(np.array(location))
                    pcd.transform(np.identity(4))
                    break
        else:
            transform = cam_loc
            location = transform[-1][0:3]
            pcd.orient_normals_towards_camera_location(np.array(location))
            pcd.transform(np.identity(4))

    return pcd, n_of_points


def load_pcds(path, cam_loc=None):
    pcds = []
    file_list = os.listdir(path)
    file_list = [file for file in file_list if '.ply' in file]
    for ply in file_list:
        if cam_loc is None:
            cam_loc = get_position(ply_path=os.path.join(path, ply))
        pcd, _ = load_ply(root=path, filename=ply, cam_loc=cam_loc)
        pcds.append(pcd)
    return pcds


def write_tri_mesh(mesh, filename, path=""):
    o3d.io.write_triangle_mesh(os.path.join(path, filename), mesh, write_ascii=True)


def write_pcd(pcd, filename, path=""):
    o3d.io.write_point_cloud(filename=os.path.join(path, filename), pointcloud=pcd, write_ascii=True)


def clone_ply(ply, root, filename):
    o3d.io.write_point_cloud(os.path.join(root, filename), ply, write_ascii=True)
    return os.path.join(root, filename)


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
