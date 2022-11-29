import copy
import os
import open3d as o3d
import numpy as np
from utilities.yaml_config import YamlConfig
from utilities.measure import get_width, get_height


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


def load_ply(root, filename, cam_loc, **kwargs):
    pcd = o3d.io.read_point_cloud(os.path.join(root, filename))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    if cam_loc is not None:
        if isinstance(cam_loc, dict):
            for key in cam_loc.keys():
                if key in filename:
                    transform = cam_loc[key]
                    location = transform[-1][0:3]
                    pcd.orient_normals_towards_camera_location(np.array(location))
                    break
        else:
            transform = cam_loc
            location = transform[-1][0:3]
            pcd.orient_normals_towards_camera_location(location)
        if "to_cam" in kwargs:
            pcd.transform(kwargs['to_cam'].T)
        else:
            pcd.transform(np.identity(4))

    return pcd


def load_mesh(root, filename):
    mesh = o3d.io.read_triangle_mesh(os.path.join(root, filename))
    return mesh


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

    new_pcd, _ = load_ply(root='', filename=ascii_path.replace('.ply', '_trim.ply'), cam_loc=None)
    return new_pcd


def crop_ply(ply: o3d.geometry.PointCloud, min_bound, max_bound):
    ply = ply.remove_duplicated_points()
    import matplotlib.pyplot as plt
    # # Flip it, otherwise the pointcloud will be upside down.
    # ply.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            ply.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    ply.colors = o3d.utility.Vector3dVector(colors[:, :3])

    largest_cluster_idx = labels.argmax()
    triangles_to_remove = labels != largest_cluster_idx

    o3d.visualization.draw([ply])
