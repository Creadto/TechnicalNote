import os
import numpy as np
import copy
import open3d as o3d


def draw_registration_result(pcds):
    pcd1 = copy.deepcopy(pcds[0])
    pcd2 = copy.deepcopy(pcds[1])
    pcd3 = copy.deepcopy(pcds[2])
    pcd1.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    pcd3.paint_uniform_color([0.706, 0, 1])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3], point_show_normal=True)


def load_ply(root, filename):
    pcd = o3d.io.read_point_cloud(os.path.join(root, filename))
    n_of_points = len(pcd.points)
    pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    if "Front" in filename:
        coord = np.array([0.0022530183, -0.121533684, -0.018694922])
        pcd.orient_normals_towards_camera_location(coord)
        pcd.transform(trans_init)
    if "Back" in filename:
        coord = np.array([-0.91260564, 0.21664503, -3.3843343])
        pcd.orient_normals_towards_camera_location(coord)
        pcd.transform(trans_init)
    if "Left" in filename:
        coord = np.array([1.1108829, 0.17426117, -2.1864507])
        pcd.orient_normals_towards_camera_location(coord)
        pcd.transform(trans_init)
    if "Right" in filename:
        coord = np.array([-2.0660899, 0.21975859, -1.3162844])
        pcd.orient_normals_towards_camera_location(coord)
        pcd.transform(trans_init)
    return pcd, n_of_points


def load_pcds(path):
    pcds = []
    file_list = os.listdir(path)
    file_list = [file for file in file_list if '.ply' in file]
    for ply in file_list:
        pcd, _ = load_ply(root=path, filename=ply)
        pcds.append(pcd)
    draw_registration_result(pcds)
    return pcds


def convert_mesh_from_pcd(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                               o3d.utility.DoubleVector(
                                                                                   [radius, radius * 2]))
    return bpa_mesh


def write_mesh(mesh, filename, path=""):
    o3d.io.write_triangle_mesh(path + filename, mesh, write_ascii=True)


def main():
    threshold = 0.005
    current_transformation = np.identity(4)
    pcds = load_pcds('./testbed')
    pcd_combined = o3d.geometry.PointCloud()
    pcd_combined += pcds[0]
    print("Initial alignment")
    for tgt_idx in range(1, 3):
        target = pcds[tgt_idx]
        result_icp = o3d.pipelines.registration.registration_icp(
            target, pcd_combined, 0.02, current_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        pcds[tgt_idx + 2].transform(result_icp.transformation)

    for idx in range(3, len(pcds)):
        pcd_combined += pcds[idx]

    o3d.visualization.draw_geometries([pcd_combined])

    bpa_mesh = convert_mesh_from_pcd(pcd=pcd_combined)
    write_mesh(bpa_mesh, "./testbed/mesh.ply")


if __name__ == '__main__':
    #main()
    pcds = load_pcds('./testbed')
    o3d.visualization.draw_geometries(pcds)

