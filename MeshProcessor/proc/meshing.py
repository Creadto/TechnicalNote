import copy
import numpy as np
import open3d as o3d


def combine_pcds(pcds, down_sampling=False):
    combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined += pcd
    if down_sampling:
        combined = combined.voxel_down_sample(voxel_size=0.001)

    return combined.remove_duplicated_points()


def gen_tri_mesh(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist

    # poisson_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd=pcd, scale=1.0)
    # mesh = poisson_mesh.subdivide_loop(number_of_iterations=2)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                               o3d.utility.DoubleVector([radius, radius * 3]))
    return mesh


def taubin_filter(mesh, itr=100):
    mesh_taubin = mesh.filter_smooth_taubin(number_of_iterations=itr)
    mesh_taubin.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_taubin])
    return mesh_taubin


def remove_noise(mesh, volume=100):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (
            mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    mesh_1 = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_1.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([mesh_1])
    return mesh_1
