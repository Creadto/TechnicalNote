import numpy as np
import open3d as o3d


def combine_pcds(pcds, down_sampling=False):
    combined = o3d.geometry.PointCloud()
    for pcd in pcds:
        combined += pcd
    if down_sampling:
        combined = combined.voxel_down_sample(voxel_size=0.001)
    return combined


def gen_tri_mesh(pcd):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
                                                                               o3d.utility.DoubleVector(
                                                                                   [radius, radius * 2]))
    return bpa_mesh