import open3d as o3d
import numpy as np


def practice_normal_vector():
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(3000)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


def read_sample(sample: str):
    data = getattr(o3d.data, sample)()
    rtn_pcd = o3d.io.read_point_cloud(data.path)
    return rtn_pcd


def make_tri_mesh(source):
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(source, o3d.utility.DoubleVector([radius, radius * 2]))
    return bpa_mesh


def make_poisson_mesh(target):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            target, depth=9)
