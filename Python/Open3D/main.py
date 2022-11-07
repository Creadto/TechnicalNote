import copy
import utilities.files as files
import utilities.visualization as vis
import preproc.registration as reg
import numpy as np


def reconstructure():
    import open3d as o3d
    import matplotlib.pyplot as plt
    bunny = o3d.data.BunnyMesh()
    mesh = o3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(750)
    o3d.visualization.draw_geometries([pcd])
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()

    pcd = gt_mesh.sample_points_poisson_disk(3000)
    o3d.visualization.draw_geometries([pcd])

    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    o3d.visualization.draw_geometries([pcd, rec_mesh])

    eagle = o3d.data.EaglePointCloud()
    pcd = o3d.io.read_point_cloud(eagle.path)

    print(pcd)
    o3d.visualization.draw_geometries([pcd],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])

    print('run Poisson surface reconstruction')
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
    print(mesh)
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])

    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.664,
                                      front=[-0.4761, -0.4698, -0.7434],
                                      lookat=[1.8900, 3.2596, 0.9284],
                                      up=[0.2304, -0.8825, 0.4101])


def KDTreeTest():
    # 내가 찾은 포인트 주변을 빠르게 검색해주는 K-dimensional Tree
    import open3d as o3d
    print("Testing kdtree in Open3D...")
    print("Load a point cloud and paint it gray.")

    sample_pcd_data = o3d.data.PCDPointCloud()
    pcd = o3d.io.read_point_cloud(sample_pcd_data.path)
    o3d.visualization.draw_geometries([pcd])
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    print("Paint the 1501st point red.")
    pcd.colors[1500] = [1, 0, 0]

    print("Find its 200 nearest neighbors, and paint them blue.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

    print("Find its neighbors with distance less than 0.2, and paint them green.")
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

    print("Visualize the point cloud.")
    o3d.visualization.draw_geometries([pcd])


def rotation():
    config = files.get_yaml(path='./data/rotate/local_config.yaml')
    pcds = files.load_pcds('./data/rotate', cam_loc=config['CameraLocation'])
    is_processing = True
    while is_processing:
        scale = float(input("insert scale"))
        dummy = copy.deepcopy(pcds)
        for idx, pcd in enumerate(dummy):
            rot = ((np.pi / 4) * idx)
            rot_mat = pcd.get_rotation_matrix_from_xyz((0, rot, 0))
            pcd.rotate(rot_mat)
            term = scale / len(dummy)
            pcd.translate((term * idx, 0, term * idx), relative=False)
            dummy[idx] = pcd

        vis.draw_color(dummy)
        is_processing = input("Do you want to exit? (Yes / No)").lower() == 'no'
        if is_processing is False:
            pcds = dummy
    vis.draw_geometries(pcds)
    vis.draw_geometries(pcds, True)


def registration():
    v_size = 0.005
    config = files.get_yaml(path='./data/rotate/local_config.yaml')
    pcds = files.load_pcds('./data/rotate', cam_loc=config['CameraLocation'])
    vis.draw_color(pcds, view_normal=False)
    source = copy.deepcopy(pcds[0])
    for idx in range(1, len(pcds)):
        target = copy.deepcopy(pcds[idx])
        src_tup, tgt_tup = reg.prepare_fpfh_dataset(source=source, target=target, voxel_size=v_size)
        result_ransac = reg.execute_global_registration(src_tup, tgt_tup, v_size)
        vis.draw_geometries([source.transform(result_ransac.transformation), target])
        vis.draw_geometries([pcds[0], target])


def cutting():
    config = files.get_yaml(path='./data/load/local_config.yaml')
    pcds = files.load_pcds('./data/load', cam_loc=config['CameraLocation'])
    is_processing = True
    vis.draw_color(pcds)
    while is_processing:
        pcd_index = int(input("Choice index number"))
        pcd = pcds[pcd_index]
        n_of_points = len(pcd.points)
        cut_range = input(str(n_of_points) + " Points object, insert cut range (begin, end)")
        cut_range = cut_range.split(sep=',')
        files.trim_ply(pcd, n_of_points * float(cut_range[0]), n_of_points * float(cut_range[1]),
                       './data/clone', filename='temp.ply')
        pcd, _ = files.load_ply('./data/clone', 'temp_trim.ply', cam_loc=None)
        pcds[pcd_index] = pcd
        vis.draw_color(pcds)
        is_processing = input("Do you want to exit? (Yes / No)").lower() == 'no'


def main():
    config = files.get_yaml(path='./data/rotate/local_config.yaml')
    pcds = files.load_pcds('./data/rotate', cam_loc=config['CameraLocation'])
    pcd = pcds[0]


if __name__ == '__main__':
    reconstructure()