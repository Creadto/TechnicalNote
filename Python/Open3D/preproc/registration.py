import os
import numpy as np
import open3d as o3d
import utilities.files as files
from utilities.basic.calculator import distance_vector3
from utilities.solver.metaheuristics import HarmonySearch


def align_point_cloud_hsa():
    HSA_parameters = {'hmcr': 0.7, 'par': 0.1, 'hms': 20, 'max': False}
    hyper_parameter = {'x-rot': {'min': 0.0, 'max': 2.0},
                       'y-rot': {'min': 0.0, 'max': 2.0},
                       'z-rot': {'min': 0.0, 'max': 2.0},
                       'x-tns': {'min': 0.000, 'max': 2.000},
                       'y-tns': {'min': 0.000, 'max': 2.000},
                       'z-tns': {'min': 0.000, 'max': 2.000},
                       }
    parameters = dict(HSA_parameters, **hyper_parameter)
    test_function = hsa_test

    solver = HarmonySearch(parameters=parameters, test_function=test_function)
    solver.start()
    output = solver.close()
    print(output)


def hsa_test(parameter):
    root = '../data/target'
    source = os.path.join(root, 'source.ply')
    target = os.path.join(root, 'target.ply')
    src_loc = files.get_position(source)
    tgt_loc = files.get_position(target)

    src_pcd, _ = files.load_ply('', source, src_loc)
    tgt_pcd, _ = files.load_ply('', target, tgt_loc)

    # action
    rotate = (np.pi * parameter['x-rot'], np.pi * parameter['y-rot'], np.pi * parameter['z-rot'])
    R = tgt_pcd.get_rotation_matrix_from_xyz(rotate)
    translate = (parameter['x-tns'], parameter['y-tns'], parameter['z-tns'])
    tgt_pcd.translate(translate)
    tgt_pcd = tgt_pcd.rotate(R)

    # evaluation
    if len(src_pcd.normals) > len(tgt_pcd.normals):
        src_points = np.asarray(tgt_pcd.normals)
        tgt_points = np.asarray(src_pcd.normals)
    else:
        src_points = np.asarray(src_pcd.normals)
        tgt_points = np.asarray(tgt_pcd.normals)

    src_colors = np.asarray(src_pcd.colors)
    tgt_colors = np.asarray(tgt_pcd.colors)
    # xyz matching
    output = 0.0
    for idx in range(len(src_points)):
        source = src_points[idx, :]
        eval_dist = distance_vector3(source, tgt_points)
        output += eval_dist
    # color matching in matched xyz
    return output


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_fpfh_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return (source_down, source_fpfh), (target_down, target_fpfh)


def execute_global_registration(source_tuple, target_tuple, voxel_size):
    (source_down, source_fpfh) = source_tuple
    (target_down, target_fpfh) = target_tuple
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result
