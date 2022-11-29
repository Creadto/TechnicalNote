import os
import copy
import numpy as np
import open3d as o3d
import utilities.files as files
from utilities.basic.calculator import distance_matrix3
from utilities.solver.metaheuristics import HarmonySearch
import datetime


def align_point_cloud_hsa():
    hsa_parameters = {'hmcr': 0.7, 'par': 0.001, 'hms': 15, 'max': False}
    hyper_parameter = {'Front-x-rot': {'min': 0.0, 'max': 0.0},
                       'Front-y-rot': {'min': 0.0, 'max': 2.0},
                       'Front-z-rot': {'min': 0.0, 'max': 0.0},
                       'Front-x-tns': {'min': -0.1, 'max': 0.1},
                       'Front-y-tns': {'min': 0.000, 'max': 0.000},
                       'Front-z-tns': {'min': -0.1, 'max': 0.1},
                       'Left-x-rot': {'min': 0.0, 'max': 0.0},
                       'Left-y-rot': {'min': 0.0, 'max': 2.0},
                       'Left-z-rot': {'min': 0.0, 'max': 0.0},
                       'Left-x-tns': {'min': -0.1, 'max': 0.1},
                       'Left-y-tns': {'min': 0.000, 'max': 0.000},
                       'Left-z-tns': {'min': -0.1, 'max': 0.1},
                       'Right-x-rot': {'min': 0.0, 'max': 0.0},
                       'Right-y-rot': {'min': 0.0, 'max': 2.0},
                       'Right-z-rot': {'min': 0.0, 'max': 0.0},
                       'Right-x-tns': {'min': -0.1, 'max': 0.1},
                       'Right-y-tns': {'min': 0.000, 'max': 0.000},
                       'Right-z-tns': {'min': -0.1, 'max': 0.1},
                       'Back-x-rot': {'min': 0.0, 'max': 0.0},
                       'Back-y-rot': {'min': 0.0, 'max': 2.0},
                       'Back-z-rot': {'min': 0.0, 'max': 0.0},
                       'Back-x-tns': {'min': -0.1, 'max': 0.1},
                       'Back-y-tns': {'min': 0.000, 'max': 0.000},
                       'Back-z-tns': {'min': -0.1, 'max': 0.1},
                       }
    parameters = dict(hsa_parameters, **hyper_parameter)
    test_function = hsa_test

    solver = HarmonySearch(parameters=parameters, test_function=test_function)
    solver.start()
    output = solver.close()
    print(output)


def update_parameters(parameters):
    rotation = dict()
    translate = dict()
    mapper = {'x': 0, 'y': 1, 'z': 2}
    for key, value in parameters.items():
        sep = key.split('-')
        if len(sep) >= 3:
            if sep[2] == 'rot':
                if sep[0] in rotation:
                    rotation[sep[0]][mapper[sep[1]]] = value
                else:
                    new_val = [0.0, 0.0, 0.0]
                    new_val[mapper[sep[1]]] = value
                    rotation[sep[0]] = new_val
            else:
                if sep[0] in translate:
                    translate[sep[0]][mapper[sep[1]]] = value
                else:
                    new_val = [0.0, 0.0, 0.0]
                    new_val[mapper[sep[1]]] = value
                    translate[sep[0]] = new_val
    return rotation, translate


def hsa_test(parameter):
    root = '../data/body'
    file_list = os.listdir(root)
    file_list = [file for file in file_list if '.ply' in file]
    normals = dict()
    rotation, translate = update_parameters(parameters=parameter)
    for ply in file_list:
        key = ply.replace('.ply', '')
        pcd = files.load_ply(root=root, filename=ply, cam_loc=None)
        downpcd = pcd.voxel_down_sample(voxel_size=0.02)
        r = downpcd.get_rotation_matrix_from_xyz(tuple(rotation[key]))
        downpcd = downpcd.translate(tuple(translate[key]))
        downpcd = downpcd.rotate(r)
        normals[key] = np.asarray(downpcd.normals)

    # evaluation: normal matching
    start = datetime.datetime.now()
    output = distance_matrix3(normals['Front'], normals['Left'])
    output += distance_matrix3(normals['Front'], normals['Right'])
    output += distance_matrix3(normals['Back'], normals['Left'])
    print(datetime.datetime.now() - start)
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
