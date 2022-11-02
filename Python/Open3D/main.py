import copy
import utilities.files as files
import utilities.visualization as vis
import preproc.registration as reg
import numpy as np


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
    rotation()


if __name__ == '__main__':
    main()