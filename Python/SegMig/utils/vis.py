import copy
import open3d as o3d
import random


def draw_color(pcds):
    show_pcds = []
    for element in pcds:
        pcd = copy.deepcopy(element)
        pcd.paint_uniform_color([1, 0.706, 0])
    pcd2.paint_uniform_color([0, 0.651, 0.929])
    pcd3.paint_uniform_color([0.706, 0, 1])
    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3], point_show_normal=True)
