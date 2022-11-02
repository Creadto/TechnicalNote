import copy
import random
import open3d as o3d


def draw_color(pcds, view_normal=False):
    show_pcds = []
    for element in pcds:
        pcd = copy.deepcopy(element)
        pcd.paint_uniform_color([random.random(), random.random(), random.random()])
        show_pcds.append(pcd)
    o3d.visualization.draw_geometries(show_pcds, point_show_normal=view_normal)


def draw_geometries(pcds, view_normal=False):
    o3d.visualization.draw_geometries(pcds, point_show_normal=view_normal)