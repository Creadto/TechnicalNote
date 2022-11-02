import os
import open3d as o3d


def load_ply(root, filename):
    pcd = o3d.io.read_point_cloud(os.path.join(root, filename))
    n_of_points = len(pcd.points)
    return pcd, n_of_points


def clone_ply(ply, root, filename):
    o3d.io.write_point_cloud(os.path.join(root, filename), ply, write_ascii=True)
    return os.path.join(root, filename)


def trim_ply(ply, begin, end, root, filename):
    ascii_path = clone_ply(ply, root, filename)
    f = open(ascii_path, 'r')
    new_f = open(ascii_path.replace('.ply', '_trim.ply'), 'w')
    row = 0
    is_header = True
    while row < end:
        line = f.readline()
        if not line:
            break
        row += 1
        if "element vertex" in line:
            line = "element vertex " + str(end - begin) + "\n"
        if row > begin or is_header:
            new_f.write(line)
        if "end_header" in line:
            end += row
            begin += row
            is_header = False
    f.close()
    new_f.close()


def main():
    file_list = os.listdir('./origin')
    file_list = [file for file in file_list if '.ply' in file]
    for ply in file_list:
        pcd, n_of_points = load_ply(root='./origin', filename=ply)
        if "Right" in ply:
            trim_ply(pcd, 0, int(n_of_points * 0.3), root='./clone', filename=ply)
        elif "Left" in ply:
            trim_ply(pcd, int(n_of_points * 0.7), n_of_points, root='./clone', filename=ply)


if __name__ == '__main__':
    main()