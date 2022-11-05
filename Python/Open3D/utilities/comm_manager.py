from http.server import HTTPServer
import time


class HttpServer:
    def __init__(self, ip: str, port: int, listener, handler):
        self.NewMessage = None
        self.listener = listener
        httpd = HTTPServer((ip, port), handler)
        httpd.serve_forever()

    def listen(self):
        while True:
            self.NewMessage = self.listener()
            time.sleep(0.001)


if __name__ == '__main__':
    is_header = False
    # ply_file = open('../plt.ply', 'r')
    # new_file = open('../plt_new.ply', 'w')
    # while True:
    #     line = ply_file.readline()
    #     if not line:
    #         break
    #     if "ply\n" == line:
    #         is_header = True
    #     if "boundary" not in line and '\n' not in line:
    #         continue
    #     if is_header:
    #         new_file.write(line)
    # new_file.close()
    # ply_file.close()
    #
    # pcd = o3d.io.read_point_cloud('../plt_new.ply')
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    # distances = pcd.compute_nearest_neighbor_distance()
    # avg_dist = np.mean(distances)
    # radius = 3 * avg_dist
    #
    # bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,
    #                                                                            o3d.utility.DoubleVector(
    #                                                                                [radius, radius * 2]))
    # o3d.io.write_triangle_mesh('../temp_mesh.ply', bpa_mesh, write_ascii=True)
    # mesh_temp = o3d.io.read_triangle_mesh('../temp_mesh.ply')
    #
    # httpd = HTTPServer(('192.168.219.102', 3000), RequestHandler)
    # httpd.serve_forever()