import os
import time
import threading
from json import dumps
import processor.meshing as meshing
from http.server import BaseHTTPRequestHandler
from utilities.comm_manager import HttpServer
import utilities.files as files


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        path = os.path.dirname(os.path.abspath(__file__))
        memory_count = int(self.headers['Content-Length'])
        receive_data = self.rfile.read(memory_count)
        status = 'Idle'
        data = ""
        if memory_count < 200:
            receive_data = receive_data.decode()
            # State check
            file_list = os.listdir(path)
            check_list = ['Meshed', 'Loaded', 'Received']
            for check in check_list:
                check_files = [file for file in file_list if check in file]
                if len(check_files) > 0:
                    status = check
                    break
            if "Check" not in receive_data:
                if status == 'Meshed':
                    in_file = open(os.path.join(path, "Meshed.ply"), "rb")
                    data = in_file.read()
                    os.remove(os.path.join(path, "Meshed.ply"))
                else:
                    status = "Invalid"

        else:
            f = open(os.path.join(path, 'Received.ply'), 'wb')
            f.write(receive_data)
            f.close()

        response = {"Status": status, "Data": data}
        self.send_dict_response(response)

    def send_dict_response(self, d):
        """ Sends a dictionary (JSON) back to the client """
        self.wfile.write(bytes(dumps(d), "utf8"))


class HttpProvider:
    def __init__(self, ip, port):
        if not isinstance(ip, str):
            raise Exception("Wrong IP number: Input IP number as string")
        if not isinstance(port, int):
            raise Exception("Wrong Port number: Input port number as integer")

        self.storage_path = os.path.dirname(os.path.abspath(__file__))
        self.server = HttpServer(ip=ip, port=port, listener=self.listener(), handler=RequestHandler)

    def serve(self):
        t = threading.Thread(target=self.server.listen)
        t.start()

        run_flag = True
        while run_flag:
            if self.server.NewMessage is not None:
                camera_location = files.get_pos_in_file(self.server.NewMessage)
                pcd = files.load_ply(root='', filename=self.server.NewMessage, cam_loc=camera_location)
                files.write_pcd(pcd=pcd, filename="Loaded.ply", path=self.storage_path)
                os.remove(self.server.NewMessage)
                self.server.NewMessage = None
                mesh = meshing.gen_tri_mesh(pcd)
                files.write_tri_mesh(mesh=mesh, filename="Meshed.ply", path=self.storage_path)
                os.remove(os.path.join(self.storage_path, "pcd.ply"))

            time.sleep(0.001)

    def listener(self):
        file_list = os.listdir(self.storage_path)
        file_list = [file for file in file_list if 'bytes' in file]
        if len(file_list) > 0:
            return os.path.join(self.storage_path, file_list[-1])
        else:
            return None
