import os
import time
import threading
from multiprocessing import Process
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
            if "check" not in receive_data.lower():
                if status == 'Meshed':
                    in_file = open(os.path.join(path, "Meshed.ply"), "rb")
                    data = in_file.read()
                    os.remove(os.path.join(path, "Meshed.ply"))
                    self.wfile.write(data)
                    return
                else:
                    status = "Invalid"

        else:
            status = "Received"
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

        self.ip = ip
        self.port = port
        self.storage_path = os.path.dirname(os.path.abspath(__file__))
        self.server = None

        self.NewMessage = None
        self.p = Process(target=self.binder)
        self.p.start()

    def serve(self):
        t = threading.Thread(target=self.listener)
        t.start()

        run_flag = True
        while run_flag:
            if self.NewMessage is not None:
                self.mesh_seq()
            time.sleep(0.01)
        self.p.join()

    def mesh_seq(self):
        loaded_path = os.path.join(self.storage_path, "Loaded.ply")
        files.convert_http_ply(self.NewMessage, loaded_path)
        os.remove(self.NewMessage)
        self.NewMessage = None
        camera_location = files.get_pos_in_file(loaded_path)
        pcd, _ = files.load_ply(root='', filename=loaded_path, cam_loc=camera_location)
        mesh = meshing.gen_tri_mesh(pcd)
        files.write_tri_mesh(mesh=mesh, filename="Meshed.ply", path=self.storage_path)
        os.remove(os.path.join(self.storage_path, "Loaded.ply"))

    def binder(self):
        self.server = HttpServer(ip=self.ip, port=self.port, listener=self.listener, handler=RequestHandler)

    def listener(self):
        while True:
            file_list = os.listdir(self.storage_path)
            file_list = [file for file in file_list if 'Received' in file]
            if len(file_list) > 0:
                self.NewMessage = os.path.join(self.storage_path, file_list[-1])
            else:
                self.NewMessage = None
