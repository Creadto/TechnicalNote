import os
import time
import threading
import cv2
from multiprocessing import Process
from proc.preprocessing import convert_img
from proc.calculating import measure_bodies
from proc.clustering import get_parts
from json import dumps
import proc.meshing as meshing
from http.server import BaseHTTPRequestHandler
from util.comm_manager import HttpServer
import util.files as files
from util.yaml_config import YamlConfig
from yaml import dump
import json


def dumper(obj):
    try:
        return obj.toJSON()
    except:
        return obj.__dict__


class RequestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.end_headers()

        path = os.path.dirname(os.path.abspath(__file__))
        memory_count = int(self.headers['Content-Length'])
        receive_data = self.rfile.read(memory_count)
        status = 'Idle'
        data = None
        if memory_count < 200:
            receive_data = receive_data.decode()
            # State check
            file_list = os.listdir(path)
            check_list = ['Meshed', 'Loaded', 'Front', 'Received', 'Measured']
            for check in check_list:
                check_files = [file for file in file_list if check in file]
                if len(check_files) > 0:
                    status = check
                    if status == 'Measured':
                        in_file = YamlConfig.get_dict(os.path.join(path, "Measured.yaml"))
                        os.remove(os.path.join(path, "Measured.yaml"))
                        #data = dump(in_file)
                        data = json.dumps(in_file)
                    elif status == 'Front':
                        status = 'Calculating'
                    break
            if "mesh" in receive_data.lower():
                if status == 'Meshed':
                    in_file = open(os.path.join(path, "Meshed.ply"), "rb")
                    data = in_file.read()
                    os.remove(os.path.join(path, "Meshed.ply"))
                    self.wfile.write(data)
                    return
                else:
                    status = "Invalid"
                    data = "Empty mesh file or Not a request"
            elif "status" in receive_data.lower():
                pass

            elif "counter" in receive_data.lower():
                sep = receive_data.split('=')
                key = sep[0]
                value = sep[1]
                f = open(os.path.join(path, value + '.obj'), 'wb')
                f.close()
            else:
                status = "Invalid"
                data = "Non-defined request"

        else:
            file_list = os.listdir(path)
            file_list = [file for file in file_list if 'obj' in file]
            os.remove(os.path.join(path, file_list[0]))
            counter = int(file_list[0].replace('.obj', ''))
            if counter > 0:
                counter -= 1
                status = "Received"
                filename = "Re"
                if counter == 0:
                    filename = status
                else:
                    f = open(os.path.join(path, str(counter) + '.obj'), 'wb')
                    f.close()
                filename += str(counter) + '.ply'
                f = open(os.path.join(path, filename), 'wb')
                f.write(receive_data)
                f.close()

            else:
                status = "Invalid"
                data = "Send number of pointcloud file(s)"

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
        pcds = []
        file_list = os.listdir(self.storage_path)
        file_list = [file for file in file_list if 'Re' in file]
        for idx, file_name in enumerate(file_list):
            load_name = "Loaded" + str(idx) + ".ply"
            loaded_path = os.path.join(self.storage_path, load_name)
            files.convert_http_ply(os.path.join(self.storage_path, file_name), loaded_path)
            os.remove(os.path.join(self.storage_path, file_name))

        self.NewMessage = None

        proc_result = {'images': dict(), 'masks': dict(), 'pcds': dict(), 'depth': dict()}
        pcds = files.load_pcds(self.storage_path)

        for name, pcd in pcds.items():
            # pcd = get_largest_cluster(pcd)
            img_rgb, depth = convert_img(pcd)
            img_bgr = img_rgb[..., ::-1]
            cv2.imwrite(os.path.join('./images', name + '.jpg'), img_bgr * 255)
            mask = get_parts(os.path.join('./images', name + '.jpg'), name)

            proc_result['images'][name] = img_rgb
            proc_result['masks'][name] = mask
            proc_result['pcds'][name] = pcd
            proc_result['depth'][name] = depth
        proc_result['res'] = 500
        proc_result['template'] = None
        output = measure_bodies(**proc_result)
        YamlConfig.write_yaml(os.path.join(self.storage_path, './Measured.yaml'), output)

        pcd = meshing.combine_pcds(pcds=pcds['Front'], down_sampling=False)
        mesh = meshing.gen_tri_mesh(pcd)
        files.write_tri_mesh(mesh=mesh, filename="Meshed.ply", path=self.storage_path)
        for idx, file_name in enumerate(file_list):
            load_name = "Loaded" + str(idx) + ".ply"
            os.remove(os.path.join(self.storage_path, load_name))

    def binder(self):
        self.server = HttpServer(ip=self.ip, port=self.port, listener=self.listener, handler=RequestHandler)

    def listener(self):
        while True:
            file_list = os.listdir(self.storage_path)
            file_list = [file for file in file_list if 'Received' in file]
            if len(file_list) > 0:
                self.NewMessage = self.storage_path
            else:
                self.NewMessage = None
