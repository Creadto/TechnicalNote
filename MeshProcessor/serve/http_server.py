import os
import time
import threading
import subprocess
import shutil
from util.files import change_filename, load_mesh, write_tri_mesh
from multiprocessing import Process
from proc.calculating import measure_bodies2
from proc.vision import VisionProcessor
from json import dumps
from http.server import BaseHTTPRequestHandler
from util.comm_manager import HttpServer
import util.files as files
from util.yaml_config import YamlConfig
import json

from testbed import crop_n_attach


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
            check_list = ['Received', 'Loaded', 'Measured', 'Meshing', 'Meshed']
            for check in check_list:
                check_files = [file for file in file_list if check in file]
                if len(check_files) > 0:
                    status = check
                    if status == 'Measured':
                        in_file = YamlConfig.get_dict(os.path.join(path, "Measured.yaml"))
                        os.remove(os.path.join(path, "Measured.yaml"))
                        # data = dump(in_file)
                        data = json.dumps(in_file)
                    elif status == 'Front':
                        status = 'Calculating'
                    break
            if "mesh" in receive_data.lower():
                if status == 'Meshed':
                    in_file = open(os.path.join(path, "Meshed.ply"), "rb")
                    data = in_file.read()
                    self.wfile.write(data)
                    in_file.close()
                    os.remove(os.path.join(path, "Meshed.ply"))
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
    def __init__(self, config):
        ip = config['server']['ip']
        port = config['server']['port']
        if not isinstance(ip, str):
            raise Exception("Wrong IP number: Input IP number as string")
        if not isinstance(port, int):
            raise Exception("Wrong Port number: Input port number as integer")

        self.ip = ip
        self.port = port
        self.server_path = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.server_path, '../data')
        self.script_path = os.path.join(self.server_path, '../script')
        self.server = None
        self.processor = VisionProcessor(config)

        # Cleaning
        folders = os.listdir(self.data_path)
        for folder in folders:
            shutil.rmtree(os.path.join(self.data_path, folder))
            os.mkdir(os.path.join(self.data_path, folder))

        # Get configuration file

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
        file_list = os.listdir(self.server_path)
        file_list = [file for file in file_list if 'Re' in file]
        for idx, file_name in enumerate(file_list):
            load_name = "Loaded" + str(idx) + ".ply"
            loaded_path = os.path.join(self.server_path, load_name)
            files.convert_http_ply(os.path.join(self.server_path, file_name), loaded_path)
            os.remove(os.path.join(self.server_path, file_name))
            shutil.copy(loaded_path, os.path.join(self.data_path, 'pointclouds'))

        self.NewMessage = None
        change_filename(os.path.join(self.data_path, 'pointclouds'))
        pcds = files.load_pcds(os.path.join(self.data_path, 'pointclouds'), imme_remove=False)
        proc_result = self.processor.get_info_from_pcds(pcds=pcds)
        output = measure_bodies2(**proc_result)
        YamlConfig.write_yaml(os.path.join(self.server_path, './Measured.yaml'), output)
        for idx, file_name in enumerate(file_list):
            load_name = "Loaded" + str(idx) + ".ply"
            os.remove(os.path.join(self.server_path, load_name))

        f = open(os.path.join(self.server_path, 'Meshing.tat'), 'wb')
        f.close()

        # pose estimation
        subprocess.call([os.path.join(self.script_path, "pose_estimation.bat")], shell=True)
        while len(os.listdir(os.path.join(self.data_path, 'keypoints'))) == 0:
            time.sleep(0.1)
        file_list = os.listdir(os.path.join(self.data_path, 'keypoints'))
        for file in file_list:
            new_name = file.replace('000000000000_', '')
            os.rename(os.path.join(self.data_path, 'keypoints', file),
                      os.path.join(self.data_path, 'keypoints', new_name))

        # make mesh file
        subprocess.call([os.path.join(self.script_path, "mesh_maker.bat")], shell=True)
        # os.system(os.path.join(self.script_path, "mesh_maker.bat"))
        mesh_path = os.path.join(self.data_path, 'meshes', 'meshes', 'front')
        while os.path.isdir(mesh_path) is False or os.path.isfile(os.path.join(mesh_path, '000.obj')) is False:
            time.sleep(1.0)

        # read mesh file
        mesh_file = load_mesh(mesh_path, '000.obj')
        # mesh_file.paint_uniform_color(face_color * 2)
        mesh_file.paint_uniform_color([222.0 / 255.0, 171.0 / 255.0, 127.0 / 255.0])
        mesh_taubin = crop_n_attach(mesh_file, proc_result)
        write_tri_mesh(mesh_taubin, filename='Meshed.ply', path=os.path.join(self.server_path))
        os.remove(os.path.join(self.server_path, 'Meshing.tat'))

    def binder(self):
        self.server = HttpServer(ip=self.ip, port=self.port, listener=self.listener, handler=RequestHandler)

    def listener(self):
        while True:
            file_list = os.listdir(self.server_path)
            file_list = [file for file in file_list if 'Received' in file]
            if len(file_list) > 0:
                self.NewMessage = self.server_path
            else:
                self.NewMessage = None
            time.sleep(0.01)
