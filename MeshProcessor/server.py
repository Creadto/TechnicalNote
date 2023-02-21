import os
import torch
from serve.http_server import HttpProvider


def main(ip, port):
    provider = HttpProvider(ip=ip, port=port)
    provider.serve()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main("192.168.219.106", 49152)
