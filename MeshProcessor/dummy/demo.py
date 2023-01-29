import torch
import tensorflow as tf
from multiprocessing import Process


def f():
    print(tf.test.is_gpu_available())


if __name__ == '__main__':
    pa = Process(target=f, args=())
    pa.start()
    pa.join()

    torch.ones(1).cuda()

    pb = Process(target=f, args=())
    pb.start()
    pb.join()
