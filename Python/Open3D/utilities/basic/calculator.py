import numpy as np


def distance_vector3(a: np.array, b: np.array):
    euc = np.linalg.norm(a - b)
    return euc


if __name__ == '__main__':
    p = np.array([3, 3, 3])
    q = np.array([6, 12, 6])
    dist = distance_vector3(p, q)
    print(dist)
