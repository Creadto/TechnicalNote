from numba import jit
import numpy as np


@jit(nopython=True)
def distance_matrix3(a: np.array, b: np.array):
    euc = 0.0
    for idx in range(len(a)):
        source = a[idx, :]
        eval_dist = np.linalg.norm(source - b)
        euc += eval_dist
    return euc
    # return sum((np.linalg.norm(a[idx, :] - b) for idx in range(len(a))))


@jit(nopython=True)
def distance_vector3(a: np.array, b: np.array):
    temp_array = a - b
    euc = np.linalg.norm(temp_array)
    return euc


if __name__ == '__main__':
    p = np.array([3, 3, 3])
    q = np.array([6, 12, 6])
    dist = distance_vector3(p, q)
    print(dist)
