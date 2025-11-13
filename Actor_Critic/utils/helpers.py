import numpy as np


def one_hot(n, idx):
    v = np.zeros(n, dtype=np.float32)
    v[idx] = 1.0
    return v
