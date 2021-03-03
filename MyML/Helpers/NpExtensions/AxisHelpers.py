import numpy as np


def add_axis_if_1d(arr: np.ndarray):
    return arr[:, None] if arr.ndim == 1 else arr
