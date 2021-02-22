import numpy as np


def add_axis_if_1d(arr: np.ndarray):
    return arr[np.newaxis, :] if arr.ndim == 1 else arr
