from typing import Sequence

import numpy as np

from MyML.CalcGraph.AbstractGraph import CalcGraph


class LinearRegression(CalcGraph):
    def __init__(self, shape: Sequence[int]):
        weights_count = np.prod(shape) + 1
        self.__all_parameters__ = np.random.normal(size=weights_count)
        self.__bias__ = self.__all_parameters__[-1:]
        self.__weights__ = self.__all_parameters__[:-1]

    def calc_grads(self, data: np.ndarray) -> np.ndarray:
        return np.concatenate((data.copy(), [1])).T

    def calc_forward(self, data: np.ndarray) -> np.ndarray:
        return np.dot(self.__weights__, data) + self.__bias__

    def get_learnable_parameters(self) -> np.ndarray:
        return self.__all_parameters__
