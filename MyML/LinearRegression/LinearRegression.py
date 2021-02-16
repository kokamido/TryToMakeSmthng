from typing import Iterable

import numpy as np

from ..CalcGraph.AbstractNode import CalcGraphNode


class LinearRegression(CalcGraphNode):

    def __init__(self, shape: Iterable[int]):
        weights_count = np.prod(shape) + 1
        self.__all_parameters__ =  np.random.normal(size=weights_count)
        self.__weights__ = self.__all_parameters__[:-1]
        self.__bias__ = self.__all_parameters__[-1]

    def calc_grads(self, point: np.ndarray) -> np.ndarray:
        return np.concatenate((point.copy(), [1]))

    def calc_forward(self, point: np.ndarray) -> np.ndarray:
        return np.dot(self.__weights__, point)+self.__bias__

    def get_parameters(self) -> np.ndarray:
        return self.__all_parameters__
