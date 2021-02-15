from typing import Iterable

import numpy as np

from ..CalcGraph.AbstractNode import CalcGraphNode


class LinearRegression(CalcGraphNode):

    def __init__(self, shape: Iterable[int]):
        weights_count = np.prod(shape) + 1
        self.__weights__ = np.random.normal(size=weights_count)

    def calc_grads(self, point: np.ndarray) -> np.ndarray:
        return np.concatenate((point.copy(), [1]))

    def calc_forward(self, point: np.ndarray) -> np.ndarray:
        return np.dot(self.__weights__[::-1], point) + self.__weights__[-1]

    def get_parameters(self) -> np.ndarray:
        return self.__weights__
