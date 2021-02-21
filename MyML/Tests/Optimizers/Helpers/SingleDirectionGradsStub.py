from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from MyML.CalcGraph.AbstractGraph import CalcGraph


class SingleDirectionGradsStub(CalcGraph):
    """
       Stub for test cases. Provides constant grads and values.
    """

    def __init__(self, shape: Sequence[int], grad_value: float, forward_value: float):
        self.__parameters__ = np.ones(shape)
        self.__grad_value__ = grad_value
        self.__forward_value__ = forward_value

    def get_learnable_parameters(self) -> np.ndarray:
        return self.__parameters__

    def calc_grads(self, data: ArrayLike) -> np.ndarray:
        return np.ones(self.__parameters__.shape) * self.__grad_value__

    def calc_forward(self, data: ArrayLike) -> np.ndarray:
        return np.ones(self.__parameters__.shape) * self.__forward_value__
