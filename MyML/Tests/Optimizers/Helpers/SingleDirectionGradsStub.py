from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike

from MyML.CalcGraph.AbstractGraph import CalcGraph


class SingleDirectionGradsStub(CalcGraph):

    def __init__(self, shape: Sequence[int]):
        self.__parameters__ = np.ones(shape)

    def get_learnable_parameters(self) -> np.ndarray:
        return self.__parameters__

    def calc_grads(self, data: ArrayLike) -> np.ndarray:
        return np.ones(self.__parameters__.shape)

    def calc_forward(self, data: ArrayLike) -> np.ndarray:
        return np.ones(self.__parameters__.shape)
