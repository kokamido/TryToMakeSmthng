from typing import Union

from numpy import ndarray
from numpy.typing import ArrayLike, NBitBase

from MyML.CalcGraph.AbstractGraph import CalcGraph


class SphereFunc(CalcGraph):
    def __init__(self, initial_point: ndarray):
        self.point = initial_point

    def get_learnable_parameters(self) -> Union[NBitBase, ArrayLike]:
        return self.point

    def calc_grads(self, _) -> ndarray:
        return 2 * self.point

    def calc_forward(self, _) -> float:
        return (self.point ** 2).sum()
