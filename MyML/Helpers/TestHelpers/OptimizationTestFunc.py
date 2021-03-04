from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NBitBase

from MyML.CalcGraph.AbstractGraph import CalcGraph


class SphereFunc(CalcGraph):
    def __init__(self, initial_point: np.ndarray):
        self.point = initial_point

    def get_learnable_parameters(self) -> Union[NBitBase, ArrayLike]:
        return self.point

    def calc_grads(self, _) -> np.ndarray:
        return 2 * self.point

    def calc_forward(self, _) -> float:
        return (self.point ** 2).sum()


class BealeFunc(CalcGraph):
    """
    https://www.sfu.ca/~ssurjano/beale.html
    """

    def __init__(self, initial_point: np.ndarray):
        assert initial_point.size == 2
        assert initial_point.ndim == 1
        self.point = initial_point

    def get_learnable_parameters(self) -> Union[NBitBase, ArrayLike]:
        return self.point

    def calc_grads(self, _) -> np.ndarray:
        x, y = self.point
        x_grad = (
            2 * x * (y ** 6 + y ** 4 - 2 * y ** 3 - y ** 2 - 2 * y + 3)
            + 5.25 * y ** 3
            + 4.5 * y ** 2
            + 3 * y
            - 12.75
        )
        y_grad = (
            6
            * x
            * (
                x * (y ** 5 + 0.666667 * y ** 3 - y ** 2 - 0.333333 * y - 0.333333)
                + 2.625 * y ** 2
                + 1.5 * y
                + 0.5
            )
        )
        return np.array((x_grad, y_grad), dtype=np.float128)

    def calc_forward(self, _) -> float:
        x, y = self.point
        return (
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * y ** 2) ** 2
            + (2.625 - x + x * y ** 3) ** 2
        )
