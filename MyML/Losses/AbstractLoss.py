from abc import abstractmethod
from typing import Tuple

from numpy import ndarray
from numpy.typing import ArrayLike

from ..CalcGraph.AbstractGraph import CalcGraph


class AbstractLoss(CalcGraph):
    @abstractmethod
    def get_learnable_parameters(self) -> ndarray:
        pass

    @abstractmethod
    def calc_grads(self, data: Tuple[ArrayLike, ArrayLike]) -> ndarray:
        """
            :param data: (actual_value, target_value)
            :return: scalar
        """
        pass

    @abstractmethod
    def calc_forward(self, data: Tuple[ArrayLike, ArrayLike]) -> ndarray:
        """
            :param data: (actual_value, target_value)
            :return: scalar
        """
        pass
