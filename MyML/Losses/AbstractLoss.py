from abc import abstractmethod
from typing import Tuple

from numpy import ndarray

from MyML.CalcGraph.AbstractGraph import CalcGraph


class AbstractLoss(CalcGraph):
    @abstractmethod
    def get_learnable_parameters(self) -> ndarray:
        pass

    @abstractmethod
    def calc_grads(self, data: Tuple[ndarray, ndarray]) -> ndarray:
        """
        :param data: (actual_value, target_value)
        :return: scalar
        """
        pass

    @abstractmethod
    def calc_forward(self, data: Tuple[ndarray, ndarray]) -> ndarray:
        """
        :param data: (actual_value, target_value)
        :return: scalar
        """
        pass
