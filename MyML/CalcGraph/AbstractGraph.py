from abc import ABC, abstractmethod
from typing import Union, Sequence, Tuple

from numpy import ndarray


class CalcGraph(ABC):
    @abstractmethod
    def get_learnable_parameters(self) -> ndarray:
        """
        :return: An array of learnable parameters of the node
        """
        pass

    @abstractmethod
    def calc_grads(self, data) -> ndarray:
        """
        :param data: where grad is calculated
        :return: An array of gradient values. Shape of the return value is the same as the one returned from the
        GetParameters method
        """
        pass

    @abstractmethod
    def calc_forward(self, data) -> ndarray:
        """
        :param data: where value is calculated
        :return: An array with the results of forwarding pass
        """
        pass
