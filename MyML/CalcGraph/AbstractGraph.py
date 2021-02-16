from abc import ABC, abstractmethod

from numpy import ndarray
from numpy.typing import ArrayLike


class CalcGraph(ABC):
    @abstractmethod
    def get_learnable_parameters(self) -> ndarray:
        """
        :return: An array of learnable parameters of node
        """
        pass

    @abstractmethod
    def calc_grads(self, data: ArrayLike) -> ndarray:
        """
        :param data: where grad is calculated
        :return: An array of gradient values. Shape of the return value is the same as the one returned from the
        GetParameters method
        """
        pass

    @abstractmethod
    def calc_forward(self, data: ndarray) -> ndarray:
        """
        :param data: where value is calculated
        :return: An array with the results of forwarding pass
        """
        pass
