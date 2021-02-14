from abc import ABC, abstractmethod

from numpy import ndarray


class CalcGraphNode(ABC):
    @abstractmethod
    def get_parameters(self) -> ndarray:
        """
        :return: An array of learnable parameters of node
        """
        pass

    @abstractmethod
    def calc_grads(self, point: ndarray) -> ndarray:
        """
        :param point: where grad is calculated
        :return: An array of gradient values. Shape of the return value is the same as the one returned from the
        GetParameters method
        """
        pass

    @abstractmethod
    def calc_forward(self, point: ndarray) -> ndarray:
        """
        :param point: where value is calculated
        :return: An array with the results of forwarding pass
        """
        pass
