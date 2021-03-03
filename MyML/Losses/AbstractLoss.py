from abc import abstractmethod
from typing import Tuple, Union

from numpy import ndarray
from numpy.typing import NBitBase

from MyML.CalcGraph.AbstractGraph import CalcGraph

LOSS_RETURN = Union[float, NBitBase]


class AbstractLoss(CalcGraph):
    @abstractmethod
    def get_learnable_parameters(self) -> ndarray:
        pass

    @abstractmethod
    def calc_grads(self, data: Tuple[ndarray, ndarray]) -> LOSS_RETURN:
        """
        :param data: (actual_value, target_value)
        :return: scalar
        """
        pass

    @abstractmethod
    def calc_forward(self, data: Tuple[ndarray, ndarray]) -> LOSS_RETURN:
        """
        :param data: (actual_value, target_value)
        :return: scalar
        """
        pass
