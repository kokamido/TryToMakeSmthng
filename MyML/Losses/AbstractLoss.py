from abc import abstractmethod
from typing import Any, Tuple, Union

from numpy import ndarray
from numpy.typing import NBitBase

from MyML.CalcGraph.AbstractGraph import CalcGraph

LOSS_OUT = Union[float, NBitBase]


class AbstractLoss(CalcGraph):
    @abstractmethod
    def get_learnable_parameters(self) -> ndarray:
        pass

    @abstractmethod
    def calc_grads(self, data: Tuple[Any, Any]) -> LOSS_OUT:
        """
        :param data: (actual_value, target_value)
        :return: scalar
        """
        pass

    @abstractmethod
    def calc_forward(self, data: Tuple[Any, Any]) -> LOSS_OUT:
        """
        :param data: (actual_value, target_value)
        :return: scalar
        """
        pass
