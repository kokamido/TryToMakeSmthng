from typing import Any, Tuple

from numpy import ndarray

from MyML.Losses.AbstractLoss import AbstractLoss


class IdenticalLoss(AbstractLoss):
    def get_learnable_parameters(self) -> ndarray:
        """
        Absolute error loss has no learnable parameters"
        :return:
        """
        raise NotImplementedError("Absolute error loss has no learnable parameters")

    def calc_grads(self, data: Tuple[Any, Any]) -> float:
        """
        :param data: (actual_value, target_value), scalars
        :return: scalar
        """
        return 1.0

    def calc_forward(self, data: Tuple[float, float]) -> float:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.abs(actual_value - target_value)
        """
        actual_value, target_value = data
        return actual_value
