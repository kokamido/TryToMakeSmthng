from typing import Tuple

from numpy import abs, ndarray, sign

from .AbstractLoss import AbstractLoss


class AbsoluteError(AbstractLoss):
    def get_learnable_parameters(self) -> ndarray:
        """
        Absolute error loss has no learnable parameters"
        :return:
        """
        raise NotImplementedError("Absolute error loss has no learnable parameters")

    def calc_grads(self, data: Tuple[ndarray, ndarray]) -> float:
        """
        :param data: (actual_value, target_value), scalars
        :return: scalar
        """
        actual_value, target_value = data
        return sign(actual_value - target_value)

    def calc_forward(self, data: Tuple[ndarray, ndarray]) -> float:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.abs(actual_value - target_value)
        """
        actual_value, target_value = data
        return abs(actual_value - target_value)
