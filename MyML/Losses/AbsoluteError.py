import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from ..Losses.AbstractLoss import AbstractLoss


class AbsoluteError(AbstractLoss):

    def get_learnable_parameters(self) -> np.ndarray:
        """
        Absolute error loss has no learnable parameters"
        :return:
        """
        raise NotImplementedError("Absolute error loss has no learnable parameters")

    def calc_grads(self, data: Tuple[ArrayLike, ArrayLike]) -> np.ndarray:
        """
        :param data: (actual_value, target_value), scalars
        :return: scalar
        """
        actual_value, target_value = data
        return np.sign(target_value - actual_value)

    def calc_forward(self, data: Tuple[ArrayLike, ArrayLike]) -> np.ndarray:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.abs(actual_value - target_value)
        """
        actual_value, target_value = data
        return np.abs(actual_value - target_value)