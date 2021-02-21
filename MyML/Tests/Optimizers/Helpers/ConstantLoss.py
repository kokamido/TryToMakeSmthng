from typing import Tuple

import numpy as np

from MyML.Losses.AbstractLoss import AbstractLoss


class ConstantLoss(AbstractLoss):
    """
    Stub for test cases. Provides constant grads and losses.
    """

    def __init(self, grads_value: float, loss_value: float):
        self.__grads_value__ = grads_value
        self.__loss_value__ = loss_value

    def get_learnable_parameters(self) -> np.ndarray:
        """
        ConstantLoss error loss has no learnable parameters"
        :return:
        """
        raise NotImplementedError("ConstantLoss error loss has no learnable parameters")

    def calc_grads(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.ones(data[1].shape)
        """
        return np.ones(data[1].shape) * self.__grads_value__

    def calc_forward(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.ones(data[1].shape)
        """
        return np.ones(data[1].shape) * self.__loss_value__
