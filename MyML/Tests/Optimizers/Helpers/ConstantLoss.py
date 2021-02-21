import numpy as np
from typing import Tuple
from MyML.Losses.AbstractLoss import AbstractLoss


class ConstantLoss(AbstractLoss):

    def get_learnable_parameters(self) -> np.ndarray:
        """
        Absolute error loss has no learnable parameters"
        :return:
        """
        raise NotImplementedError("ConstantLoss error loss has no learnable parameters")

    def calc_grads(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.ones(data[1].shape)
        """
        return np.ones(data[1].shape)

    def calc_forward(self, data: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.ones(data[1].shape)
        """
        return np.ones(data[1].shape)