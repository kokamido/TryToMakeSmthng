from typing import Tuple

from numpy import ndarray

from MyML.Losses.AbstractLoss import LOSS_OUT, AbstractLoss


class IdenticalLoss(AbstractLoss):
    def get_learnable_parameters(self) -> ndarray:
        """
        Absolute error loss has no learnable parameters"
        :return:
        """
        raise NotImplementedError("Absolute error loss has no learnable parameters")

    def calc_grads(self, data: Tuple[ndarray, ndarray]) -> LOSS_OUT:
        """
        :param data: (actual_value, target_value), scalars
        :return: scalar
        """
        return 1

    def calc_forward(self, data: Tuple[LOSS_OUT, LOSS_OUT]) -> LOSS_OUT:
        """
        :param data: (actual_value, target_value), scalars
        :return: np.abs(actual_value - target_value)
        """
        actual_value, target_value = data
        return actual_value
